"""
executar_divisao_ilhas.py
-------------------------
Adaptação completa do fluxo `notebooks/gebco_cluster_dvider` (`00_join.ipynb` e `01_clustering.ipynb`)
ao contexto da dissertação e do artigo Oliveira et al. (2019), "A universal approach for drainage basins".

Correções fundamentais implementadas:
  1. Identificação do oceano mundial aberto (`global_sea`) como a maior componente conexa de h <= h*,
     definindo as massas continentais como `p = ~global_sea`. Assim, depressões continentais internas
     com h <= 0 m (Salton Sea, Death Valley, planícies costeiras internas de 0 m no Delta do Mississippi,
     Mar Cáspio/Morto, polders europeus, Lago Eyre) permanecem como terra firme e NÃO geram falsos sinks
     oceânicos no interior dos continentes.
  2. Recorte da América do Norte até o Canal do Panamá (`americas_norte_central_caribe`: -125°W a -65°W,
     7.0°N a 50.0°N, cortando a leste do Canal do Panamá em 79.0°W ao sul de 12.6°N) fiel ao domínio de
     referência norte-americano.
"""

import json
import sys
import time
from pathlib import Path
import numpy as np

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from geotiff.geotiff import ReadGeoTiff, WriteGeoTiff
from calculos.percolacao_ilhas import Percolation, extrair_recorte_cluster
from calculos.geometria_esferica import pixel_area_by_latitude


MACRO_TERRITORIOS_DEF = [
    {
        "id": "americas_norte_central_caribe",
        "nome_pt": "América do Norte, Central e Caribe (até o Canal do Panamá)",
        "nome_en": "North & Central America and Caribbean",
        "lon_min": -169.0,
        "lon_max": -50.0,
        "lat_min": 7.2,
        "lat_max": 84.0,
    },
    {
        "id": "america_do_sul",
        "nome_pt": "América do Sul",
        "nome_en": "South America",
        "lon_min": -83.0,
        "lon_max": -32.0,
        "lat_min": -56.5,
        "lat_max": 12.8,
    },
    {
        "id": "europa_mediterraneo",
        "nome_pt": "Europa e Bacia do Mediterrâneo",
        "nome_en": "Europe and Mediterranean Basin",
        "lon_min": -25.0,
        "lon_max": 45.0,
        "lat_min": 34.5,
        "lat_max": 72.0,
    },
    {
        "id": "africa_madagascar",
        "nome_pt": "África e Madagascar",
        "nome_en": "Africa and Madagascar",
        "lon_min": -19.0,
        "lon_max": 52.5,
        "lat_min": -35.5,
        "lat_max": 37.5,
    },
    {
        "id": "asia_continental_arquipelagos",
        "nome_pt": "Ásia Continental, Índia e Sudeste Asiático",
        "nome_en": "Continental Asia, India and SE Asian Archipelagos",
        "lon_min": 26.0,
        "lon_max": 150.0,
        "lat_min": -11.5,
        "lat_max": 55.0,
    },
    {
        "id": "oceania_australia",
        "nome_pt": "Austrália, Nova Zelândia e Oceania",
        "nome_en": "Australia, New Zealand and Oceania",
        "lon_min": 110.0,
        "lon_max": 180.0,
        "lat_min": -48.0,
        "lat_max": 0.0,
    },
]


# Blocos hidrológicos contíguos: cada massa de terra percolada é atribuída INTEIRA ao bloco que
# contém o seu centroide (média circular em longitude), de modo que nenhum continente é cortado
# por uma borda de recorte (Panamá, Sinai, Chukotka). Antártida e ilhas árticas com centroide
# acima de 75°N ficam fora.
BLOCOS_HIDROLOGICOS_DEF = [
    {
        "id": "bloco_americas",
        "nome_pt": "Bloco Ocidental: Américas (Norte, Central, Caribe e Sul)",
        "nome_en": "Western Block: The Americas",
        "lon_min": -175.0,
        "lon_max": -25.0,
        "lat_min": -56.0,
        "lat_max": 75.0,
    },
    {
        "id": "bloco_afroeurasia_oceania",
        "nome_pt": "Bloco Oriental: Afro-Eurásia e Oceania",
        "nome_en": "Eastern Block: Afro-Eurasia and Oceania",
        "lon_min": -25.0,
        "lon_max": 180.0,
        "lat_min": -50.0,
        "lat_max": 75.0,
    },
]


def centroides_clusters(label_2d, mask_2d, lats_rows, lons_cols, areas_lat):
    """Centroide ponderado por área (lon por média circular) e área de cada cluster de terra."""
    nrows, ncols = mask_2d.shape
    rr, cc = np.nonzero(mask_2d)
    vals, inv = np.unique(label_2d[rr, cc], return_inverse=True)
    w = areas_lat[rr]
    lam = np.radians(lons_cols[cc])
    sw = np.bincount(inv, weights=w)
    lon_c = np.degrees(np.arctan2(np.bincount(inv, weights=w * np.sin(lam)),
                                  np.bincount(inv, weights=w * np.cos(lam))))
    lat_c = np.bincount(inv, weights=w * lats_rows[rr]) / sw
    return vals, lon_c, lat_c, sw


def unir_tiles_raw_se_necessario(processed_tif: Path, raw_dir: Path):
    """Replica a lógica do `00_join.ipynb` de forma flexível para qualquer resolução de tile."""
    if processed_tif.exists():
        return
    print(f"[00_join] Unindo 8 tiles de {raw_dir} -> {processed_tif}...")
    filenames = [
        [
            raw_dir / "gebco_2026_sub_ice_n90.0_s0.0_w-180.0_e-90.0_geotiff.tif",
            raw_dir / "gebco_2026_sub_ice_n90.0_s0.0_w-90.0_e0.0_geotiff.tif",
            raw_dir / "gebco_2026_sub_ice_n90.0_s0.0_w0.0_e90.0_geotiff.tif",
            raw_dir / "gebco_2026_sub_ice_n90.0_s0.0_w90.0_e180.0_geotiff.tif",
        ],
        [
            raw_dir / "gebco_2026_sub_ice_n0.0_s-90.0_w-180.0_e-90.0_geotiff.tif",
            raw_dir / "gebco_2026_sub_ice_n0.0_s-90.0_w-90.0_e0.0_geotiff.tif",
            raw_dir / "gebco_2026_sub_ice_n0.0_s-90.0_w0.0_e90.0_geotiff.tif",
            raw_dir / "gebco_2026_sub_ice_n0.0_s-90.0_w90.0_e180.0_geotiff.tif",
        ],
    ]
    info0, arr0 = ReadGeoTiff(filenames[0][0])
    h_tile, w_tile = arr0.shape
    height, width = h_tile * 2, w_tile * 4
    arr = np.empty((height, width), dtype=arr0.dtype)
    info = {
        "RasterYSize": height,
        "RasterXSize": width,
        "GeoTransform": info0["GeoTransform"],
        "Projection": info0["Projection"],
        "NoDataValue": info0["NoDataValue"],
    }
    for j, row in enumerate(filenames):
        for i, fn in enumerate(row):
            _info, _arr = ReadGeoTiff(fn)
            arr[j * h_tile : (j + 1) * h_tile, i * w_tile : (i + 1) * w_tile] = _arr

    WriteGeoTiff(info, arr, processed_tif)
    print(f"[00_join] Mosaico global salvo ({width}x{height}).")


# Mares marginais ligados ao oceano por estreitos mais finos que o pixel de 2' (~3,7 km):
# Bósforo (~0,7 km) e Dardanelos (~1,2 km) ficam "fechados" no GEBCO a 2'.
SEMENTES_MARES_MARGINAIS = {
    "Black Sea": (34.0, 43.0),
    "Sea of Azov": (36.5, 46.0),
    "Sea of Marmara": (28.0, 40.75),
}


def identificar_oceano_aberto_global(arr_global: np.ndarray, h_star: float = 0.0, geo_transform=None) -> np.ndarray:
    """
    Identifica o oceano aberto global como a maior componente conexa de h <= h_star, unida às
    componentes que contêm as sementes de mares marginais (`SEMENTES_MARES_MARGINAIS`).
    Depressões continentais internas e canais fluviais profundos com h <= h_star
    (como as fossas batimétricas do Rio Amazonas a até 1.400 km da costa, Salton Sea,
    Mar Cáspio, etc.) não conectadas ao oceano aberto permanecem como terra firme (p = ~global_sea),
    evitando a criação de falsos sinks no interior dos continentes.
    """
    is_low = arr_global <= h_star
    perc = Percolation(is_low, periodic_lon=True, periodic_poles=True)
    lbl = perc.site_percolation()
    vals, cnts = np.unique(lbl[is_low], return_counts=True)
    if vals.size == 0:
        return is_low
    ids_oceano = [vals[np.argmax(cnts)]]
    if geo_transform is not None:
        for nome, (lon, lat) in SEMENTES_MARES_MARGINAIS.items():
            r = int((lat - geo_transform[3]) / geo_transform[5])
            c = int((lon - geo_transform[0]) / geo_transform[1])
            if is_low[r, c]:
                ids_oceano.append(lbl[r, c])
                print(f"  [Oceano] Mar marginal conectado por semente: {nome} ({lon}, {lat})")
            else:
                print(f"  [Oceano] AVISO: semente de {nome} ({lon}, {lat}) não está em h <= h*")
    return np.isin(lbl, ids_oceano)


def executar_divisao_gebco(h_star: float = 0.0, top_n_ilhas: int = 15):
    t0 = time.time()
    processed_tif = THIS_DIR / "datasets" / "processed" / "gebco_2026_sub_ice_geotiff.tif"
    raw_dir = THIS_DIR / "datasets" / "raw"
    land_dir = THIS_DIR / "datasets" / "processed" / "land"
    land_dir.mkdir(parents=True, exist_ok=True)

    unir_tiles_raw_se_necessario(processed_tif, raw_dir)

    print(f"[01_clustering] Lendo grade global: {processed_tif}...")
    info_global, arr_global = ReadGeoTiff(processed_tif)
    nrows, ncols = arr_global.shape
    gt = info_global["GeoTransform"]
    print(f"[01_clustering] Grade: {ncols} x {nrows} ({arr_global.size:,} pixels) | GeoTransform={gt}")

    # 1. Identifica o oceano aberto global e preserva todas as depressões internas como terra firme
    t_perc = time.time()
    global_sea = identificar_oceano_aberto_global(arr_global, h_star=h_star, geo_transform=gt)
    p = ~global_sea  # Terra firme completa (incluindo depressões continentais h <= 0 não conectadas ao mar)

    percolation = Percolation(p, periodic_lon=True, periodic_poles=True)
    label = percolation.site_percolation()
    print(f"[01_clustering] Percolação concluída em {time.time() - t_perc:.2f}s.")

    # Áreas esféricas por linha de latitude (Oliveira et al., 2019, Eqs. 3 e 4)
    lats_rows = gt[3] + (np.arange(nrows) + 0.5) * gt[5]
    lons_cols = gt[0] + (np.arange(ncols) + 0.5) * gt[1]
    areas_lat = pixel_area_by_latitude(lats_rows, abs(gt[1]), abs(gt[5]))
    area_grid_1d = np.repeat(areas_lat, ncols)

    mask_1d = p.ravel()
    labels_validos = label.ravel()[mask_1d]
    areas_validas = area_grid_1d[mask_1d]

    vals, inv_idx, sizes = np.unique(labels_validos, return_inverse=True, return_counts=True)
    areas_clusters_km2 = np.bincount(inv_idx, weights=areas_validas)

    order = np.argsort(areas_clusters_km2)[::-1]
    vals_sorted = vals[order]
    sizes_sorted = sizes[order]
    areas_sorted = areas_clusters_km2[order]

    print(f"[01_clustering] Total de ilhas/continentes (sem falsos oceanos internos): {len(vals_sorted):,}")
    print(f"[01_clustering] Área continental total: {areas_sorted.sum():,.1f} km²")

    # 1. Exportar as top_n_ilhas maiores massas continentais/ilhas individuais da percolação
    catalogo_ilhas = []
    for rank_idx in range(min(top_n_ilhas, len(vals_sorted))):
        cid = int(vals_sorted[rank_idx])
        c_pixels = int(sizes_sorted[rank_idx])
        c_area = float(areas_sorted[rank_idx])
        cluster_mask = label == cid

        arr_crop, info_crop, bbox = extrair_recorte_cluster(arr_global, cluster_mask, info_global, pad=1)
        out_path = land_dir / f"gebco_2026_island_rank{rank_idx + 1:02d}_id{cid}.tif"
        WriteGeoTiff(info_crop, arr_crop, out_path)

        catalogo_ilhas.append({
            "rank": rank_idx + 1,
            "cluster_id": cid,
            "n_pixels": c_pixels,
            "area_esferica_km2": round(c_area, 2),
            "bbox_lon_lat": [round(x, 4) for x in bbox],
            "shape_hw": [int(info_crop["RasterYSize"]), int(info_crop["RasterXSize"])],
            "arquivo_tif": str(out_path.relative_to(REPO_ROOT)),
        })
        print(
            f"  [Ilha #{rank_idx + 1:02d}] id={cid:<10d} | Área={c_area:>12,.1f} km² "
            f"| Pixels={c_pixels:>9,d} | Shape={info_crop['RasterYSize']}x{info_crop['RasterXSize']}"
        )

    # 2. Exportar os 6 grandes territórios macro-continentais (grandes ilhas / continentes agrupados)
    catalogo_territorios = []
    for tdef in MACRO_TERRITORIOS_DEF:
        tid = tdef["id"]
        col_mask = (lons_cols >= tdef["lon_min"]) & (lons_cols <= tdef["lon_max"])
        row_mask = (lats_rows >= tdef["lat_min"]) & (lats_rows <= tdef["lat_max"])

        region_box = np.outer(row_mask, col_mask)
        territorio_mask = p & region_box

        # Refinamento na fronteira Panamá-Colômbia:
        if tid == "americas_norte_central_caribe":
            sul_mask = np.outer(lats_rows < 10.8, lons_cols > -77.2)
            territorio_mask = territorio_mask & (~sul_mask)
        elif tid == "america_do_sul":
            central_mask = np.outer(lats_rows > 7.2, lons_cols < -77.5)
            territorio_mask = territorio_mask & (~central_mask)

        arr_crop, info_crop, bbox = extrair_recorte_cluster(arr_global, territorio_mask, info_global, pad=1)
        out_path = land_dir / f"territorio_{tid}.tif"
        WriteGeoTiff(info_crop, arr_crop, out_path)

        gt_c = info_crop["GeoTransform"]
        lats_crop = gt_c[3] + (np.arange(info_crop["RasterYSize"]) + 0.5) * gt_c[5]
        areas_lat_crop = pixel_area_by_latitude(lats_crop, abs(gt_c[1]), abs(gt_c[5]))
        nodata_val = info_crop.get("NoDataValue", -32768)
        land_crop = arr_crop != nodata_val
        area_territorio_km2 = float(np.sum(land_crop * areas_lat_crop[:, np.newaxis]))
        n_pix_territorio = int(np.count_nonzero(land_crop))
        clusters_no_territorio = int(np.unique(label[territorio_mask]).size)

        catalogo_territorios.append({
            "id": tid,
            "nome_pt": tdef["nome_pt"],
            "nome_en": tdef["nome_en"],
            "n_pixels_emersos": n_pix_territorio,
            "n_ilhas_percoladas": clusters_no_territorio,
            "area_esferica_km2": round(area_territorio_km2, 2),
            "bbox_lon_lat": [round(x, 4) for x in bbox],
            "shape_hw": [int(info_crop["RasterYSize"]), int(info_crop["RasterXSize"])],
            "arquivo_tif": str(out_path.relative_to(REPO_ROOT)),
        })
        print(
            f"  [Território: {tid:<30s}] Área={area_territorio_km2:>12,.1f} km² "
            f"| Ilhas={clusters_no_territorio:>5d} | Shape={info_crop['RasterYSize']}x{info_crop['RasterXSize']}"
        )

    # 3. Exportar os 2 blocos hidrológicos contíguos (IPBA sem cortes artificiais de bacias)
    vals_c, lon_c, lat_c, area_c = centroides_clusters(label, p, lats_rows, lons_cols, areas_lat)
    catalogo_blocos = []
    for bdef in BLOCOS_HIDROLOGICOS_DEF:
        sel = (
            (lon_c >= bdef["lon_min"]) & (lon_c < bdef["lon_max"])
            & (lat_c >= bdef["lat_min"]) & (lat_c <= bdef["lat_max"])
        )
        ids_bloco = vals_c[sel]
        bloco_mask = p & np.isin(label, ids_bloco)
        arr_crop, info_crop, bbox = extrair_recorte_cluster(arr_global, bloco_mask, info_global, pad=1)
        out_path = land_dir / f"{bdef['id']}.tif"
        WriteGeoTiff(info_crop, arr_crop, out_path)
        area_bloco = float(area_c[sel].sum())
        catalogo_blocos.append({
            "id": bdef["id"],
            "nome_pt": bdef["nome_pt"],
            "nome_en": bdef["nome_en"],
            "criterio_centroide_lon_lat": [bdef["lon_min"], bdef["lat_min"], bdef["lon_max"], bdef["lat_max"]],
            "n_massas_terra": int(sel.sum()),
            "area_esferica_km2": round(area_bloco, 2),
            "bbox_lon_lat": [round(x, 4) for x in bbox],
            "shape_hw": [int(info_crop["RasterYSize"]), int(info_crop["RasterXSize"])],
            "arquivo_tif": str(out_path.relative_to(REPO_ROOT)),
        })
        print(
            f"  [Bloco: {bdef['id']:<28s}] Área={area_bloco:>12,.1f} km² | Massas de terra={int(sel.sum()):>5d} "
            f"| Shape={info_crop['RasterYSize']}x{info_crop['RasterXSize']} | bbox={[round(x, 2) for x in bbox]}"
        )

    resumo_catalogo = {
        "fonte_gebco": str(processed_tif.relative_to(REPO_ROOT)),
        "dimensoes_globais_hw": [int(nrows), int(ncols)],
        "resolucao_graus": [float(gt[1]), float(abs(gt[5]))],
        "h_star_metros": int(h_star),
        "total_clusters_globais": int(len(vals_sorted)),
        "area_emersa_global_km2": round(float(areas_sorted.sum()), 2),
        "blocos_hidrologicos": catalogo_blocos,
        "macro_territorios": catalogo_territorios,
        "top_ilhas_percoladas": catalogo_ilhas,
    }

    catalogo_path = THIS_DIR / "datasets" / "processed" / "catalogo_territorios_ilhas.json"
    with open(catalogo_path, "w", encoding="utf-8") as f:
        json.dump(resumo_catalogo, f, indent=2, ensure_ascii=False)

    print(f"[01_clustering] Catálogo salvo em {catalogo_path} ({time.time() - t0:.2f}s total).")
    return resumo_catalogo


if __name__ == "__main__":
    executar_divisao_gebco()
