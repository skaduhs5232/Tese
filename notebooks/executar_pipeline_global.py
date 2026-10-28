"""
executar_pipeline_global.py
---------------------------
Pipeline global padrão-ouro (GEBCO 2026 + GPW v4 Rev11 + NOAA Marine Microplastics Costeiro)
baseado no artigo:
  Oliveira, E. A. et al. (2019). "A universal approach for drainage basins",
  Scientific Reports, 9, 9845. https://doi.org/10.1038/s41598-019-46165-0

Arquitetura:
  1. Hidrologia em 2 blocos contíguos (`blocos_hidrologicos` do catálogo GEBCO): Américas e
     Afro-Eurásia + Oceania. O IPBA do orientador (`notebooks/IPBA`) roda sobre cada bloco inteiro,
     sem cortes artificiais de bacias (Panamá, Sinai/Nilo).
  2. Bacias endorreicas (Cáspio, Aral, Chade, Great Basin, Eyre, ...) são separadas por
     `calculos.hidrologia_endorreica` (`is_endorheic = True`) e excluídas das bacias emissoras.
  3. Demografia em 6 regiões disjuntas (cada pixel do GPW pertence a uma única região). Em cada
     região, (D*, L*) do CCA é calibrado dinamicamente pela Lei de Zipf: min |alpha - 1| com
     R² >= 0,98 na regressão rank-tamanho das 100 maiores manchas.
  4. Manchas urbanas são divididas entre bacias somando a contagem de pessoas por pixel; a população
     de uma bacia é a soma das contribuições de todas as regiões que ela cruza.
  5. NOAA: apenas amostras costeiras ('Nurdle Patrol', 'Beach', 'Ocean sediment'); 'Ocean water' é
     excluído. Só entram na correlação as grandes bacias exorreicas (P = 0,90, >= 4.500 km²) com
     estação de coleta a <= R km da foz (R = 15, 25, 38, 50 km).
  6. Cenários comparativos: A = baseline v1 (snapshot), B = hidrologia v2 + NOAA legado,
     C = hidrologia v2 + NOAA nova (resultado oficial).
"""

import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
CCA_DIR = THIS_DIR / "CCA"
IPBA_DIR = THIS_DIR / "IPBA"

for p in (REPO_ROOT, CCA_DIR, IPBA_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Importação direta dos algoritmos do orientador (notebooks/CCA e notebooks/IPBA)
from cca.city_clustering import cca_grid as cca_grid_orientador
from drainage_basins import run_ipba_grid

# Importação das formulações matemáticas/estatísticas manuais em NumPy (pacote calculos)
from calculos import (
    haversine_deg,
    pixel_area_by_latitude,
    somar_areas_por_bacia,
    identificar_oceano_e_sinks,
    calcular_densidade_gpw,
    calibrar_zipf_dinamico,
    ajuste_csn_zipf,
    dividir_manchas_e_somar_populacao_bacias,
    ajuste_escala_log_log,
    parear_fozes_e_sites,
    identificar_bacias_endorreicas,
)
from notebooks.gebco_cluster_dvider.geotiff.geotiff import ReadGeoTiff, WriteGeoTiff


CATALOGO_PATH = REPO_ROOT / "notebooks" / "gebco_cluster_dvider" / "datasets" / "processed" / "catalogo_territorios_ilhas.json"
GPW_PATH = REPO_ROOT / "dados" / "gpw_data" / "gpw_v4_population_count_rev11_2020_30_sec.tif"
NOAA_PATH = REPO_ROOT / "dados" / "noaa_microplastics.csv"
NOAA_LEGADO_PATH = REPO_ROOT / "dados" / "noaa_microplastics_legacy.csv"
MEIOS_COSTEIROS = ("Nurdle Patrol", "Beach", "Ocean sediment")
RESULTS_DIR = REPO_ROOT / "notebooks" / "results"
BASELINE_DIR = RESULTS_DIR / "_baseline_v1"

D_GRADE = np.arange(100, 2001, 100)
L_GRADE = np.arange(1000, 5001, 500)
RAIOS_KM = [15.0, 25.0, 38.0, 50.0]
RAIO_PADRAO_KM = 25.0
N_BOOT = 10_000
AREA_MIN_GRANDE_BACIA_KM2 = 4500.0
FRACAO_P90 = 0.90

UNIDADES_INFO = [
    ("nurdle_10min", "pieces/10 mins", "Beach Nurdle Patrol (pieces / 10 min)"),
    ("water_m3", "pieces/m3", "Beach Coastal Survey (pieces / m³)"),
    ("sediment_kg", "pieces kg-1 d.w.", "Coastal Sediment (pieces / kg d.w.)"),
]

# Regiões demográficas disjuntas (códigos de `codigo_regiao`) e janelas de leitura do GPW
REGIOES_DEMOGRAFICAS = [
    {"id": "americas_norte_central_caribe", "codigo": 0, "bloco": "bloco_americas",
     "nome_pt": "América do Norte, Central e Caribe", "nome_en": "North & Central America and Caribbean",
     "bbox_gpw": (-180.0, 7.0, -11.0, 84.0)},
    {"id": "america_do_sul", "codigo": 1, "bloco": "bloco_americas",
     "nome_pt": "América do Sul", "nome_en": "South America",
     "bbox_gpw": (-92.0, -56.5, -25.0, 13.0)},
    {"id": "europa_mediterraneo", "codigo": 2, "bloco": "bloco_afroeurasia_oceania",
     "nome_pt": "Europa (até os Urais)", "nome_en": "Europe (to the Urals)",
     "bbox_gpw": (-32.0, 34.0, 60.0, 84.0)},
    {"id": "africa_madagascar", "codigo": 3, "bloco": "bloco_afroeurasia_oceania",
     "nome_pt": "África e Madagascar", "nome_en": "Africa and Madagascar",
     "bbox_gpw": (-26.0, -47.0, 63.5, 38.0)},
    {"id": "asia_continental_arquipelagos", "codigo": 4, "bloco": "bloco_afroeurasia_oceania",
     "nome_pt": "Ásia (Sibéria, Oriente Médio, Índia e Sudeste Asiático)",
     "nome_en": "Asia (Siberia, Middle East, India and SE Asia)",
     "bbox_gpw": (25.0, -15.0, 180.0, 84.0)},
    {"id": "oceania_australia", "codigo": 5, "bloco": "bloco_afroeurasia_oceania",
     "nome_pt": "Austrália, Nova Zelândia e Oceania", "nome_en": "Australia, New Zealand and Oceania",
     "bbox_gpw": (110.0, -56.0, 180.0, 0.5)},
]
REGIAO_POR_CODIGO = {r["codigo"]: r for r in REGIOES_DEMOGRAFICAS}

# Divisor marítimo Europa/África no Mediterrâneo (lat em função da lon)
_MED_LON = np.array([-26.0, -5.6, -2.0, 0.0, 3.0, 11.0, 12.0, 26.0, 36.0])
_MED_LAT = np.array([35.9, 35.95, 36.0, 37.0, 38.0, 37.6, 35.0, 33.6, 33.6])

# Áreas de referência da literatura (km²) e foz aproximada (lon, lat) para validação do IPBA
BACIAS_REFERENCIA = {
    "Amazon": (6.1e6, -50.5, -0.2),
    "Congo": (3.7e6, 12.4, -6.0),
    "Nile": (3.25e6, 31.0, 31.5),
    "Mississippi": (3.2e6, -89.3, 29.1),
    "Parana / La Plata": (3.1e6, -57.8, -34.5),
    "Ob": (2.99e6, 70.0, 66.5),
    "Yenisei": (2.58e6, 83.0, 71.5),
    "Lena": (2.49e6, 127.5, 72.8),
    "Niger": (2.27e6, 5.8, 4.9),
    "Amur": (1.86e6, 141.0, 52.5),
    "Yangtze": (1.8e6, 121.5, 31.4),
    "Mackenzie": (1.8e6, -135.5, 69.0),
    "Ganges-Brahmaputra": (1.73e6, 90.6, 22.5),
    "Zambezi": (1.39e6, 36.3, -18.8),
    "Indus": (1.12e6, 67.5, 24.0),
    "Nelson": (1.07e6, -92.6, 57.0),
    "Murray-Darling": (1.06e6, 139.0, -35.6),
    "St. Lawrence": (1.03e6, -69.5, 47.5),
    "Orange": (0.97e6, 16.5, -28.6),
    "Tocantins": (0.95e6, -49.0, -1.8),
    "Orinoco": (0.88e6, -61.0, 8.6),
    "Yukon": (0.83e6, -164.8, 62.8),
    "Danube": (0.80e6, 29.6, 45.2),
    "Mekong": (0.80e6, 106.5, 9.7),
    "Yellow River": (0.75e6, 119.0, 37.7),
    "Columbia": (0.67e6, -124.0, 46.25),
    "Colorado": (0.64e6, -114.8, 31.8),
    "Sao Francisco": (0.64e6, -36.4, -10.5),
    "Pearl": (0.45e6, 113.6, 22.4),
    "Rhine": (0.185e6, 4.1, 51.95),
}


# ==============================================================================
# LEITURA DE DADOS
# ==============================================================================

def ler_janela_gpw(gpw_path: Path, bbox_lon_lat: tuple, fator_agregacao: int = 2):
    """
    Lê a janela correspondente a `bbox_lon_lat = (lon_min, lat_min, lon_max, lat_max)`
    diretamente do GeoTIFF global do GPW v4 (`population_count`, contagem de habitantes por pixel).
    Quando `fator_agregacao > 1`, soma exatamente os habitantes dos blocos (fator x fator).
    """
    lon_min, lat_min, lon_max, lat_max = bbox_lon_lat
    ds = gdal.Open(str(gpw_path), gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    ncols_full = ds.RasterXSize
    nrows_full = ds.RasterYSize
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()

    dx = gt[1]
    dy = abs(gt[5])

    x0 = int(np.clip(np.floor((lon_min - gt[0]) / dx + 1e-6), 0, ncols_full - 1))
    x1 = int(np.clip(np.ceil((lon_max - gt[0]) / dx - 1e-6), x0 + 1, ncols_full))
    y0 = int(np.clip(np.floor((gt[3] - lat_max) / dy + 1e-6), 0, nrows_full - 1))
    y1 = int(np.clip(np.ceil((gt[3] - lat_min) / dy - 1e-6), y0 + 1, nrows_full))

    win_w = x1 - x0
    win_h = y1 - y0
    if fator_agregacao > 1:
        win_w = (win_w // fator_agregacao) * fator_agregacao
        win_h = (win_h // fator_agregacao) * fator_agregacao

    arr = band.ReadAsArray(x0, y0, win_w, win_h)
    invalido = (arr < 0) | (~np.isfinite(arr))
    if nodata is not None:
        invalido |= arr == nodata
    arr[invalido] = 0
    del invalido

    lon_origin = gt[0] + x0 * dx
    lat_origin = gt[3] - y0 * dy

    if fator_agregacao > 1:
        f = int(fator_agregacao)
        new_h = win_h // f
        new_w = win_w // f
        arr = arr.reshape(new_h, f, new_w, f).sum(axis=(1, 3), dtype=np.float64)
        dx_out = dx * f
        dy_out = dy * f
    else:
        arr = arr.astype(np.float64)
        new_h, new_w = win_h, win_w
        dx_out, dy_out = dx, dy

    lats_centers = lat_origin - (np.arange(new_h) + 0.5) * dy_out
    lons_centers = lon_origin + (np.arange(new_w) + 0.5) * dx_out
    gt_out = (lon_origin, dx_out, 0.0, lat_origin, 0.0, -dy_out)

    del band
    del ds
    return arr, lats_centers, lons_centers, gt_out


def carregar_noaa_limpo(noaa_path: Path, apenas_costeiro: bool = True) -> pd.DataFrame:
    """
    Carrega e higieniza a base de microplásticos da NOAA.
    Aceita o CSV padronizado gerado por `dados/ingestao_noaa_geojson.py` (colunas `lat`, `lon`,
    `meio`, `unidade`, `medida`) e o CSV legado do FeatureServer (`Latitude (degree)`, `Marine Setting`, ...).
    Se `apenas_costeiro=True` (padrão), exclui todos os pontos de mar aberto ('Ocean water'),
    mantendo apenas 'Nurdle Patrol', 'Beach' e 'Ocean sediment'.
    """
    df = pd.read_csv(noaa_path, low_memory=False)
    if "medida" not in df.columns:
        m_main = pd.to_numeric(df.get("Microplastics Measurement"), errors="coerce")
        if "Standardized Nurdle  Amount" in df.columns:
            m_nurdle = pd.to_numeric(df["Standardized Nurdle  Amount"], errors="coerce")
            m_main = m_main.fillna(m_nurdle)
        df["medida"] = m_main
        df = df.rename(
            columns={
                "Latitude (degree)": "lat",
                "Longitude (degree)": "lon",
                "Unit": "unidade",
                "Marine Setting": "meio",
                "Ocean": "oceano",
                "Region": "regiao",
                "Subregion": "subregiao",
                "Country": "pais",
                "State": "estado",
                "Beach Location": "local_praia",
            }
        )
    df["meio"] = df["meio"].astype(str).str.strip().replace({"Beach-Nurdle Patrol": "Nurdle Patrol"})
    df["unidade"] = df["unidade"].astype(str).str.strip()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["medida"] = pd.to_numeric(df["medida"], errors="coerce")
    df = df.dropna(subset=["lat", "lon", "medida", "unidade"]).copy()
    df = df[df["medida"] >= 0].copy()

    if apenas_costeiro:
        df = df[df["meio"].isin(MEIOS_COSTEIROS)].copy()

    return df


def agrupar_sites_noaa(df_sub: pd.DataFrame, casas_decimais: int = 4) -> pd.DataFrame:
    """Agrupa coletas repetidas na mesma coordenada (arredondada para ~11 m com 4 casas)."""
    if df_sub.empty:
        return pd.DataFrame(
            columns=["site_id", "lat", "lon", "n_amostras", "soma_medida", "media_medida", "mediana_medida", "max_medida"]
        )
    df_c = df_sub.copy()
    df_c["lat_r"] = df_c["lat"].round(casas_decimais)
    df_c["lon_r"] = df_c["lon"].round(casas_decimais)
    grp = (
        df_c.groupby(["lat_r", "lon_r"], as_index=False)
        .agg(
            lat=("lat", "mean"),
            lon=("lon", "mean"),
            n_amostras=("medida", "size"),
            soma_medida=("medida", "sum"),
            media_medida=("medida", "mean"),
            mediana_medida=("medida", "median"),
            max_medida=("medida", "max"),
        )
        .reset_index(drop=True)
    )
    grp["site_id"] = np.arange(1, len(grp) + 1, dtype=np.int32)
    return grp


# ==============================================================================
# REGIÕES DEMOGRÁFICAS DISJUNTAS
# ==============================================================================

def codigo_regiao(lat, lon) -> np.ndarray:
    """
    Código da região demográfica (0..5, ver `REGIOES_DEMOGRAFICAS`) de cada ponto. Regras em ordem
    de prioridade: América do Sul (Colômbia/Panamá em 77,35°W), Açores, Chukotka, Polinésia,
    resto do hemisfério ocidental e Groenlândia; no hemisfério oriental, África ao sul do divisor
    do Mediterrâneo e a oeste do Suez/Mar Vermelho, Oceania, Europa a oeste dos Urais e ao norte do
    Cáucaso/Anatólia, e Ásia no restante.
    """
    lat, lon = np.broadcast_arrays(np.asarray(lat, dtype=np.float64), np.asarray(lon, dtype=np.float64))
    cod = np.full(lat.shape, -1, dtype=np.int8)

    def atribui(mask, c):
        cod[mask & (cod == -1)] = c

    oeste = lon < -25.0
    groenlandia = (lon >= -25.0) & (lon < -11.0) & (lat > 67.5)
    atribui((lat <= 12.8) & (lat >= -60.0) & (lon >= -92.0) & oeste & ((lon >= -77.35) | (lat < 7.2)), 1)
    atribui((lon >= -32.0) & (lon < -24.0) & (lat > 36.0) & (lat < 40.5), 2)
    atribui((lon < -168.5) & (lat > 62.0), 4)
    atribui(oeste & (lon < -120.0) & (lat < 7.2), 5)
    atribui(oeste | groenlandia, 0)

    lat_med = np.interp(lon, _MED_LON, _MED_LAT)
    lon_mar_vermelho = np.where(lat >= 30.0, 32.35, 32.5 + (30.0 - lat) * (43.3 - 32.5) / (30.0 - 12.6))
    africa = (
        (lat >= -47.0) & (lat < 38.0) & (lat < lat_med) & (lon >= -26.0) & (lon <= 63.5)
        & ((lat < 12.6) | (lon <= lon_mar_vermelho))
    )
    atribui(africa, 3)
    atribui(((lat <= 0.0) & ((lon >= 141.0) | ((lon >= 110.0) & (lat < -11.5)))) | ((lat < -15.0) & (lon > 63.5)), 5)
    asia = (
        ((lon >= 26.0) & (lat < 41.3)) | ((lon >= 37.0) & (lat < 44.8))
        | ((lon >= 45.0) & (lat < 47.0)) | (lon >= 60.0)
    )
    atribui(~asia & (lat >= 34.0), 2)
    atribui(np.ones(lat.shape, dtype=bool), 4)
    return cod


def normalizar_lon(lon):
    return ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0


def mapear_grade_bloco_para_gpw(grade_bloco, gt_bloco, lats_gpw, lons_gpw):
    """Rótulo da grade do bloco (DEM 2') no centro de cada pixel GPW (1'), com longitudes > 180° no bloco."""
    h, w = grade_bloco.shape
    lon_w = np.where(lons_gpw < gt_bloco[0], lons_gpw + 360.0, lons_gpw)
    cols = np.floor((lon_w - gt_bloco[0]) / gt_bloco[1]).astype(np.int64)
    rows = np.floor((lats_gpw - gt_bloco[3]) / gt_bloco[5]).astype(np.int64)
    ok_c = (cols >= 0) & (cols < w)
    ok_r = (rows >= 0) & (rows < h)
    out = grade_bloco[np.clip(rows, 0, h - 1)[:, None], np.clip(cols, 0, w - 1)[None, :]]
    out[~ok_r, :] = 0
    out[:, ~ok_c] = 0
    return np.ascontiguousarray(out, dtype=np.int32)


# ==============================================================================
# HIDROLOGIA POR BLOCO (IPBA + ENDORREISMO)
# ==============================================================================

def processar_bloco_hidrologico(bloco: dict, sensibilidade: bool):
    bid = bloco["id"]
    t0 = time.time()
    info, dem = ReadGeoTiff(REPO_ROOT / bloco["arquivo_tif"])
    gt = info["GeoTransform"]
    nodata = float(info["NoDataValue"])
    nrows, ncols = dem.shape
    lats = gt[3] + (np.arange(nrows) + 0.5) * gt[5]
    lons = gt[0] + (np.arange(ncols) + 0.5) * gt[1]
    row_areas = pixel_area_by_latitude(lats, abs(gt[1]), abs(gt[5]))

    sea_mask, sinks_grid, n_sinks = identificar_oceano_e_sinks(dem.astype(np.float64), nodata=nodata, h_star=0.0)
    basin_grid = run_ipba_grid(dem, sinks_grid, sea_mask=sea_mask, nodata=nodata, seed=42, ispbc=False)
    print(f"  [IPBA {bid}] {dem.shape[0]}x{dem.shape[1]} px | {n_sinks:,} sinks costeiros | {time.time() - t0:.1f}s")

    endo, depressoes, diag = identificar_bacias_endorreicas(dem, sea_mask, lats, lons, row_areas)
    print(
        f"  [Endorreismo {bid}] {diag['n_candidatas']} depressões candidatas -> {diag['n_endorreicas_aceitas']} "
        f"bacias endorreicas documentadas ({diag['area_endorreica_total_km2']:,.0f} km²) | "
        f"sem correspondência: {diag['sistemas_sem_correspondencia']}"
    )

    linhas_sens = []
    if sensibilidade:
        for a_min, p_min in [(500.0, 20.0), (5000.0, 20.0), (1000.0, 10.0), (1000.0, 50.0), (1000.0, 20.0)]:
            e_s, _, d_s = identificar_bacias_endorreicas(dem, sea_mask, lats, lons, row_areas,
                                                         area_min_km2=a_min, prof_min_m=p_min)
            linhas_sens.append({
                "bloco_id": bid, "area_min_km2": a_min, "prof_min_m": p_min,
                "n_candidatas": d_s["n_candidatas"], "n_endorreicas": d_s["n_endorreicas_aceitas"],
                "area_endorreica_km2": d_s["area_endorreica_total_km2"],
                "sistemas_sem_correspondencia": ";".join(d_s["sistemas_sem_correspondencia"]),
            })
            del e_s

    # Grade combinada: exorreicas = id do sink costeiro; endorreicas = n_sinks + k; mar = 0
    endo_offset = int(n_sinks) + 1
    G = np.where(endo > 0, endo_offset + endo, basin_grid).astype(np.int32)
    G[sea_mask] = 0
    ids, area_px, area_km2 = somar_areas_por_bacia(G, row_areas)
    ids0, _, area0 = somar_areas_por_bacia(np.where(sea_mask, 0, basin_grid).astype(np.int32), row_areas)
    area_ipba = dict(zip(ids0.tolist(), area0.tolist()))

    sink_rows, sink_cols = np.nonzero(sinks_grid > 0)
    sink_vals = sinks_grid[sink_rows, sink_cols]
    foz_lon = np.zeros(endo_offset, dtype=np.float64)
    foz_lat = np.zeros(endo_offset, dtype=np.float64)
    foz_lon[sink_vals] = lons[sink_cols]
    foz_lat[sink_vals] = lats[sink_rows]

    ex = ids < endo_offset
    df_exo = pd.DataFrame({
        "bloco_id": bid,
        "grid_id": ids[ex],
        "basin_id": ids[ex],
        "is_endorheic": False,
        "nome_sistema_endorreico": "",
        "foz_lon": normalizar_lon(foz_lon[ids[ex]]),
        "foz_lat": foz_lat[ids[ex]],
        "area_px": area_px[ex],
        "area_km2": area_km2[ex],
        "area_ipba_sem_filtro_km2": [area_ipba.get(int(i), np.nan) for i in ids[ex]],
    }).sort_values("area_km2", ascending=False).reset_index(drop=True)
    frac = np.cumsum(df_exo["area_km2"].to_numpy()) / max(1e-12, df_exo["area_km2"].sum())
    df_exo["frac_area_acum"] = frac
    df_exo["in_p90"] = (frac <= FRACAO_P90) & (df_exo["area_km2"] >= AREA_MIN_GRANDE_BACIA_KM2)

    dep_por_id = {d["endo_id"]: d for d in depressoes}
    en_ids = ids[~ex]
    df_endo = pd.DataFrame({
        "bloco_id": bid,
        "grid_id": en_ids,
        "basin_id": -(en_ids - endo_offset),
        "is_endorheic": True,
        "nome_sistema_endorreico": [dep_por_id[int(i - endo_offset)]["nome_sistema"] for i in en_ids],
        "foz_lon": [dep_por_id[int(i - endo_offset)]["pit_lon"] for i in en_ids],
        "foz_lat": [dep_por_id[int(i - endo_offset)]["pit_lat"] for i in en_ids],
        "area_px": area_px[~ex],
        "area_km2": area_km2[~ex],
        "area_ipba_sem_filtro_km2": np.nan,
        "frac_area_acum": np.nan,
        "in_p90": False,
        "pit_elev_m": [dep_por_id[int(i - endo_offset)]["pit_elev_m"] for i in en_ids],
        "nivel_transbordo_m": [dep_por_id[int(i - endo_offset)]["nivel_transbordo_m"] for i in en_ids],
        "profundidade_m": [dep_por_id[int(i - endo_offset)]["profundidade_m"] for i in en_ids],
    })

    # Raster de saída: exorreicas > 0, endorreicas < 0, mar 0
    raster = np.where(G >= endo_offset, -(G - endo_offset), G).astype(np.int32)
    info_out = dict(info)
    info_out["NoDataValue"] = None
    WriteGeoTiff(info_out, raster, RESULTS_DIR / f"bacias_{bid}_ipba.tif", gdal_dtype=gdal.GDT_Int32)
    print(f"  [Bloco {bid}] concluído em {time.time() - t0:.1f}s")

    return dict(
        id=bid, info=info_out, gt=gt, G=G, raster=raster, endo_offset=endo_offset,
        df_exo=df_exo, df_endo=df_endo, diag=diag, sens=linhas_sens,
    )


def exportar_recortes_regionais(blocos_proc: dict):
    """Recorta o raster de bacias de cada bloco na janela de cada região (compatibilidade com as figuras)."""
    for reg in REGIOES_DEMOGRAFICAS:
        b = blocos_proc[reg["bloco"]]
        gt = b["gt"]
        h, w = b["raster"].shape
        lon_min, lat_min, lon_max, lat_max = reg["bbox_gpw"]
        c0 = int(np.clip(np.floor((lon_min - gt[0]) / gt[1]), 0, w - 1))
        c1 = int(np.clip(np.ceil((lon_max - gt[0]) / gt[1]), c0 + 1, w))
        r0 = int(np.clip(np.floor((lat_max - gt[3]) / gt[5]), 0, h - 1))
        r1 = int(np.clip(np.ceil((lat_min - gt[3]) / gt[5]), r0 + 1, h))
        info = dict(b["info"])
        info["RasterXSize"] = c1 - c0
        info["RasterYSize"] = r1 - r0
        info["GeoTransform"] = (gt[0] + c0 * gt[1], gt[1], 0.0, gt[3] + r0 * gt[5], 0.0, gt[5])
        WriteGeoTiff(info, b["raster"][r0:r1, c0:c1], RESULTS_DIR / f"bacias_{reg['id']}_ipba.tif",
                     gdal_dtype=gdal.GDT_Int32)


# ==============================================================================
# DEMOGRAFIA POR REGIÃO (ZIPF DINÂMICO + CCA + DIVISÃO DE MANCHAS)
# ==============================================================================

def calibrar_regiao(reg: dict, pop_gpw, lats_gpw, dlon, dlat, reusar: bool):
    rid = reg["id"]
    json_path = RESULTS_DIR / f"cca_calibracao_{rid}.json"
    npz_path = RESULTS_DIR / f"zipf_matriz_{rid}.npz"
    if reusar and json_path.exists() and npz_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            cal_json = json.load(f)
        if cal_json.get("metodo") == "zipf_dinamico_ols_r2":
            print(f"  [Zipf {rid}] reutilizando calibração salva (D*={cal_json['D_star_hab_km2']}, L*={cal_json['l_metros']} m)")
            return cal_json, None

    t0 = time.time()
    cal = calibrar_zipf_dinamico(pop_gpw, lats_gpw, dlon, dlat, D_grade=D_GRADE, L_grade=L_GRADE)
    print(
        f"  [Zipf {rid}] varredura {len(L_GRADE)}x{len(D_GRADE)} em {time.time() - t0:.1f}s -> "
        f"D*={cal['D_star']} hab/km², L*={cal['l_metros']} m | alpha={cal['alpha_ols']:.3f} R²={cal['r2_ols']:.4f} "
        f"| restrição R²>=0,98 atendida: {cal['restricao_r2_atendida']}"
    )
    return cal, cal["matriz"]


def processar_regiao_demografica(reg: dict, blocos_proc: dict, reusar_zipf: bool):
    rid = reg["id"]
    t0 = time.time()
    b = blocos_proc[reg["bloco"]]
    pop_gpw, lats_gpw, lons_gpw, gt_gpw = ler_janela_gpw(GPW_PATH, reg["bbox_gpw"], fator_agregacao=2)
    dlon = abs(gt_gpw[1])
    dlat = abs(gt_gpw[5])
    for i0 in range(0, pop_gpw.shape[0], 512):
        sl = slice(i0, i0 + 512)
        fora = codigo_regiao(lats_gpw[sl, None], lons_gpw[None, :]) != reg["codigo"]
        pop_gpw[sl][fora] = 0.0
    _, row_areas_gpw = calcular_densidade_gpw(pop_gpw, lats_gpw, dlon, dlat)
    pop_total_regiao = float(pop_gpw.sum())

    cal, matriz = calibrar_regiao(reg, pop_gpw, lats_gpw, dlon, dlat, reusar_zipf)
    D_star = int(cal["D_star"] if "D_star" in cal else cal["D_star_hab_km2"])
    l_metros = int(cal["l_metros"])

    cidades_grid, pop_cidades, area_cidades = cca_grid_orientador(
        pop_gpw, lons_gpw, lats_gpw, row_areas_gpw[:, None], float(D_star), l_metros / 1000.0
    )
    csn = ajuste_csn_zipf(pop_cidades, min_cauda=25)

    if matriz is not None:
        np.savez_compressed(
            RESULTS_DIR / f"zipf_matriz_{rid}.npz",
            D=D_GRADE, l=L_GRADE, D_star=D_star, l_metros=l_metros, L_MIN=cal["L_MIN"],
            pop_cidades=pop_cidades, area_cidades=area_cidades, **matriz,
        )
        cal_json = {
            "territorio_id": rid,
            "nome_pt": reg["nome_pt"],
            "nome_en": reg["nome_en"],
            "bloco_hidrologico": reg["bloco"],
            "bbox": list(reg["bbox_gpw"]),
            "metodo": "zipf_dinamico_ols_r2",
            "criterio": "min |alpha_OLS(N=100) - 1| sujeito a R2 >= 0.98 e L >= L_min",
            "D_grade": [int(D_GRADE.min()), int(D_GRADE.max()), int(D_GRADE[1] - D_GRADE[0])],
            "L_grade": [int(L_GRADE.min()), int(L_GRADE.max()), int(L_GRADE[1] - L_GRADE[0])],
            "D_star_hab_km2": D_star,
            "l_metros": l_metros,
            "l_km": l_metros / 1000.0,
            "L_MIN_metros": cal["L_MIN"],
            "alpha_ols": cal["alpha_ols"],
            "r2_ols": cal["r2_ols"],
            "alpha_ols_N": cal["alpha_ols_N"],
            "r2_ols_N": cal["r2_ols_N"],
            "restricao_r2_atendida": cal["restricao_r2_atendida"],
            "alpha_hill": cal["alpha_hill"],
            "delta_alpha": cal["delta"],
            "n_manchas_urbanas": int(len(pop_cidades)),
            "populacao_urbana_total_hab": float(pop_cidades.sum()),
            "populacao_territorio_total_hab": pop_total_regiao,
            "maior_aglomeracao_hab": float(pop_cidades.max()) if pop_cidades.size else 0.0,
            "csn_zipf": csn,
        }
        if rid == "oceania_australia":
            cal_json["nota_macrocefalia"] = (
                "A Austrália concentra a população em 5 capitais costeiras (Sydney, Melbourne, Brisbane, "
                "Perth, Adelaide) com interior desértico; a curva rank-tamanho é mais íngreme que Zipf "
                "(alpha > 1), fenômeno de macrocefalia urbana documentado na literatura."
            )
        with open(RESULTS_DIR / f"cca_calibracao_{rid}.json", "w", encoding="utf-8") as f:
            json.dump(cal_json, f, indent=2, ensure_ascii=False)
    else:
        cal_json = cal
        cal_json["n_manchas_urbanas"] = int(len(pop_cidades))
        cal_json["populacao_urbana_total_hab"] = float(pop_cidades.sum())
        cal_json["populacao_territorio_total_hab"] = pop_total_regiao
        cal_json["maior_aglomeracao_hab"] = float(pop_cidades.max()) if pop_cidades.size else 0.0
        cal_json["csn_zipf"] = csn

    G_gpw = mapear_grade_bloco_para_gpw(b["G"], b["gt"], lats_gpw, lons_gpw)
    ids_todos = np.arange(int(b["G"].max()) + 1, dtype=np.int64)
    _, pop_total, pop_urbana, pedacos = dividir_manchas_e_somar_populacao_bacias(
        cidades_grid=cidades_grid,
        basin_grid_dem=G_gpw,
        pop_count=pop_gpw,
        row_areas_gpw_km2=row_areas_gpw,
        major_basin_ids=ids_todos,
        max_realoc_px=6,
    )
    if pedacos["mancha"].size > 0:
        sel = pedacos["basin_id"] > 0
        _, cnt = np.unique(pedacos["mancha"][sel], return_counts=True)
        n_div = int(np.sum(cnt > 1))
    else:
        n_div = 0
    print(
        f"  [CCA {rid}] {len(pop_cidades):,} manchas (maior {pop_cidades.max() if pop_cidades.size else 0:,.0f} hab) | "
        f"pop região {pop_total_regiao / 1e6:,.1f} M | manchas divididas entre bacias: {n_div:,} | "
        f"fora de bacias: {pedacos['hab_fora_bacias'] / 1e6:,.2f} M | {time.time() - t0:.1f}s"
    )
    df_ped = pd.DataFrame({
        "territorio_id": rid,
        "bloco_id": reg["bloco"],
        "mancha_id": pedacos["mancha"],
        "grid_id": pedacos["basin_id"],
        "pop_pedaco_hab": pedacos["pop"],
        "area_pedaco_km2": pedacos["area_km2"],
        "fracao_pop_mancha": pedacos["fracao_pop"],
    })
    df_ped = df_ped[df_ped["grid_id"] > 0]
    resumo = dict(n_manchas_divididas=n_div, hab_fora_bacias=pedacos["hab_fora_bacias"],
                  hab_realocados=pedacos["hab_realocados"])
    return cal_json, pop_total, pop_urbana, df_ped, resumo


# ==============================================================================
# PAREAMENTO NOAA E ALOMETRIA
# ==============================================================================

def parear_bacias_noaa(df_major: pd.DataFrame, df_noaa: pd.DataFrame, dlon: float, dlat: float) -> pd.DataFrame:
    df = df_major.copy().reset_index(drop=True)
    for un_slug, un_raw, _ in UNIDADES_INFO:
        sites = agrupar_sites_noaa(df_noaa[df_noaa["unidade"] == un_raw])
        if sites.empty:
            df[f"dist_km_{un_slug}"] = np.inf
            df[f"site_id_{un_slug}"] = -1
            for c in ("site_lon", "site_lat"):
                df[f"{c}_{un_slug}"] = np.nan
            for c in ("n_amostras", "soma", "media", "mediana", "max", "qtd_x_area"):
                df[f"{c}_{un_slug}"] = 0.0
            continue
        idx, dist, _ = parear_fozes_e_sites(
            sink_ids=df["basin_id"].to_numpy(),
            sink_lons=df["foz_lon"].to_numpy(),
            sink_lats=df["foz_lat"].to_numpy(),
            dlon_deg=dlon,
            dlat_deg=dlat,
            site_lons=sites["lon"].to_numpy(),
            site_lats=sites["lat"].to_numpy(),
        )
        m = sites.iloc[idx].reset_index(drop=True)
        df[f"dist_km_{un_slug}"] = dist
        df[f"site_id_{un_slug}"] = m["site_id"].to_numpy()
        df[f"site_lon_{un_slug}"] = m["lon"].to_numpy()
        df[f"site_lat_{un_slug}"] = m["lat"].to_numpy()
        df[f"n_amostras_{un_slug}"] = m["n_amostras"].to_numpy()
        df[f"soma_{un_slug}"] = m["soma_medida"].to_numpy()
        df[f"media_{un_slug}"] = m["media_medida"].to_numpy()
        df[f"mediana_{un_slug}"] = m["mediana_medida"].to_numpy()
        df[f"max_{un_slug}"] = m["max_medida"].to_numpy()
        df[f"qtd_x_area_{un_slug}"] = m["media_medida"].to_numpy() * df["area_km2"].to_numpy()
    return df


def _limpar_fit(fit: dict) -> dict:
    return {k: v for k, v in fit.items() if k not in ("xs", "ys", "banda")}


def executar_alometria(df_par: pd.DataFrame, tag: str, n_boot: int = N_BOOT, verbose: bool = True):
    """Correlações por unidade e raio + ajuste combinado normalizado pela mediana de cada unidade."""
    metricas = {"por_unidade_e_raio": {}}
    linhas = []
    if verbose:
        print(f"\n  [Alometria | {tag}]")
    for un_slug, un_raw, un_label in UNIDADES_INFO:
        metricas["por_unidade_e_raio"][un_slug] = {}
        for r_busca in RAIOS_KM:
            sub = df_par[
                (df_par[f"dist_km_{un_slug}"] <= r_busca)
                & (df_par[f"n_amostras_{un_slug}"] >= 1)
                & (df_par["pop_total"] > 0)
                & (df_par[f"media_{un_slug}"] > 0)
            ].copy()
            if not sub.empty:
                sub = (
                    sub.sort_values([f"dist_km_{un_slug}", "area_km2"], ascending=[True, False])
                    .drop_duplicates(subset=[f"site_id_{un_slug}"], keep="first")
                    .sort_values("pop_total", ascending=False)
                    .reset_index(drop=True)
                )
            if r_busca == RAIO_PADRAO_KM:
                for _, row in sub.iterrows():
                    linhas.append({
                        "unidade_slug": un_slug,
                        "unidade": un_raw,
                        "unidade_label_en": un_label,
                        "raio_busca_km": r_busca,
                        "bloco_id": row["bloco_id"],
                        "territorio_id": row["territorio_id"],
                        "territorio_nome_en": row["territorio_nome_en"],
                        "global_basin_uid": row["global_basin_uid"],
                        "basin_id": int(row["basin_id"]),
                        "in_p90": bool(row["in_p90"]),
                        "foz_lon": float(row["foz_lon"]),
                        "foz_lat": float(row["foz_lat"]),
                        "site_id": int(row[f"site_id_{un_slug}"]),
                        "site_lon": float(row[f"site_lon_{un_slug}"]),
                        "site_lat": float(row[f"site_lat_{un_slug}"]),
                        "dist_foz_km": float(row[f"dist_km_{un_slug}"]),
                        "area_km2": float(row["area_km2"]),
                        "pop_total": float(row["pop_total"]),
                        "pop_urbana": float(row["pop_urbana"]),
                        "pop_rural": float(row["pop_rural"]),
                        "n_manchas_na_bacia": int(row["n_manchas_na_bacia"]),
                        "n_amostras_noaa": int(row[f"n_amostras_{un_slug}"]),
                        "mp_soma": float(row[f"soma_{un_slug}"]),
                        "mp_media": float(row[f"media_{un_slug}"]),
                        "mp_mediana": float(row[f"mediana_{un_slug}"]),
                        "mp_max": float(row[f"max_{un_slug}"]),
                        "mp_x_area": float(row[f"qtd_x_area_{un_slug}"]),
                    })
            res = {}
            for x_col in ["pop_total", "pop_urbana", "area_km2"]:
                for y_col in [f"soma_{un_slug}", f"media_{un_slug}", f"max_{un_slug}", f"qtd_x_area_{un_slug}"]:
                    y_short = "qtd_x_area" if y_col.startswith("qtd_x_area_") else y_col.split("_")[0]
                    chave = f"{x_col}__vs__{y_short}"
                    v = sub[(sub[x_col] > 0) & (sub[y_col] > 0)] if not sub.empty else sub
                    if len(v) >= 3:
                        fit = ajuste_escala_log_log(v[x_col].to_numpy(), v[y_col].to_numpy(), n_boot=n_boot, semente=42)
                        res[chave] = dict(_limpar_fit(fit), n_territorios=int(v["territorio_id"].nunique()))
                    else:
                        res[chave] = {"n": int(len(v)), "r": np.nan, "beta": np.nan}
            metricas["por_unidade_e_raio"][un_slug][f"R_{int(r_busca)}km"] = res
            if verbose:
                mq = res.get("pop_total__vs__qtd_x_area", {})
                mm = res.get("pop_total__vs__media", {})
                print(
                    f"    [{un_slug:<13s} R<={r_busca:>4.0f} km] n={mq.get('n', 0):>3d} | "
                    f"Conc: beta={mm.get('beta', np.nan):6.3f} r={mm.get('r', np.nan):6.3f} | "
                    f"Conc×Área: beta={mq.get('beta', np.nan):6.3f} r={mq.get('r', np.nan):6.3f} "
                    f"(rho={mq.get('rho', np.nan):6.3f}, p={mq.get('p_r', np.nan):.1e})"
                )

    df_escala = pd.DataFrame(linhas)
    if df_escala.empty:
        metricas["combinado_global_costeiro"] = {}
        return metricas, df_escala

    for col in ("mp_soma", "mp_media", "mp_x_area"):
        df_escala[f"{col}_norm_mediana"] = df_escala.groupby("unidade_slug")[col].transform(lambda s: s / s.median())
    df_un = df_escala.drop_duplicates(subset=["global_basin_uid"], keep="first").copy()
    fit_med = ajuste_escala_log_log(df_un["pop_total"].to_numpy(), df_un["mp_media_norm_mediana"].to_numpy(), n_boot=n_boot, semente=42)
    fit_qtd = ajuste_escala_log_log(df_un["pop_total"].to_numpy(), df_un["mp_x_area_norm_mediana"].to_numpy(), n_boot=n_boot, semente=42)
    m_urb = df_un["pop_urbana"] > 0
    fit_urb = ajuste_escala_log_log(df_un.loc[m_urb, "pop_urbana"].to_numpy(), df_un.loc[m_urb, "mp_x_area_norm_mediana"].to_numpy(), n_boot=n_boot, semente=42)
    metricas["combinado_global_costeiro"] = {
        "raio_km": RAIO_PADRAO_KM,
        "n_bacias_unicas": int(len(df_un)),
        "n_territorios": int(df_un["territorio_id"].nunique()),
        "pop_total__vs__mp_media_norm": _limpar_fit(fit_med),
        "pop_total__vs__mp_x_area_norm": _limpar_fit(fit_qtd),
        "pop_urbana__vs__mp_x_area_norm": _limpar_fit(fit_urb),
    }
    if verbose:
        for nome, f in [("Conc. norm.", fit_med), ("Conc×Área norm.", fit_qtd), ("P_urb vs Conc×Área", fit_urb)]:
            print(
                f"    [COMBINADO {nome:<20s}] n={f['n']} beta={f['beta']:.3f} [{f['ic95_inf']:.3f}, {f['ic95_sup']:.3f}] "
                f"r={f['r']:.3f} (p={f['p_r']:.2e}) rho={f['rho']:.3f} (p={f['p_rho']:.2e})"
            )
    return metricas, df_escala


def resumo_cenario(metricas: dict) -> dict:
    comb = metricas.get("combinado_global_costeiro", {})
    out = {}
    for chave in ("pop_total__vs__mp_x_area_norm", "pop_total__vs__mp_media_norm", "pop_urbana__vs__mp_x_area_norm"):
        f = comb.get(chave, {})
        out[chave] = {k: f.get(k) for k in ("n", "beta", "ic95_inf", "ic95_sup", "r", "p_r", "rho", "p_rho", "r2", "p_perm")}
    por_un = {}
    for un_slug, _, _ in UNIDADES_INFO:
        f = metricas.get("por_unidade_e_raio", {}).get(un_slug, {}).get(f"R_{int(RAIO_PADRAO_KM)}km", {}).get("pop_total__vs__qtd_x_area", {})
        por_un[un_slug] = {k: f.get(k) for k in ("n", "beta", "ic95_inf", "ic95_sup", "r", "p_r", "rho", "p_rho")}
    out["por_unidade_R25_pop_total__vs__qtd_x_area"] = por_un
    return out


def bacias_novas(df_novo: pd.DataFrame, df_ref: pd.DataFrame, tol_km: float = 10.0) -> pd.DataFrame:
    """Bacias com coleta em `df_novo` cuja foz não tem correspondente em `df_ref` a <= tol_km."""
    if df_novo.empty:
        return df_novo
    nov = df_novo.drop_duplicates("global_basin_uid").copy()
    if df_ref is None or df_ref.empty:
        nov["nova"] = True
        return nov
    ref = df_ref.drop_duplicates(subset=["foz_lon", "foz_lat"])
    d = haversine_deg(nov["foz_lon"].to_numpy()[:, None], nov["foz_lat"].to_numpy()[:, None],
                      ref["foz_lon"].to_numpy()[None, :], ref["foz_lat"].to_numpy()[None, :])
    nov["dist_foz_ref_km"] = d.min(axis=1)
    return nov[nov["dist_foz_ref_km"] > tol_km]


def validar_bacias_literatura(df_exo_todas: pd.DataFrame, raio_km: float = 200.0) -> pd.DataFrame:
    linhas = []
    for nome, (a_ref, lon, lat) in BACIAS_REFERENCIA.items():
        d = haversine_deg(df_exo_todas["foz_lon"].to_numpy(), df_exo_todas["foz_lat"].to_numpy(),
                          np.full(len(df_exo_todas), lon), np.full(len(df_exo_todas), lat))
        cand = df_exo_todas[d <= raio_km]
        if cand.empty:
            linhas.append({"bacia": nome, "area_literatura_km2": a_ref, "status": "foz nao encontrada"})
            continue
        row = cand.loc[cand["area_km2"].idxmax()]
        desvio = (row["area_km2"] - a_ref) / a_ref
        linhas.append({
            "bacia": nome,
            "area_literatura_km2": a_ref,
            "area_ipba_endo_km2": float(row["area_km2"]),
            "area_ipba_sem_filtro_km2": float(row["area_ipba_sem_filtro_km2"]),
            "desvio_relativo": float(desvio),
            "dentro_20pct": bool(abs(desvio) <= 0.20),
            "bloco_id": row["bloco_id"],
            "global_basin_uid": row["global_basin_uid"],
            "foz_lon": float(row["foz_lon"]),
            "foz_lat": float(row["foz_lat"]),
            "status": "ok" if abs(desvio) <= 0.20 else "fora da tolerancia",
        })
    return pd.DataFrame(linhas)


# ==============================================================================
# PIPELINE
# ==============================================================================

def executar_pipeline(reusar_zipf: bool = False, sensibilidade: bool = True):
    t_inicio = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CATALOGO_PATH, "r", encoding="utf-8") as f:
        catalogo = json.load(f)
    if "blocos_hidrologicos" not in catalogo:
        raise RuntimeError("Catálogo sem 'blocos_hidrologicos': rode notebooks/gebco_cluster_dvider/executar_divisao_ilhas.py")

    df_noaa = carregar_noaa_limpo(NOAA_PATH, apenas_costeiro=True)
    print(f"[NOAA nova] {len(df_noaa):,} amostras costeiras (sem 'Ocean water')")
    for meio, cnt in df_noaa["meio"].value_counts().items():
        print(f"       - meio {meio:<16s}: {cnt:>6,d}")
    df_noaa_leg = carregar_noaa_limpo(NOAA_LEGADO_PATH, apenas_costeiro=True) if NOAA_LEGADO_PATH.exists() else None
    if df_noaa_leg is not None:
        print(f"[NOAA legado] {len(df_noaa_leg):,} amostras costeiras")

    # 1. Hidrologia em 2 blocos
    print("\n======================================================================")
    print("[1] HIDROLOGIA: IPBA EM 2 BLOCOS CONTÍGUOS + BACIAS ENDORREICAS")
    print("======================================================================")
    blocos_proc = {}
    for bloco in catalogo["blocos_hidrologicos"]:
        blocos_proc[bloco["id"]] = processar_bloco_hidrologico(bloco, sensibilidade)
    exportar_recortes_regionais(blocos_proc)

    # 2. Demografia por região
    print("\n======================================================================")
    print("[2] DEMOGRAFIA: ZIPF DINÂMICO + CCA EM 6 REGIÕES DISJUNTAS")
    print("======================================================================")
    pop_tot = {bid: np.zeros(int(b["G"].max()) + 1) for bid, b in blocos_proc.items()}
    pop_urb = {bid: np.zeros(int(b["G"].max()) + 1) for bid, b in blocos_proc.items()}
    pop_por_regiao = {}
    calibracoes, pedacos_all, resumo_div = [], [], {}
    for reg in REGIOES_DEMOGRAFICAS:
        cal_json, p_tot, p_urb, df_ped, resumo = processar_regiao_demografica(reg, blocos_proc, reusar_zipf)
        calibracoes.append(cal_json)
        resumo_div[reg["id"]] = resumo
        pop_tot[reg["bloco"]] += p_tot
        pop_urb[reg["bloco"]] += p_urb
        pop_por_regiao[(reg["bloco"], reg["id"])] = p_tot
        pedacos_all.append(df_ped)

    df_ped_all = pd.concat(pedacos_all, ignore_index=True)
    n_manchas = {}
    for bid, b in blocos_proc.items():
        sub = df_ped_all[df_ped_all["bloco_id"] == bid]
        n_manchas[bid] = np.bincount(sub["grid_id"].to_numpy(), minlength=int(b["G"].max()) + 1)

    # 3. Tabela consolidada de bacias
    tabelas = []
    for bid, b in blocos_proc.items():
        for df_b in (b["df_exo"], b["df_endo"]):
            if df_b.empty:
                continue
            g = df_b["grid_id"].to_numpy().astype(np.int64)
            df_b = df_b.copy()
            df_b["pop_total"] = pop_tot[bid][g]
            df_b["pop_urbana"] = pop_urb[bid][g]
            df_b["pop_rural"] = np.maximum(0.0, df_b["pop_total"] - df_b["pop_urbana"])
            df_b["n_manchas_na_bacia"] = n_manchas[bid][g]
            regs = []
            for i_g, gid in enumerate(g):
                tot = df_b["pop_total"].iat[i_g]
                regs.append(";".join(
                    rid for (bb, rid), arr in pop_por_regiao.items()
                    if bb == bid and tot > 0 and arr[gid] / tot >= 0.01
                ))
            df_b["regioes_demograficas"] = regs
            tabelas.append(df_b)
    df_todas = pd.concat(tabelas, ignore_index=True)
    cods = codigo_regiao(df_todas["foz_lat"].to_numpy(), df_todas["foz_lon"].to_numpy())
    df_todas["territorio_id"] = [REGIAO_POR_CODIGO[int(c)]["id"] for c in cods]
    df_todas["territorio_nome_en"] = [REGIAO_POR_CODIGO[int(c)]["nome_en"] for c in cods]
    cal_por_reg = {c["territorio_id"]: c for c in calibracoes}
    df_todas["D_star_territorio"] = [cal_por_reg[t]["D_star_hab_km2"] for t in df_todas["territorio_id"]]
    df_todas["l_metros_territorio"] = [cal_por_reg[t]["l_metros"] for t in df_todas["territorio_id"]]
    df_todas["global_basin_uid"] = [
        f"{b}__{'E' if e else 'B'}{abs(int(i))}" for b, e, i in zip(df_todas["bloco_id"], df_todas["is_endorheic"], df_todas["basin_id"])
    ]

    df_exo_todas = df_todas[~df_todas["is_endorheic"]].copy()
    df_major = df_exo_todas[df_exo_todas["in_p90"]].copy()
    df_endo = df_todas[df_todas["is_endorheic"]].sort_values("area_km2", ascending=False)

    cols_saida = [
        "bloco_id", "territorio_id", "territorio_nome_en", "basin_id", "global_basin_uid", "is_endorheic",
        "nome_sistema_endorreico", "foz_lon", "foz_lat", "area_px", "area_km2", "area_ipba_sem_filtro_km2",
        "frac_area_acum", "in_p90", "pop_total", "pop_urbana", "pop_rural", "n_manchas_na_bacia",
        "regioes_demograficas", "D_star_territorio", "l_metros_territorio",
    ]
    pd.concat([df_major, df_endo], ignore_index=True)[cols_saida].to_csv(RESULTS_DIR / "bacias_global_consolidado.csv", index=False)
    df_endo[cols_saida + ["pit_elev_m", "nivel_transbordo_m", "profundidade_m"]].to_csv(
        RESULTS_DIR / "bacias_endorreicas_global.csv", index=False)
    df_valid = validar_bacias_literatura(df_exo_todas)
    df_valid.to_csv(RESULTS_DIR / "validacao_bacias_literatura.csv", index=False)
    sens = [l for b in blocos_proc.values() for l in b["sens"]]
    if sens:
        pd.DataFrame(sens).to_csv(RESULTS_DIR / "sensibilidade_endorreica.csv", index=False)

    df_ped_all["bacia_principal"] = False
    for bid, b in blocos_proc.items():
        m = df_ped_all["bloco_id"] == bid
        majors = set(df_major.loc[df_major["bloco_id"] == bid, "grid_id"].tolist())
        df_ped_all.loc[m, "bacia_principal"] = df_ped_all.loc[m, "grid_id"].isin(majors)
        off = b["endo_offset"]
        g = df_ped_all.loc[m, "grid_id"].to_numpy()
        df_ped_all.loc[m, "basin_id"] = np.where(g >= off, -(g - off), g)
    df_ped_all["basin_id"] = df_ped_all["basin_id"].astype(np.int64)
    df_ped_all.drop(columns=["grid_id"]).to_csv(RESULTS_DIR / "pedacos_mancha_bacia_global.csv", index=False)

    df_cal = pd.DataFrame([{
        "territorio_id": c["territorio_id"],
        "nome_en": c["nome_en"],
        "bloco_hidrologico": c.get("bloco_hidrologico"),
        "D_star_hab_km2": c["D_star_hab_km2"],
        "l_metros": c["l_metros"],
        "L_MIN_metros": c["L_MIN_metros"],
        "alpha_ols_100": c["alpha_ols"],
        "r2_ols_100": c["r2_ols"],
        "restricao_r2_atendida": c["restricao_r2_atendida"],
        "alpha_ols_50": c["alpha_ols_N"].get("50"),
        "alpha_ols_200": c["alpha_ols_N"].get("200"),
        "alpha_50": c["alpha_hill"].get("50"),
        "alpha_100": c["alpha_hill"].get("100"),
        "alpha_200": c["alpha_hill"].get("200"),
        "alpha_csn": c["csn_zipf"].get("alpha"),
        "r2_ols_zipf": c["csn_zipf"].get("r2_ols"),
        "n_manchas": c["n_manchas_urbanas"],
        "pop_urbana_hab": c["populacao_urbana_total_hab"],
        "pop_total_hab": c["populacao_territorio_total_hab"],
        "maior_mancha_hab": c["maior_aglomeracao_hab"],
        "n_manchas_divididas_entre_bacias": resumo_div[c["territorio_id"]]["n_manchas_divididas"],
    } for c in calibracoes])
    df_cal.to_csv(RESULTS_DIR / "cca_calibracao_global_resumo.csv", index=False)
    with open(RESULTS_DIR / "regioes_demograficas.json", "w", encoding="utf-8") as f:
        json.dump(REGIOES_DEMOGRAFICAS, f, indent=2, ensure_ascii=False)

    print("\n[Validação IPBA x literatura]")
    for _, r in df_valid.iterrows():
        if "area_ipba_endo_km2" in r and pd.notna(r.get("area_ipba_endo_km2")):
            print(f"   {r['bacia']:<20s} IPBA={r['area_ipba_endo_km2'] / 1e6:5.2f} Mkm² | lit={r['area_literatura_km2'] / 1e6:5.2f} "
                  f"| desvio={100 * r['desvio_relativo']:+6.1f}% | {r['status']}")
    print(f"   Bacias exorreicas com foz a <= 50 km da foz do Nilo: "
          f"{int((haversine_deg(df_major['foz_lon'].to_numpy(), df_major['foz_lat'].to_numpy(), np.full(len(df_major), 31.0), np.full(len(df_major), 31.5)) <= 50).sum())} grandes bacias")

    # 4. Pareamento NOAA e alometria
    print("\n======================================================================")
    print("[3] PAREAMENTO BACIA <-> NOAA COSTEIRO E ALOMETRIA")
    print("======================================================================")
    gt0 = next(iter(blocos_proc.values()))["gt"]
    dlon, dlat = abs(gt0[1]), abs(gt0[5])
    df_par_c = parear_bacias_noaa(df_major, df_noaa, dlon, dlat)
    metricas_c, df_escala_c = executar_alometria(df_par_c, "C: hidrologia v2 + NOAA nova (oficial)")
    df_escala_c.to_csv(RESULTS_DIR / "escala_pop_lixo_global.csv", index=False)

    comparativo = {"C_hidrologia_v2_noaa_nova": resumo_cenario(metricas_c)}
    df_escala_b = pd.DataFrame()
    if df_noaa_leg is not None:
        df_par_b = parear_bacias_noaa(df_major, df_noaa_leg, dlon, dlat)
        metricas_b, df_escala_b = executar_alometria(df_par_b, "B: hidrologia v2 + NOAA legado")
        comparativo["B_hidrologia_v2_noaa_legado"] = resumo_cenario(metricas_b)
    df_escala_a = pd.DataFrame()
    if (BASELINE_DIR / "metricas_alometria_global.json").exists():
        with open(BASELINE_DIR / "metricas_alometria_global.json", "r", encoding="utf-8") as f:
            comparativo["A_baseline_v1"] = resumo_cenario(json.load(f))
        df_escala_a = pd.read_csv(BASELINE_DIR / "escala_pop_lixo_global.csv")

    nov_vs_b = bacias_novas(df_escala_c, df_escala_b)
    nov_vs_a = bacias_novas(df_escala_c, df_escala_a)
    cols_nov = ["global_basin_uid", "territorio_id", "unidade_slug", "foz_lon", "foz_lat", "area_km2",
                "pop_total", "pop_urbana", "n_amostras_noaa", "dist_foz_km", "mp_media", "mp_x_area"]
    pd.concat([
        nov_vs_b[cols_nov].assign(comparado_com="B_noaa_legado"),
        nov_vs_a[cols_nov].assign(comparado_com="A_baseline_v1"),
    ], ignore_index=True).to_csv(RESULTS_DIR / "bacias_novas_com_coleta.csv", index=False)
    comparativo["bacias_novas_com_coleta"] = {
        "vs_B_noaa_legado": int(len(nov_vs_b)),
        "vs_A_baseline_v1": int(len(nov_vs_a)),
    }
    comparativo["noaa_amostras_costeiras"] = {
        "nova": int(len(df_noaa)),
        "legado": int(len(df_noaa_leg)) if df_noaa_leg is not None else None,
        "nova_por_meio": {k: int(v) for k, v in df_noaa["meio"].value_counts().items()},
        "legado_por_meio": {k: int(v) for k, v in df_noaa_leg["meio"].value_counts().items()} if df_noaa_leg is not None else None,
    }

    metricas_c["resumo_destaque"] = {
        "raio_km": RAIO_PADRAO_KM,
        "modelo": "(M x A)_norm = c * P_total^beta",
        **comparativo["C_hidrologia_v2_noaa_nova"]["pop_total__vs__mp_x_area_norm"],
    }
    metricas_c["comparativo"] = comparativo
    metricas_c["hidrologia"] = {
        bid: {"diagnostico_endorreico": b["diag"], "n_bacias_exorreicas": int(len(b["df_exo"])),
              "n_grandes_bacias_p90": int(b["df_exo"]["in_p90"].sum()), "n_bacias_endorreicas": int(len(b["df_endo"]))}
        for bid, b in blocos_proc.items()
    }
    metricas_c["validacao_literatura"] = {
        "n_bacias_referencia": int(df_valid["area_ipba_endo_km2"].notna().sum()) if "area_ipba_endo_km2" in df_valid else 0,
        "n_dentro_20pct": int(df_valid.get("dentro_20pct", pd.Series(dtype=bool)).fillna(False).sum()),
    }
    with open(RESULTS_DIR / "metricas_alometria_global.json", "w", encoding="utf-8") as f:
        json.dump(metricas_c, f, indent=2, ensure_ascii=False, default=float)

    print(f"\n[PIPELINE GLOBAL CONCLUÍDO EM {time.time() - t_inicio:.1f}s]")
    return metricas_c


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline global padrão-ouro (IPBA 2 blocos + endorreismo + Zipf dinâmico + NOAA)")
    parser.add_argument("--reusar-zipf", action="store_true", help="Reutiliza calibrações Zipf dinâmicas já salvas")
    parser.add_argument("--sem-sensibilidade", action="store_true", help="Pula a análise de sensibilidade endorreica")
    args = parser.parse_args()
    executar_pipeline(reusar_zipf=args.reusar_zipf, sensibilidade=not args.sem_sensibilidade)
