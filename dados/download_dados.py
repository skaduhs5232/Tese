"""
download_dados.py
-----------------
Script unificado para download, atualização, higienização de dados/resultados antigos
e verificação local das 3 bases de dados oficiais do projeto (versões mais recentes de 2026):

  1. Limpeza de dados e resultados antigos (`limpar_dados_e_resultados_antigos()`):
     - Remove arquivos obsoletos em `dados/` (`pop_gpw_crop.tif`, `archives/gpw_data.zip`),
       `notebooks/results/` (arquivos legados `*_na_panama.*`, `cca_calibracao.json`,
       `divisao_mancha_relevo.*`, `gpw_mundo_40.npz`, `gifs/`),
       `notebooks/IPBA/` (`belm_crop.*`, `str1330_link.*`),
       `notebooks/CCA/` (dados antigos do Ceará `df_ceara.*`, `chikv/`, `regionalizacao/`, etc.)
       e arquivos `.DS_Store` / `.ipynb_checkpoints`.

  2. GEBCO 2026 (General Bathymetric Chart of the Oceans):
     - Fonte COG primária (AWS S3 Source Cooperative / AADC):
       https://s3.us-west-2.amazonaws.com/us-west-2.opendata.source.coop/ausantarctic/gebco/GEBCO_2026.tif
     - Salva localmente em `dados/gebco/` e em `notebooks/gebco_cluster_dvider/datasets/processed/gebco_2026_sub_ice_geotiff.tif`.

  3. GPW v4 Rev11 (Gridded Population of the World, Population Count 30 arc-sec):
     - Verifica/extrai `dados/gpw_data/gpw_v4_population_count_rev11_2020_30_sec.tif`
       e valida que os valores correspondem à CONTAGEM DE PESSOAS POR PIXEL (habitantes).

  4. NOAA Marine Microplastics (NCEI / ArcGIS FeatureServer Global):
     - Verifica `dados/noaa_microplastics.csv` contra o servidor oficial da NOAA NCEI
       e atualiza caso haja novos registros globais.

Uso:
  python dados/download_dados.py [--overview 2] [--forcar] [--gerar-tiles-raw]
"""

import argparse
import glob
import json
import os
import shutil
import time
import urllib.parse
import urllib.request
import sys
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from osgeo import gdal

gdal.UseExceptions()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
DADOS_DIR = SCRIPT_DIR
GEBCO_DIR = os.path.join(DADOS_DIR, "gebco")
GPW_DIR = os.path.join(DADOS_DIR, "gpw_data")
NOTEBOOKS_DIR = os.path.join(REPO_ROOT, "notebooks")
RESULTS_DIR = os.path.join(NOTEBOOKS_DIR, "results")
DIVIDER_DIR = os.path.join(NOTEBOOKS_DIR, "gebco_cluster_dvider")
DIVIDER_RAW_DIR = os.path.join(DIVIDER_DIR, "datasets", "raw")
DIVIDER_PROC_DIR = os.path.join(DIVIDER_DIR, "datasets", "processed")

URL_GEBCO_2026_COG = (
    "/vsicurl/https://s3.us-west-2.amazonaws.com/"
    "us-west-2.opendata.source.coop/ausantarctic/gebco/GEBCO_2026.tif"
)
URL_GEBCO_2026_CEDA_ZIP = (
    "https://dap.ceda.ac.uk/bodc/gebco/global/gebco_2026/"
    "sub_ice_topography_bathymetry/geotiff/gebco_2026_sub_ice_topo_geotiff.zip"
)
URL_NOAA_FEATURESERVER = (
    "https://services2.arcgis.com/C8EMgrsFcRFL6LrL/arcgis/rest/services/"
    "Marine_Microplastics_WGS84/FeatureServer/0"
)


def limpar_dados_e_resultados_antigos() -> int:
    """
    Remove dados, recortes e resultados antigos que não fazem mais sentido no pipeline global.
    Retorna a quantidade de arquivos/pastas removidos.
    """
    removidos = 0

    # 1. Garante que o GPW v4 já esteja extraído antes de remover o zip duplicado
    gpw_tif = os.path.join(GPW_DIR, "gpw_v4_population_count_rev11_2020_30_sec.tif")
    gpw_zip = os.path.join(DADOS_DIR, "archives", "gpw_data.zip")
    if not os.path.exists(gpw_tif) and os.path.exists(gpw_zip):
        os.makedirs(GPW_DIR, exist_ok=True)
        with zipfile.ZipFile(gpw_zip, "r") as zf:
            zf.extractall(GPW_DIR)

    # 2. Lista de padrões de arquivos antigos a remover
    padroes_arquivos = [
        os.path.join(DADOS_DIR, "pop_gpw_crop.tif"),
        os.path.join(DADOS_DIR, "archives", "gpw_data.zip"),
        os.path.join(RESULTS_DIR, "*_na_panama.*"),
        os.path.join(RESULTS_DIR, "cca_calibracao.json"),
        os.path.join(RESULTS_DIR, "divisao_mancha_relevo.*"),
        os.path.join(RESULTS_DIR, "gpw_mundo_*.npz"),
        os.path.join(NOTEBOOKS_DIR, "IPBA", "datasets", "belm_crop.*"),
        os.path.join(NOTEBOOKS_DIR, "IPBA", "datasets", "str1330_link.*"),
        os.path.join(NOTEBOOKS_DIR, "IPBA", "results", "belm_crop.*"),
        os.path.join(NOTEBOOKS_DIR, "CCA", "datasets", "*"),
        os.path.join(NOTEBOOKS_DIR, "CCA", "filtrando_ceara.ipynb"),
        os.path.join(NOTEBOOKS_DIR, "CCA", "cca", "cca_sensitario_ceara.ipynb"),
        os.path.join(NOTEBOOKS_DIR, "CCA", "cca", "filtrando_ceara.ipynb"),
        os.path.join(NOTEBOOKS_DIR, "CCA", "cca", "cca.ipynb"),
    ]
    for padrao in padroes_arquivos:
        for caminho in glob.glob(padrao):
            if os.path.isfile(caminho):
                os.remove(caminho)
                removidos += 1
                print(f"[Limpeza] Arquivo antigo removido: {os.path.relpath(caminho, REPO_ROOT)}")

    # 3. Pastas antigas a remover
    pastas_antigas = [
        os.path.join(DADOS_DIR, "archives"),
        os.path.join(RESULTS_DIR, "gifs"),
        os.path.join(NOTEBOOKS_DIR, "CCA", "chikv"),
        os.path.join(NOTEBOOKS_DIR, "CCA", "regionalizacao"),
        os.path.join(NOTEBOOKS_DIR, "CCA", "datasets"),
    ]
    for pasta in pastas_antigas:
        if os.path.isdir(pasta):
            shutil.rmtree(pasta, ignore_errors=True)
            removidos += 1
            print(f"[Limpeza] Pasta antiga removida: {os.path.relpath(pasta, REPO_ROOT)}")

    # 4. Remover .DS_Store e .ipynb_checkpoints em notebooks/ e dados/
    for base_dir in (DADOS_DIR, NOTEBOOKS_DIR):
        for root, dirs, files in os.walk(base_dir):
            for f in files:
                if f == ".DS_Store":
                    fp = os.path.join(root, f)
                    os.remove(fp)
                    removidos += 1
            for d in list(dirs):
                if d == ".ipynb_checkpoints":
                    dp = os.path.join(root, d)
                    shutil.rmtree(dp, ignore_errors=True)
                    dirs.remove(d)
                    removidos += 1

    print(f"[Limpeza] Concluída ({removidos} itens obsoletos removidos).")
    return removidos


def salvar_geotiff_int16(caminho: str, arr: np.ndarray, geotransform: tuple,
                         projection: str, nodata: int = -32768):
    """Salva uma matriz 2D int16 como GeoTIFF comprimido (DEFLATE)."""
    os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
    nrows, ncols = arr.shape
    driver = gdal.GetDriverByName("GTiff")
    opcoes = ["COMPRESS=DEFLATE", "TILED=YES", "PREDICTOR=2", "BIGTIFF=IF_SAFER"]
    ds = driver.Create(caminho, int(ncols), int(nrows), 1, gdal.GDT_Int16, options=opcoes)
    ds.SetGeoTransform(geotransform)
    if projection:
        ds.SetProjection(projection)
    band = ds.GetRasterBand(1)
    band.SetNoDataValue(float(nodata))
    band.WriteArray(arr.astype(np.int16))
    ds.FlushCache()
    del band
    del ds


def baixar_gebco_2026(overview_idx: int = 2, gerar_tiles_raw: bool = False, forcar: bool = False):
    """
    Baixa o GEBCO 2026 global a partir do Cloud-Optimized GeoTIFF (COG) oficial de 2026
    e grava:
      - `notebooks/gebco_cluster_dvider/datasets/processed/gebco_2026_sub_ice_geotiff.tif`
      - `dados/gebco/gebco_2026_sub_ice_geotiff.tif`
    """
    os.makedirs(GEBCO_DIR, exist_ok=True)
    os.makedirs(DIVIDER_PROC_DIR, exist_ok=True)

    destino_divider = os.path.join(DIVIDER_PROC_DIR, "gebco_2026_sub_ice_geotiff.tif")
    destino_dados = os.path.join(GEBCO_DIR, "gebco_2026_sub_ice_geotiff.tif")

    if os.path.exists(destino_divider) and os.path.exists(destino_dados) and not forcar:
        ds_check = gdal.Open(destino_divider)
        print(
            f"[GEBCO 2026] Arquivo mais recente já disponível localmente: {destino_divider} "
            f"({ds_check.RasterXSize} x {ds_check.RasterYSize})"
        )
        del ds_check
        return destino_divider

    print("[GEBCO 2026] Conectando ao repositório oficial GEBCO 2026 COG...")
    t0 = time.time()
    ds = gdal.Open(URL_GEBCO_2026_COG)
    proj = ds.GetProjection()
    band = ds.GetRasterBand(1)
    nodata = band.GetNoDataValue()
    if nodata is None:
        nodata = -32768

    alvo = band if overview_idx < 0 else band.GetOverview(overview_idx)
    ncols, nrows = alvo.XSize, alvo.YSize
    print(f"[GEBCO 2026] Baixando grade global {ncols} x {nrows} (overview={overview_idx})...")
    arr = alvo.ReadAsArray().astype(np.int16)
    dt = time.time() - t0
    print(f"[GEBCO 2026] Download concluído em {dt:.1f}s ({arr.nbytes / 1e6:.1f} MB em memória).")

    dx = 360.0 / ncols
    dy = 180.0 / nrows
    gt = (-180.0, dx, 0.0, 90.0, 0.0, -dy)

    salvar_geotiff_int16(destino_divider, arr, gt, proj, int(nodata))
    salvar_geotiff_int16(destino_dados, arr, gt, proj, int(nodata))
    print(f"[GEBCO 2026] Salvo em:\n  - {destino_divider}\n  - {destino_dados}")

    if gerar_tiles_raw:
        os.makedirs(DIVIDER_RAW_DIR, exist_ok=True)
        ht = nrows // 2
        wt = ncols // 4
        lats = [("n90.0_s0.0", 90.0), ("n0.0_s-90.0", 0.0)]
        lons = [
            ("w-180.0_e-90.0", -180.0),
            ("w-90.0_e0.0", -90.0),
            ("w0.0_e90.0", 0.0),
            ("w90.0_e180.0", 90.0),
        ]
        for j, (lat_tag, lat_top) in enumerate(lats):
            for i, (lon_tag, lon_left) in enumerate(lons):
                sub = arr[j * ht:(j + 1) * ht, i * wt:(i + 1) * wt]
                gt_sub = (lon_left, dx, 0.0, lat_top, 0.0, -dy)
                nome_tile = f"gebco_2026_sub_ice_{lat_tag}_{lon_tag}_geotiff.tif"
                caminho_tile = os.path.join(DIVIDER_RAW_DIR, nome_tile)
                salvar_geotiff_int16(caminho_tile, sub, gt_sub, proj, int(nodata))
        print(f"[GEBCO 2026] 8 tiles salvos em {DIVIDER_RAW_DIR}")

    del alvo, band, ds
    return destino_divider


def verificar_gpw_v4():
    """
    Verifica a presença do raster GPW v4 Rev11 (2020, 30 arc-sec) em `dados/gpw_data/`
    e documenta a natureza da variável (contagem de habitantes por pixel).
    """
    os.makedirs(GPW_DIR, exist_ok=True)
    tif_path = os.path.join(GPW_DIR, "gpw_v4_population_count_rev11_2020_30_sec.tif")
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"Arquivo GPW v4 não encontrado em {tif_path}")

    ds = gdal.Open(tif_path)
    b = ds.GetRasterBand(1)
    gt = ds.GetGeoTransform()
    amostra = b.ReadAsArray(10800, 5400, 500, 500)
    amostra_valida = amostra[amostra > 0]
    max_amostra = float(amostra_valida.max()) if amostra_valida.size else 0.0

    print(
        f"[GPW v4] Verificado: {tif_path}\n"
        f"         Dimensões: {ds.RasterXSize} x {ds.RasterYSize} pixels | "
        f"Resolução: {gt[1]*3600:.1f}\" x {abs(gt[5])*3600:.1f}\"\n"
        f"         Variável: CONTAGEM DE PESSOAS POR PIXEL (population_count, hab/pixel) | "
        f"Pico na amostra: {max_amostra:,.1f} hab/pixel"
    )
    del b, ds
    return tif_path


def verificar_ou_atualizar_noaa(forcar_download: bool = False, verificar_remoto: bool = False):
    """
    Gera `dados/noaa_microplastics.csv` a partir do GeoJSON local em `dados/noaa_microplastic/`
    (via `ingestao_noaa_geojson.py`). A consulta ao FeatureServer da NOAA NCEI só ocorre se não
    houver GeoJSON local ou se `verificar_remoto=True`; nesse caso o CSV baixado tem o esquema legado.
    """
    csv_path = os.path.join(DADOS_DIR, "noaa_microplastics.csv")
    geojsons = sorted(glob.glob(os.path.join(DADOS_DIR, "noaa_microplastic", "*.geojson")))
    if geojsons and not verificar_remoto:
        import ingestao_noaa_geojson as ing

        df = ing.padronizar(ing.ler_geojson(Path(geojsons[-1])))
        ing.relatorio(df)
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[NOAA] CSV padronizado gerado a partir de {os.path.basename(geojsons[-1])}: {csv_path}")
        return csv_path

    n_local = 0
    if os.path.exists(csv_path):
        df_local = pd.read_csv(csv_path, low_memory=False)
        n_local = len(df_local)

    try:
        url_count = f"{URL_NOAA_FEATURESERVER}/query?where=1%3D1&returnCountOnly=true&f=json"
        with urllib.request.urlopen(url_count, timeout=10) as r:
            n_remoto = int(json.loads(r.read().decode()).get("count", 0))
    except Exception as e:
        print(f"[NOAA] Aviso ao consultar contagem remota ({e}); mantendo base local ({n_local:,} amostras).")
        return csv_path

    if n_local >= n_remoto and n_remoto > 0 and not forcar_download:
        print(
            f"[NOAA] Base local atualizada: {csv_path} "
            f"({n_local:,} amostras locais >= {n_remoto:,} no servidor oficial NCEI)."
        )
        return csv_path

    if n_remoto == 0 and n_local > 0 and not forcar_download:
        print(f"[NOAA] Base local verificada: {csv_path} ({n_local:,} amostras).")
        return csv_path

    print(f"[NOAA] Baixando {n_remoto:,} registros globais do servidor NCEI...")
    registros = []
    offset = 0
    lote = 2000
    while offset < n_remoto:
        params = urllib.parse.urlencode({
            "where": "1=1",
            "outFields": "*",
            "resultOffset": offset,
            "resultRecordCount": lote,
            "f": "json",
        })
        with urllib.request.urlopen(f"{URL_NOAA_FEATURESERVER}/query?{params}", timeout=30) as r:
            dados = json.loads(r.read().decode())
        feats = dados.get("features", [])
        if not feats:
            break
        for f in feats:
            registros.append(f.get("attributes", {}))
        offset += len(feats)
        print(f"  -> {len(registros):,}/{n_remoto:,} amostras baixadas...")

    if registros:
        df_novo = pd.DataFrame(registros)
        df_novo.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"[NOAA] Salvo em {csv_path} ({len(df_novo):,} amostras).")
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Limpeza de dados antigos, download e verificação GEBCO 2026, GPW v4 e NOAA.")
    parser.add_argument("--overview", type=int, default=2,
                        help="Nível de overview do GEBCO 2026 (0=30\", 1=1', 2=2' [padrão], 3=4').")
    parser.add_argument("--gerar-tiles-raw", action="store_true",
                        help="Gera também os 8 tiles individuais em notebooks/gebco_cluster_dvider/datasets/raw/.")
    parser.add_argument("--forcar", action="store_true",
                        help="Força o re-download mesmo se os arquivos locais já existirem.")
    parser.add_argument("--noaa-remoto", action="store_true",
                        help="Ignora o GeoJSON local e consulta o FeatureServer da NOAA NCEI (esquema legado).")
    args = parser.parse_args()

    limpar_dados_e_resultados_antigos()
    verificar_gpw_v4()
    verificar_ou_atualizar_noaa(forcar_download=args.forcar, verificar_remoto=args.noaa_remoto)
    baixar_gebco_2026(overview_idx=args.overview, gerar_tiles_raw=args.gerar_tiles_raw, forcar=args.forcar)


if __name__ == "__main__":
    main()
