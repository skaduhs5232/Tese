"""
dados/microplasticos.py
-----------------------
Ingestão, inventário de unidades e harmonização física sem distorção entre:
  - AWI LITTERBASE (OGC WFS GeoServer oficial: camadas `itemsm3` e `itemskm2`)
  - NOAA NCEI Marine Microplastics (GeoJSON local / FeatureServer oficial)

Regra metodológica de unidade:
  - Não se converte área (items/km²) em volume (items/m³).
  - Não se incluem unidades de tempo (pieces/10 mins, Nurdle Patrol) na regressão.
  - A única grandeza física idêntica presente em ambas as bases para microplásticos
    de praia ('Beach', tamanho 'micro') é a concentração volumétrica em sedimento:
    `items/m3` no LITTERBASE e `pieces/m3` no NOAA (fator 1:1 exato, sem conversão geométrica).
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DADOS_DIR = Path(__file__).resolve().parent
GEBCO_GLOBAL_TIF = DADOS_DIR / "gebco" / "gebco_2026_sub_ice_geotiff.tif"
GPW_GLOBAL_TIF = DADOS_DIR / "gpw_data" / "gpw_v4_population_count_rev11_2020_30_sec.tif"
NOAA_GEOJSON_LOCAL = (
    DADOS_DIR / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
)

LITTERBASE_CACHE_DIR = DADOS_DIR / "litterbase"
CSV_UNIFICADO_AMERICAS = DADOS_DIR / "microplasticos_americas_items_m3.csv"
CSV_UNIFICADO_GLOBAL = DADOS_DIR / "microplasticos_global_items_m3.csv"
JSON_INVENTARIO = DADOS_DIR / "inventario_unidades_microplasticos.json"

URL_AWI_WFS = "https://maps.awi.de/services/projects/gs/litter_and_micro_plastic_distribution/ows"
BBOX_AMERICAS = (-170.0, -60.0, -30.0, 80.0)  # (lon_min, lat_min, lon_max, lat_max)

COLUNAS_PADRAO = [
    "site_id",
    "latitude",
    "longitude",
    "densidade_microplasticos",
    "unidade",
    "unidade_original",
    "meio",
    "fonte",
]


def verificar_rasters_base() -> dict[str, Any]:
    """Verifica a presença e dimensões dos rasters globais GEBCO 2026 e GPW v4 Rev11."""
    from osgeo import gdal

    gdal.UseExceptions()
    info = {}
    for nome, caminho in [("gebco_2026", GEBCO_GLOBAL_TIF), ("gpw_v4_2020", GPW_GLOBAL_TIF)]:
        if not caminho.is_file():
            raise FileNotFoundError(f"Raster obrigatório não encontrado: {caminho}")
        ds = gdal.Open(str(caminho), gdal.GA_ReadOnly)
        info[nome] = {
            "caminho": str(caminho),
            "largura_px": int(ds.RasterXSize),
            "altura_px": int(ds.RasterYSize),
            "geotransform": tuple(float(x) for x in ds.GetGeoTransform()),
        }
        del ds
    return info


def _baixar_camada_litterbase_wfs(
    layer: str,
    size: str = "micro",
    habitat: str = "Beach",
    timeout: int = 30,
    max_tentativas: int = 3,
) -> pd.DataFrame:
    """
    Baixa ou lê do cache local uma camada do GeoServer WFS do AWI LITTERBASE
    filtrada por `size` e `habitat` em escala global.
    """
    LITTERBASE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_geojson = LITTERBASE_CACHE_DIR / f"litterbase_{layer}_{habitat.lower()}_{size.lower()}.geojson"

    data = None
    cql_filter = f"size = '{size}' AND habitat = '{habitat}'"
    params = {
        "service": "WFS",
        "version": "1.0.0",
        "request": "GetFeature",
        "typeName": layer,
        "outputFormat": "application/json",
        "cql_filter": cql_filter,
    }
    url = f"{URL_AWI_WFS}?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) litterbaseNoaa/1.0"}

    for tentativa in range(1, max_tentativas + 1):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw_bytes = resp.read()
            data = json.loads(raw_bytes.decode("utf-8"))
            with open(cache_geojson, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False)
            break
        except Exception:
            if tentativa < max_tentativas:
                time.sleep(1.5 * tentativa)
            elif cache_geojson.is_file():
                with open(cache_geojson, "r", encoding="utf-8") as f:
                    data = json.load(f)

    if not data or not data.get("features"):
        return pd.DataFrame()

    registros = []
    for feat in data["features"]:
        props = dict(feat.get("properties") or {})
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if coords and len(coords) >= 2:
            if props.get("longitude") is None:
                props["longitude"] = float(coords[0])
            if props.get("latitude") is None:
                props["latitude"] = float(coords[1])
        registros.append(props)
    return pd.DataFrame(registros)


def _carregar_noaa_geojson(caminho: Path = NOAA_GEOJSON_LOCAL) -> pd.DataFrame:
    """Carrega todas as feições do GeoJSON global da NOAA NCEI."""
    if not caminho.is_file():
        raise FileNotFoundError(f"Arquivo GeoJSON da NOAA não encontrado em {caminho}")
    with open(caminho, "r", encoding="utf-8") as f:
        gj = json.load(f)

    registros = []
    for feat in gj.get("features", []):
        props = dict(feat.get("properties") or {})
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if coords and len(coords) >= 2:
            if props.get("Longitude_degree_") is None:
                props["Longitude_degree_"] = float(coords[0])
            if props.get("Latitude__degree_") is None:
                props["Latitude__degree_"] = float(coords[1])
        registros.append(props)
    return pd.DataFrame(registros)


def _recortar_bbox(df: pd.DataFrame, bbox: tuple[float, float, float, float]) -> pd.DataFrame:
    lon_min, lat_min, lon_max, lat_max = bbox
    m = (
        (df["longitude"] >= lon_min)
        & (df["longitude"] <= lon_max)
        & (df["latitude"] >= lat_min)
        & (df["latitude"] <= lat_max)
    )
    return df[m].copy().reset_index(drop=True)


def inventariar_e_unificar_microplasticos(
    bbox: tuple[float, float, float, float] = BBOX_AMERICAS,
    agrupar_mesma_coordenada: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Inventaria as unidades disponíveis em LITTERBASE (WFS) e NOAA (GeoJSON),
    verifica quais grandezas físicas podem ser combinadas sem distorção e
    retorna `(df_americas, inventario)`.

    Unidade harmonizada selecionada:
      `items/m3` (concentração volumétrica em areia de praia), combinando:
        - LITTERBASE camada `itemsm3` (`habitat='Beach'`, `size='micro'`, unidade `items/m3`)
        - NOAA NCEI (`Medium='Beach'`, `Unit='pieces/m3'`, apenas valores > 0)
    """
    # 1. Baixar camadas de praia/micro do LITTERBASE
    df_lb_m3_raw = _baixar_camada_litterbase_wfs(layer="itemsm3", size="micro", habitat="Beach")
    df_lb_km2_raw = _baixar_camada_litterbase_wfs(layer="itemskm2", size="micro", habitat="Beach")

    # 2. Carregar NOAA global
    df_noaa_raw = _carregar_noaa_geojson()

    # Padronizar LITTERBASE items/m3 (global)
    df_lb_m3 = pd.DataFrame(
        {
            "site_id": "litterbase_" + df_lb_m3_raw["objectid"].astype(str),
            "latitude": pd.to_numeric(df_lb_m3_raw["latitude"], errors="coerce"),
            "longitude": pd.to_numeric(df_lb_m3_raw["longitude"], errors="coerce"),
            "densidade_microplasticos": pd.to_numeric(df_lb_m3_raw["loc_value"], errors="coerce"),
            "unidade": "items/m3",
            "unidade_original": df_lb_m3_raw["unit"].astype(str).str.strip(),
            "meio": "Beach",
            "fonte": "litterbase",
        }
    )
    df_lb_m3 = df_lb_m3[
        df_lb_m3["latitude"].notna()
        & df_lb_m3["longitude"].notna()
        & (df_lb_m3["densidade_microplasticos"] > 0)
    ].copy()

    # Padronizar NOAA Beach pieces/m3 (global, > 0 para escala log-log)
    m_noaa_beach_m3 = (
        (df_noaa_raw["Medium"].astype(str).str.strip() == "Beach")
        & (df_noaa_raw["Unit"].astype(str).str.strip() == "pieces/m3")
    )
    sub_noaa = df_noaa_raw[m_noaa_beach_m3].copy()
    id_col = "ObjectId" if "ObjectId" in sub_noaa.columns else "UniqueID"
    df_noaa_m3 = pd.DataFrame(
        {
            "site_id": "noaa_" + sub_noaa[id_col].astype(str),
            "latitude": pd.to_numeric(sub_noaa["Latitude__degree_"], errors="coerce"),
            "longitude": pd.to_numeric(sub_noaa["Longitude_degree_"], errors="coerce"),
            "densidade_microplasticos": pd.to_numeric(
                sub_noaa["Microplastics_measurement"], errors="coerce"
            ),
            "unidade": "items/m3",
            "unidade_original": "pieces/m3",
            "meio": "Beach",
            "fonte": "noaa",
        }
    )
    n_noaa_global_total = int(len(df_noaa_m3))
    df_noaa_m3_pos = df_noaa_m3[
        df_noaa_m3["latitude"].notna()
        & df_noaa_m3["longitude"].notna()
        & (df_noaa_m3["densidade_microplasticos"] > 0)
    ].copy()

    # Unificar globalmente na mesma unidade física (items/m3 == pieces/m3)
    df_global = pd.concat([df_noaa_m3_pos, df_lb_m3], ignore_index=True)

    if agrupar_mesma_coordenada:
        df_global["lat_r"] = df_global["latitude"].round(5)
        df_global["lon_r"] = df_global["longitude"].round(5)
        df_global = (
            df_global.groupby(["lat_r", "lon_r", "unidade", "meio", "fonte"], as_index=False)
            .agg(
                site_id=("site_id", "first"),
                latitude=("latitude", "mean"),
                longitude=("longitude", "mean"),
                densidade_microplasticos=("densidade_microplasticos", "median"),
                unidade_original=("unidade_original", "first"),
            )[COLUNAS_PADRAO]
        )

    df_americas = _recortar_bbox(df_global, bbox)

    # Contagens para o inventário de unidades nas Américas
    lon_min, lat_min, lon_max, lat_max = bbox
    m_am_noaa = (
        (pd.to_numeric(df_noaa_raw["Longitude_degree_"], errors="coerce") >= lon_min)
        & (pd.to_numeric(df_noaa_raw["Longitude_degree_"], errors="coerce") <= lon_max)
        & (pd.to_numeric(df_noaa_raw["Latitude__degree_"], errors="coerce") >= lat_min)
        & (pd.to_numeric(df_noaa_raw["Latitude__degree_"], errors="coerce") <= lat_max)
    )
    noaa_am_grupos = (
        df_noaa_raw[m_am_noaa]
        .groupby(["Medium", "Unit"], dropna=False)
        .size()
        .to_dict()
    )

    n_lb_km2_am = 0
    if not df_lb_km2_raw.empty:
        n_lb_km2_am = int(
            (
                (pd.to_numeric(df_lb_km2_raw["longitude"], errors="coerce") >= lon_min)
                & (pd.to_numeric(df_lb_km2_raw["longitude"], errors="coerce") <= lon_max)
                & (pd.to_numeric(df_lb_km2_raw["latitude"], errors="coerce") >= lat_min)
                & (pd.to_numeric(df_lb_km2_raw["latitude"], errors="coerce") <= lat_max)
            ).sum()
        )

    inventario = {
        "unidade_harmonizada": "items/m3",
        "justificativa_fisica": (
            "1 item/m³ (LITTERBASE camada itemsm3, Beach, micro) equivale exatamente a "
            "1 piece/m³ (NOAA NCEI, Beach). Ambas medem concentração volumétrica de "
            "microplásticos em sedimento de praia sem qualquer fator de conversão geométrica. "
            "Registros por área (items/km²) e por esforço temporal (pieces/10 mins, Nurdle Patrol) "
            "são mantidos fora da regressão para evitar distorção dimensional."
        ),
        "global": {
            "noaa_beach_pieces_m3_total": n_noaa_global_total,
            "noaa_beach_pieces_m3_positivos": int(len(df_noaa_m3_pos)),
            "litterbase_beach_items_m3_positivos": int(len(df_lb_m3)),
            "unificado_items_m3_positivos": int(len(df_global)),
        },
        "americas": {
            "noaa_beach_pieces_m3_positivos": int((df_americas["fonte"] == "noaa").sum()),
            "litterbase_beach_items_m3_positivos": int((df_americas["fonte"] == "litterbase").sum()),
            "unificado_items_m3_positivos": int(len(df_americas)),
            "litterbase_beach_items_km2_descartados_por_dimensao": n_lb_km2_am,
            "noaa_por_meio_e_unidade_bruto": {
                f"{k[0]} | {k[1]}": int(v) for k, v in noaa_am_grupos.items()
            },
        },
    }

    df_global.to_csv(CSV_UNIFICADO_GLOBAL, index=False, encoding="utf-8")
    df_americas.to_csv(CSV_UNIFICADO_AMERICAS, index=False, encoding="utf-8")
    with open(JSON_INVENTARIO, "w", encoding="utf-8") as f:
        json.dump(inventario, f, ensure_ascii=False, indent=2)

    return df_americas, inventario


def obter_microplasticos_por_fonte_e_bbox(
    fonte: str = "ambos",
    bbox: tuple[float, float, float, float] | None = None,
    apenas_america_norte_central: bool = False,
) -> pd.DataFrame:
    """
    Carrega os pontos de microplásticos costeiros harmonizados em `items/m3`
    a partir do dataset global (`CSV_UNIFICADO_GLOBAL`), filtrando:
      - `fonte`: `'ambos'` (NOAA + LITTERBASE), `'litterbase'` ou `'noaa'`.
      - `bbox`: `(lon_min, lat_min, lon_max, lat_max)` opcional.
      - `apenas_america_norte_central`: se True, aplica também a máscara da fronteira
        Colômbia–Panamá (`codigo_regiao_americas == 0`) para excluir o norte da América do Sul.
    """
    if not CSV_UNIFICADO_GLOBAL.is_file():
        inventariar_e_unificar_microplasticos()
    df = pd.read_csv(CSV_UNIFICADO_GLOBAL)

    f_norm = fonte.strip().lower()
    if f_norm in ("litterbase", "noaa"):
        df = df[df["fonte"] == f_norm].copy()
    elif f_norm not in ("ambos", "all", "litterbase_noaa"):
        raise ValueError(f"Fonte inválida: {fonte!r} (use 'ambos', 'litterbase' ou 'noaa')")

    if bbox is not None:
        df = _recortar_bbox(df, bbox)

    if apenas_america_norte_central and not df.empty:
        from calculos.geometria import codigo_regiao_americas

        cod = codigo_regiao_americas(df["latitude"].to_numpy(), df["longitude"].to_numpy())
        df = df[cod == 0].copy().reset_index(drop=True)

    return df.reset_index(drop=True)


def obter_nurdle_patrol_noaa(
    bbox: tuple[float, float, float, float] | None = BBOX_AMERICAS,
    apenas_america_norte_central: bool = False,
    apenas_positivos: bool = True,
) -> pd.DataFrame:
    """
    Lê do GeoJSON local da NOAA exclusivamente a camada de Ciência Cidadã
    `Medium == 'Nurdle Patrol'` com `Unit == 'pieces/10 mins'`, sem misturar nem
    converter para `items/m3`.

    Retorna DataFrame no esquema canônico `COLUNAS_PADRAO` com `unidade = 'pieces/10 mins'`.
    """
    df_raw = _carregar_noaa_geojson()
    m_np = (
        (df_raw["Medium"].astype(str).str.strip() == "Nurdle Patrol")
        & (df_raw["Unit"].astype(str).str.strip() == "pieces/10 mins")
    )
    sub = df_raw[m_np].copy()
    id_col = "ObjectId" if "ObjectId" in sub.columns else "UniqueID"

    val_np = pd.to_numeric(sub["Standardized_Nurdle__Amount"], errors="coerce").fillna(
        pd.to_numeric(sub["Microplastics_measurement"], errors="coerce")
    )

    df = pd.DataFrame(
        {
            "site_id": "noaa_np_" + sub[id_col].astype(str),
            "latitude": pd.to_numeric(sub["Latitude__degree_"], errors="coerce"),
            "longitude": pd.to_numeric(sub["Longitude_degree_"], errors="coerce"),
            "densidade_microplasticos": val_np,
            "unidade": "pieces/10 mins",
            "unidade_original": "pieces/10 mins",
            "meio": "Nurdle Patrol",
            "fonte": "noaa_nurdle_patrol",
        }
    )
    df = df[df["latitude"].notna() & df["longitude"].notna()].copy()
    if apenas_positivos:
        df = df[df["densidade_microplasticos"] > 0].copy()

    if bbox is not None:
        df = _recortar_bbox(df, bbox)

    if apenas_america_norte_central and not df.empty:
        from calculos.geometria import codigo_regiao_americas

        cod = codigo_regiao_americas(df["latitude"].to_numpy(), df["longitude"].to_numpy())
        df = df[cod == 0].copy().reset_index(drop=True)

    return df[COLUNAS_PADRAO].reset_index(drop=True)


def carregar_microplasticos_globais_para_mapa() -> pd.DataFrame:
    """
    Retorna todas as estações costeiras globais (NOAA Beach `pieces/m3`, NOAA `Nurdle Patrol`
    e LITTERBASE Beach `items/m3`) para o mapa-múndi F3, identificando a fonte e o meio.
    """
    df_noaa_raw = _carregar_noaa_geojson()
    m_cost = df_noaa_raw["Medium"].isin(["Beach", "Nurdle Patrol"])
    sub = df_noaa_raw[m_cost].copy()
    df_noaa = pd.DataFrame(
        {
            "latitude": pd.to_numeric(sub["Latitude__degree_"], errors="coerce"),
            "longitude": pd.to_numeric(sub["Longitude_degree_"], errors="coerce"),
            "densidade_microplasticos": pd.to_numeric(
                sub["Microplastics_measurement"], errors="coerce"
            ),
            "unidade": sub["Unit"].astype(str).str.strip(),
            "meio": sub["Medium"].astype(str).str.strip(),
            "fonte": "noaa",
        }
    ).dropna(subset=["latitude", "longitude"])

    df_lb_raw = _baixar_camada_litterbase_wfs(layer="itemsm3", size="micro", habitat="Beach")
    if not df_lb_raw.empty:
        df_lb = pd.DataFrame(
            {
                "latitude": pd.to_numeric(df_lb_raw["latitude"], errors="coerce"),
                "longitude": pd.to_numeric(df_lb_raw["longitude"], errors="coerce"),
                "densidade_microplasticos": pd.to_numeric(df_lb_raw["loc_value"], errors="coerce"),
                "unidade": "items/m3",
                "meio": "Beach (LITTERBASE)",
                "fonte": "litterbase",
            }
        ).dropna(subset=["latitude", "longitude"])
        return pd.concat([df_noaa, df_lb], ignore_index=True)
    return df_noaa
