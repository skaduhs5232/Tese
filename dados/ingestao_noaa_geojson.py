"""
Ingestão da base NOAA NCEI Marine Microplastics em GeoJSON (Hub ArcGIS) para o CSV padronizado do pipeline.

Uso:
    python dados/ingestao_noaa_geojson.py
    python dados/ingestao_noaa_geojson.py --geojson dados/noaa_microplastic/<arquivo>.geojson

A geometria do GeoJSON vem em Web Mercator (EPSG:3857) e algumas features têm geometria nula;
por isso lat/lon são lidas das propriedades `Latitude__degree_` e `Longitude_degree_` (WGS84).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

DADOS_DIR = Path(__file__).resolve().parent
GEOJSON_PADRAO = DADOS_DIR / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
CSV_SAIDA = DADOS_DIR / "noaa_microplastics.csv"

MEIOS_COSTEIROS = ("Nurdle Patrol", "Beach", "Ocean sediment")

MAPA_COLUNAS = {
    "UniqueID": "unique_id",
    "Date__MM_DD_YYYY_": "data",
    "Latitude__degree_": "lat",
    "Longitude_degree_": "lon",
    "Location_Oceans": "oceano",
    "Location_Regions": "regiao",
    "Location_SubRegions": "subregiao",
    "Country": "pais",
    "State": "estado",
    "Beach_Location": "local_praia",
    "Medium": "meio",
    "Sampling_Method": "metodo_amostragem",
    "Mesh_size__mm_": "malha_mm",
    "Microplastics_measurement": "medida_original",
    "Standardized_Nurdle__Amount": "nurdle_padronizado",
    "Unit": "unidade",
    "Concentration_class_text": "classe_concentracao",
    "Short_Reference": "referencia",
    "DOI": "doi",
    "ORGANIZATION": "organizacao",
    "NCEI_Accession_No": "ncei_accession",
}


def ler_geojson(path: Path) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        gj = json.load(f)
    return pd.DataFrame([feat.get("properties") or {} for feat in gj["features"]])


def padronizar(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.rename(columns=MAPA_COLUNAS)[list(MAPA_COLUNAS.values())].copy()
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df["medida_original"] = pd.to_numeric(df["medida_original"], errors="coerce")
    df["nurdle_padronizado"] = pd.to_numeric(df["nurdle_padronizado"], errors="coerce")
    df["medida"] = df["medida_original"].fillna(df["nurdle_padronizado"])
    df["meio"] = df["meio"].astype("string").str.strip()
    df["unidade"] = df["unidade"].astype("string").str.strip()
    df["costeiro"] = df["meio"].isin(MEIOS_COSTEIROS)
    return df


def relatorio(df: pd.DataFrame) -> dict:
    n_total = len(df)
    n_oceano = int((df["meio"] == "Ocean water").sum())
    cost = df[df["costeiro"]]
    print(f"[NOAA] Registros brutos no GeoJSON          : {n_total:>7,d}")
    print(f"[NOAA] Excluídos 'Ocean water' (pelágico)   : {n_oceano:>7,d}")
    print(f"[NOAA] Retidos costeiros/terra              : {len(cost):>7,d}")
    for meio, cnt in cost["meio"].value_counts().items():
        print(f"         - meio    {meio:<20s}: {cnt:>7,d}")
    for un, cnt in cost["unidade"].value_counts().items():
        print(f"         - unidade {un:<20s}: {cnt:>7,d}")
    print(f"[NOAA] Costeiros com medida válida (>= 0)    : {int((cost['medida'] >= 0).sum()):>7,d}")

    assert df["lat"].isna().sum() == 0, "Há latitudes nulas"
    assert df["lon"].isna().sum() == 0, "Há longitudes nulas"
    assert df["lat"].between(-90, 90).all(), "Latitude fora de [-90, 90]"
    assert df["lon"].between(-180, 180).all(), "Longitude fora de [-180, 180]"
    print("[NOAA] Integridade: lat/lon sem nulos e dentro dos limites WGS84.")
    return {
        "n_total": n_total,
        "n_ocean_water": n_oceano,
        "n_costeiro": int(len(cost)),
        "por_meio": {k: int(v) for k, v in cost["meio"].value_counts().items()},
        "por_unidade": {k: int(v) for k, v in cost["unidade"].value_counts().items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Ingestão do GeoJSON NOAA para dados/noaa_microplastics.csv")
    parser.add_argument("--geojson", type=Path, default=GEOJSON_PADRAO)
    parser.add_argument("--saida", type=Path, default=CSV_SAIDA)
    args = parser.parse_args()

    df = padronizar(ler_geojson(args.geojson))
    resumo = relatorio(df)
    df.to_csv(args.saida, index=False, encoding="utf-8")
    with open(args.saida.with_suffix(".resumo.json"), "w", encoding="utf-8") as f:
        json.dump(resumo, f, indent=2, ensure_ascii=False)
    print(f"[NOAA] Base padronizada salva em {args.saida} ({len(df):,} registros, {np.sum(df['costeiro']):,} costeiros).")


if __name__ == "__main__":
    main()
