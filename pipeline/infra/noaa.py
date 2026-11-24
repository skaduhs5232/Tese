"""Ingestão do GeoJSON da NOAA e separação pelos 4 tipos de lixo marinho."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import GEBCO_TIF, NOAA_GEOJSON
from .rasters import amostrar_cota_gebco


def carregar_noaa_por_tipo(
    geojson_path: str | Path = NOAA_GEOJSON,
    gebco_path: str | Path = GEBCO_TIF,
) -> dict[str, pd.DataFrame]:
    """Lê o GeoJSON da NOAA, anota a cota GEBCO e separa os 4 tipos de lixo."""
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    df = pd.DataFrame([ft["properties"] for ft in gj["features"]])
    df["lat"] = pd.to_numeric(df["Latitude__degree_"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude_degree_"], errors="coerce")
    df["mesh_mm"] = pd.to_numeric(df["Mesh_size__mm_"], errors="coerce")
    df["mp_val"] = pd.to_numeric(df["Microplastics_measurement"], errors="coerce")
    df["nurdle_val"] = pd.to_numeric(df["Standardized_Nurdle__Amount"], errors="coerce")
    df["valor"] = np.where(df["Medium"] == "Nurdle Patrol", df["nurdle_val"], df["mp_val"])
    df = df[df["lat"].notna() & df["lon"].notna() & df["valor"].notna()].copy().reset_index(drop=True)

    df["gebco_h"] = amostrar_cota_gebco(df["lat"].to_numpy(), df["lon"].to_numpy(), gebco_path)
    mar = df["gebco_h"] <= 0.0

    return {
        "Água oceânica": df[(df["Medium"] == "Ocean water") & (df["Unit"] == "pieces/m3") & df["mesh_mm"].between(0.02, 5.0) & mar].reset_index(drop=True),
        "Areia de praia": df[(df["Medium"] == "Beach") & (df["Unit"] == "pieces/m3")].reset_index(drop=True),
        "Sedimento oceânico": df[(df["Medium"] == "Ocean sediment") & mar].reset_index(drop=True),
        "Nurdle Patrol": df[df["Medium"] == "Nurdle Patrol"].reset_index(drop=True),
    }
