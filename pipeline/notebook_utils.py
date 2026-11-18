"""
pipeline/notebook_utils.py
--------------------------
Funcoes auxiliares enxutas para os notebooks de analise da Lei de Potencia:
  - Carregamento e separacao dos 4 tipos de lixo da base NOAA c/ cota GEBCO.
  - Identificacao das porcoes de terra ativas via `externos/GebcoCutter`.
  - Associacao da media aritmetica no circulo de raio R em cada sink e filtro
    apenas das bacias que possuem valor associado ao sink.
  - Ajuste OLS log-log (Lei de Potencia) e plotagem padronizada.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from osgeo import gdal
from scipy import stats
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

from geracao_graficos.estilo_erneson import aplicar_estilo, rotulo_painel, salvar
from pipeline.gebco_cutter import (
    REPO_ROOT,
    executar_percolacao_gebco_cutter,
    processar_porcao_terra_gebco_cutter,
    recortar_cluster_gebco_cutter,
)
from pipeline.ipba_runner import EARTH_RADIUS_KM

NOAA_GEOJSON = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
GEBCO_TIF = REPO_ROOT / "dados" / "gebco" / "gebco_2026_sub_ice_geotiff.tif"
GPW_TIF = REPO_ROOT / "dados" / "gpw_data" / "gpw_v4_population_count_rev11_2020_30_sec.tif"
CACHE_PERC = REPO_ROOT / "relatorios" / "cache_percolacao_gebco_h0.npz"
PORCOES_DIR = REPO_ROOT / "relatorios" / "porcoes_terra"


def to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def carregar_noaa_por_tipo() -> dict[str, pd.DataFrame]:
    """
    Carrega o GeoJSON da NOAA, anota a elevacao `gebco_h` e separa os 4 tipos de lixo:
      - `Ocean water`: malha padrao 0,02-5,0 mm (pieces/m3), filtrado no mar (`gebco_h <= 0`)
      - `Ocean sediment`: sedimento marinho (pieces kg-1 d.w.), filtrado no mar (`gebco_h <= 0`)
      - `Nurdle Patrol`: pellets industriais na costa/praia (pieces/10 mins)
      - `Beach`: areia de praia (pieces/m3)
    """
    with open(NOAA_GEOJSON, "r", encoding="utf-8") as f:
        gj = json.load(f)

    df = pd.DataFrame([ft["properties"] for ft in gj["features"]])
    df["lat"] = pd.to_numeric(df["Latitude__degree_"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude_degree_"], errors="coerce")
    df["mesh_mm"] = pd.to_numeric(df["Mesh_size__mm_"], errors="coerce")
    df["mp_val"] = pd.to_numeric(df["Microplastics_measurement"], errors="coerce")
    df["nurdle_val"] = pd.to_numeric(df["Standardized_Nurdle__Amount"], errors="coerce")
    df["valor"] = np.where(df["Medium"] == "Nurdle Patrol", df["nurdle_val"], df["mp_val"])
    df = df[df["lat"].notna() & df["lon"].notna() & df["valor"].notna()].copy().reset_index(drop=True)

    gdal.UseExceptions()
    ds = gdal.Open(str(GEBCO_TIF), gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    dem = ds.GetRasterBand(1).ReadAsArray()
    cols = np.clip(np.floor((df["lon"].to_numpy() - gt[0]) / gt[1]).astype(int), 0, ds.RasterXSize - 1)
    rows = np.clip(np.floor((gt[3] - df["lat"].to_numpy()) / abs(gt[5])).astype(int), 0, ds.RasterYSize - 1)
    df["gebco_h"] = dem[rows, cols].astype(float)
    del ds

    return {
        "Ocean water": df[
            (df["Medium"] == "Ocean water")
            & (df["Unit"] == "pieces/m3")
            & (df["mesh_mm"].between(0.02, 5.0))
            & (df["gebco_h"] <= 0.0)
        ].reset_index(drop=True),
        "Ocean sediment": df[
            (df["Medium"] == "Ocean sediment") & (df["gebco_h"] <= 0.0)
        ].reset_index(drop=True),
        "Nurdle Patrol": df[df["Medium"] == "Nurdle Patrol"].reset_index(drop=True),
        "Beach": df[(df["Medium"] == "Beach") & (df["Unit"] == "pieces/m3")].reset_index(drop=True),
    }


def identificar_porcoes_ativas(
    pontos_df: pd.DataFrame,
    max_r_km: float = 50.0,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, list[int]]:
    """Segmenta o globo via `externos/GebcoCutter` e lista as porcoes de terra proximas aos pontos."""
    info, arr, label, vals, _ = executar_percolacao_gebco_cutter(GEBCO_TIF, h_star=0.0, cache_npz=CACHE_PERC)
    gt = info["GeoTransform"]
    p = arr > 0
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    coast = p & binary_dilation(~p, structure=struct4)
    sr, sc = np.nonzero(coast)
    slats = gt[3] + (sr + 0.5) * gt[5]
    slons = ((gt[0] + (sc + 0.5) * gt[1] + 180.0) % 360.0) - 180.0

    tree_p = cKDTree(to_xyz(pontos_df["lat"].to_numpy(), pontos_df["lon"].to_numpy()))
    chord = float(2.0 * np.sin((max_r_km / EARTH_RADIUS_KM) / 2.0))
    has_pts = np.array([len(x) > 0 for x in tree_p.query_ball_point(to_xyz(slats, slons), r=chord)])
    ativos_set = {int(v) for v in np.unique(label[sr, sc][has_pts])}
    porcoes_ordenadas = [int(v) for v in vals if int(v) in ativos_set]
    return info, arr, label, porcoes_ordenadas


def processar_ou_carregar_porcao(
    val: int,
    info: dict[str, Any],
    arr: np.ndarray,
    label: np.ndarray,
) -> pd.DataFrame:
    """Executa (ou le do cache) `externos/IPBA` + `externos/CCA` para uma porcao de terra `val`."""
    PORCOES_DIR.mkdir(parents=True, exist_ok=True)
    cache_csv = PORCOES_DIR / f"bacias_porcao_{int(val)}.csv"
    if cache_csv.is_file() and cache_csv.stat().st_size > 10:
        return pd.read_csv(cache_csv)
    arr_c, cl_c, info_c, jmin, jmax, cols = recortar_cluster_gebco_cutter(info, arr, label, int(val))
    return processar_porcao_terra_gebco_cutter(int(val), arr_c, cl_c, info_c, jmin, jmax, cols, GPW_TIF, out_dir=PORCOES_DIR)


def media_por_sink_raio(
    bacias_df: pd.DataFrame,
    pontos_df: pd.DataFrame,
    r_km: float,
) -> pd.DataFrame:
    """
    Calcula a MEDIA aritmetica simples dos pontos dentro do raio `r_km` de cada sink
    e retorna APENAS as bacias que obtiveram valor positivo (`mp_mean > 0`).
    """
    out = bacias_df.copy().reset_index(drop=True)
    tree_p = cKDTree(to_xyz(pontos_df["lat"].to_numpy(), pontos_df["lon"].to_numpy()))
    chord = float(2.0 * np.sin((float(r_km) / EARTH_RADIUS_KM) / 2.0))
    p_lists = tree_p.query_ball_point(to_xyz(out["latitude"].to_numpy(), out["longitude"].to_numpy()), r=chord)

    vals = pontos_df["valor"].to_numpy()
    means = np.full(len(out), np.nan, dtype=float)
    n_pts = np.zeros(len(out), dtype=int)
    for i, idx_list in enumerate(p_lists):
        if idx_list:
            v = vals[np.array(idx_list, dtype=int)]
            means[i] = float(np.mean(v))
            n_pts[i] = len(v)

    out["mp_mean"] = means
    out["n_points"] = n_pts
    out["flux"] = out["mp_mean"] * out["area_km2"]
    return out[(out["mp_mean"] > 0) & np.isfinite(out["mp_mean"])].copy().reset_index(drop=True)


def ajustar_lei_potencia(
    df_pareado: pd.DataFrame,
    pop_col: str = "total_population",
    y_col: str = "mp_mean",
) -> dict[str, Any]:
    """Ajusta regressao linear OLS em log10(pop_col) vs log10(y_col)."""
    sub = df_pareado[(df_pareado[pop_col] > 0) & (df_pareado[y_col] > 0) & np.isfinite(df_pareado[y_col])].copy()
    n = len(sub)
    if n < 4:
        return {"N": n, "beta": np.nan, "se": np.nan, "r": np.nan, "r2_pct": np.nan, "p_value": np.nan, "rho": np.nan, "r2_bins_pct": np.nan}

    x = np.log10(sub[pop_col].to_numpy())
    y = np.log10(sub[y_col].to_numpy())
    sl, ic, rv, pv, se = stats.linregress(x, y)
    rho, _ = stats.spearmanr(x, y)

    r2_bins_pct = np.nan
    if n >= 20:
        sub["bin"] = pd.cut(x, bins=np.linspace(x.min(), x.max(), 9), include_lowest=True)
        b_df = (
            sub.groupby("bin", observed=True)
            .agg(n_b=(y_col, "count"), xm=(pop_col, lambda s: float(np.mean(np.log10(s)))), ym=(y_col, lambda s: float(np.mean(np.log10(s)))))
            .dropna()
        )
        b_df = b_df[b_df["n_b"] >= 3]
        if len(b_df) >= 4:
            _, _, rv_b, _, _ = stats.linregress(b_df["xm"].to_numpy(), b_df["ym"].to_numpy())
            r2_bins_pct = float(rv_b**2 * 100.0)

    return {
        "N": int(n),
        "beta": float(sl),
        "intercept": float(ic),
        "se": float(se),
        "r": float(rv),
        "r2_pct": float(rv**2 * 100.0),
        "p_value": float(pv),
        "rho": float(rho),
        "r2_bins_pct": r2_bins_pct,
    }


def plotar_lei_potencia(
    ax: plt.Axes,
    df_pareado: pd.DataFrame,
    pop_col: str,
    y_col: str,
    titulo: str,
    xlabel: str,
    ylabel: str,
    cor_pts: str = "#0284c7",
    cor_reta: str = "#0369a1",
) -> dict[str, Any]:
    """Plota o scatter log-log com reta OLS, banda IC 95% e medias por faixa log."""
    aplicar_estilo()
    sub = df_pareado[(df_pareado[pop_col] > 0) & (df_pareado[y_col] > 0) & np.isfinite(df_pareado[y_col])].copy()
    res = ajustar_lei_potencia(sub, pop_col, y_col)
    if res["N"] < 4:
        ax.text(0.5, 0.5, "Amostra insuficiente", ha="center", va="center", transform=ax.transAxes)
        return res

    x = np.log10(sub[pop_col].to_numpy())
    y = np.log10(sub[y_col].to_numpy())
    ax.scatter(x, y, edgecolors=cor_pts, facecolors="none", s=30, linewidths=1.0, alpha=0.72, label=f"Bacias (N = {res['N']})")

    xl = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
    yl = res["intercept"] + res["beta"] * xl
    ax.plot(xl, yl, color=cor_reta, linewidth=2.0, label="Ajuste Lei de Potência")

    n = res["N"]
    t_val = float(stats.t.ppf(0.975, df=n - 2))
    s_err = float(np.sqrt(np.sum((y - (res["intercept"] + res["beta"] * x)) ** 2) / (n - 2)))
    ci = t_val * s_err * np.sqrt(1.0 / n + (xl - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    ax.fill_between(xl, yl - ci, yl + ci, color=cor_reta, alpha=0.16)

    if n >= 20:
        sub["bin"] = pd.cut(x, bins=np.linspace(x.min(), x.max(), 9), include_lowest=True)
        b_df = (
            sub.groupby("bin", observed=True)
            .agg(n_b=(y_col, "count"), xm=(pop_col, lambda s: float(np.mean(np.log10(s)))), ym=(y_col, lambda s: float(np.mean(np.log10(s)))))
            .dropna()
        )
        b_df = b_df[b_df["n_b"] >= 3]
        if len(b_df) >= 4:
            ax.plot(b_df["xm"], b_df["ym"], "ko", markersize=4.8, zorder=7, label="Média por faixa log")

    p_txt = "p < 0,0001" if res["p_value"] < 1e-4 else f"p = {res['p_value']:.4f}".replace(".", ",")
    bins_txt = f"\nR²_bins = {res['r2_bins_pct']:.1f}%".replace(".", ",") if np.isfinite(res["r2_bins_pct"]) else ""
    stat_box = (
        f"β = {res['beta']:+.2f} ± {res['se']:.2f}\n"
        f"R² = {res['r2_pct']:.1f}% ({p_txt})\n"
        f"Spearman ρ = {res['rho']:+.2f}{bins_txt}"
    ).replace(".", ",")

    ax.text(
        0.04, 0.95, stat_box, transform=ax.transAxes, ha="left", va="top", fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#cbd5e1", alpha=0.94),
    )
    ax.set_title(titulo, fontsize=10.2, pad=8)
    ax.set_xlabel(xlabel, fontsize=9.6)
    ax.set_ylabel(ylabel, fontsize=9.6)
    y_span = max(float(y.max() - y.min()), 2.8)
    ax.set_ylim(float(y.min()) - 0.25 * y_span, float(y.max()) + 0.42 * y_span)
    ax.legend(loc="lower right", fontsize=7.2, frameon=True, facecolor="white", edgecolor="#cccccc")
    return res
