"""
scripts/analise_r22km_mar_territorial.py
----------------------------------------
Implementa integralmente o protocolo definido pelo Prof. Dr. Erneson Oliveira nos
novos audios (17/11/2026) utilizando os algoritmos oficiais da pasta `externos/`:
  1. Divisao do globo em porcoes de terra conectadas via `externos/GebcoCutter`
     (`Percolation.site_percolation()` + recorte com periodicidade longitudinal).
  2. Extracao das bacias hidrográficas em cada porcao de terra via `externos/IPBA`
     (`Lattice`, `Heap`, `SetDrainageBasins`, corte h = 0) e area esferica exata
     (Triangulos Esfericos de L'Huilier + Haversine).
  3. Delimitacao das manchas urbanas em cada porcao de terra via `externos/CCA`
     (`burn` calibrado por Lei de Zipf no GPW v4 alinhado 4x4).
  4. Filtro estrito de MAR nos pontos de Agua Oceanica (malha padrao 0,02-5,0 mm):
     mantem apenas pontos em pixels com elevacao `GEBCO <= 0` (n = 14.363),
     descartando pontos rio acima / dentro do continente (`GEBCO > 0`, n = 410).
  5. Raio fixo R = 22 km (mar territorial = 12 milhas nauticas) em cada sink:
     calcula a MEDIA aritmetica simples (pieces/m3) sem descartar sobreposicao
     entre circulos vizinhos, junta todas as porcoes de terra do mundo e testa a
     Lei de Potencia.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon
from osgeo import gdal
from scipy import stats
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from geracao_graficos.estilo_erneson import (
    WORLD_GEOJSON,
    aplicar_estilo,
    eixo_mapa,
    rotulo_painel,
    salvar,
)
from pipeline.gebco_cutter import (
    executar_percolacao_gebco_cutter,
    gerar_bacias_globais_gebco_cutter,
)
from pipeline.ipba_runner import EARTH_RADIUS_KM

RAIO_MAR_TERRITORIAL_KM = 22.0


def to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def fmt_int_pt(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def _ltx_num(v: float, dec: int = 2, signed: bool = False) -> str:
    s = f"{v:+.{dec}f}" if signed else f"{v:.{dec}f}"
    return s.replace(".", "{,}")


def carregar_ocean_water_filtrado_mar() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega Agua Oceanica (malha padrao 0,02-5,0 mm, pieces/m3) e classifica cada ponto
    pelo raster GEBCO 2026:
      - `is_sea == True`: pixel com `elevacao GEBCO <= 0` (com certeza no mar)
      - `is_sea == False`: pixel com `elevacao GEBCO > 0` (rio acima / continente)
    """
    noaa_path = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
    with open(noaa_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    df = pd.DataFrame([ft["properties"] for ft in gj["features"]])
    df["lat"] = pd.to_numeric(df["Latitude__degree_"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude_degree_"], errors="coerce")
    df["mesh_mm"] = pd.to_numeric(df["Mesh_size__mm_"], errors="coerce")
    df["valor"] = pd.to_numeric(df["Microplastics_measurement"], errors="coerce")

    ow = df[
        (df["Medium"] == "Ocean water")
        & (df["Unit"] == "pieces/m3")
        & (df["mesh_mm"] >= 0.02)
        & (df["mesh_mm"] <= 5.0)
        & df["lat"].notna()
        & df["lon"].notna()
        & df["valor"].notna()
    ].copy().reset_index(drop=True)

    gdal.UseExceptions()
    gebco_path = REPO_ROOT / "dados" / "gebco" / "gebco_2026_sub_ice_geotiff.tif"
    ds = gdal.Open(str(gebco_path), gdal.GA_ReadOnly)
    gt = ds.GetGeoTransform()
    dem = ds.GetRasterBand(1).ReadAsArray()

    cols = np.clip(np.floor((ow["lon"].to_numpy() - gt[0]) / gt[1]).astype(int), 0, ds.RasterXSize - 1)
    rows = np.clip(np.floor((gt[3] - ow["lat"].to_numpy()) / abs(gt[5])).astype(int), 0, ds.RasterYSize - 1)
    ow["gebco_elevation_m"] = dem[rows, cols].astype(float)
    ow["is_sea"] = ow["gebco_elevation_m"] <= 0.0
    del ds

    ow_sea = ow[ow["is_sea"]].copy().reset_index(drop=True)
    return ow, ow_sea


def identificar_porcoes_terra_ativas(
    ow_sea: pd.DataFrame,
    max_radius_km: float = 50.0,
) -> set[int]:
    """
    Usa `externos/GebcoCutter` para identificar todas as porcoes de terra conectadas
    que possuem pelo menos um sink costeiro (h = 0) a ate `max_radius_km` de uma
    coleta marinha (`ow_sea`).
    """
    gebco_path = REPO_ROOT / "dados" / "gebco" / "gebco_2026_sub_ice_geotiff.tif"
    cache_perc = REPO_ROOT / "relatorios" / "cache_percolacao_gebco_h0.npz"
    info, arr, label, _, _ = executar_percolacao_gebco_cutter(gebco_path, h_star=0.0, cache_npz=cache_perc)

    gt = info["GeoTransform"]
    p = arr > 0
    struct4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    coast = p & binary_dilation(~p, structure=struct4)
    sr, sc = np.nonzero(coast)
    slats = gt[3] + (sr + 0.5) * gt[5]
    slons = ((gt[0] + (sc + 0.5) * gt[1] + 180.0) % 360.0) - 180.0
    svals = label[sr, sc]

    tree_p = cKDTree(to_xyz(ow_sea["lat"].to_numpy(), ow_sea["lon"].to_numpy()))
    chord = float(2.0 * np.sin((max_radius_km / EARTH_RADIUS_KM) / 2.0))
    p_lists = tree_p.query_ball_point(to_xyz(slats, slons), r=chord)
    has_pts = np.array([len(x) > 0 for x in p_lists])
    return {int(v) for v in np.unique(svals[has_pts])}


def carregar_ou_gerar_bacias_globais_externos(ow_sea: pd.DataFrame) -> pd.DataFrame:
    """
    Carrega (ou gera com `externos/GebcoCutter` + `externos/IPBA` + `externos/CCA`)
    o catalogo completo de bacias hidrográficas das porcoes de terra ativas do mundo.
    """
    global_csv = REPO_ROOT / "relatorios" / "bacias_globais_gebco_cutter_ipba_cca.csv"
    if global_csv.is_file() and global_csv.stat().st_size > 1000:
        return pd.read_csv(global_csv)

    gebco_path = REPO_ROOT / "dados" / "gebco" / "gebco_2026_sub_ice_geotiff.tif"
    gpw_path = REPO_ROOT / "dados" / "gpw_data" / "gpw_v4_population_count_rev11_2020_30_sec.tif"
    cache_perc = REPO_ROOT / "relatorios" / "cache_percolacao_gebco_h0.npz"

    active_vals = identificar_porcoes_terra_ativas(ow_sea, max_radius_km=50.0)
    print(f"   Porcoes de terra GebcoCutter ativas (ate R=50 km): {len(active_vals)}", flush=True)

    df_all = gerar_bacias_globais_gebco_cutter(
        gebco_path=gebco_path,
        gpw_path=gpw_path,
        target_cluster_vals=active_vals,
        cache_percolation_npz=cache_perc,
        out_dir=REPO_ROOT / "relatorios" / "porcoes_terra",
    )
    df_all.to_csv(global_csv, index=False)
    print(f"   Catalogo global salvo em: {global_csv} ({len(df_all)} bacias totais)", flush=True)
    return df_all


def associar_media_r22km_mar(
    bacias_df: pd.DataFrame,
    ow_sea: pd.DataFrame,
    r_km: float = RAIO_MAR_TERRITORIAL_KM,
) -> tuple[pd.DataFrame, int]:
    """
    Para cada sink em `bacias_df`, toma todos os pontos de `ow_sea` (apenas no mar)
    a ate `r_km` (22 km) e calcula a MEDIA aritmetica simples (pieces/m3),
    permitindo sobreposicao entre circulos vizinhos.
    """
    out = bacias_df.copy().reset_index(drop=True)
    s_xyz = to_xyz(out["latitude"].to_numpy(), out["longitude"].to_numpy())
    p_xyz = to_xyz(ow_sea["lat"].to_numpy(), ow_sea["lon"].to_numpy())

    tree_p = cKDTree(p_xyz)
    chord = float(2.0 * np.sin((r_km / EARTH_RADIUS_KM) / 2.0))
    p_lists = tree_p.query_ball_point(s_xyz, r=chord)

    vals = ow_sea["valor"].to_numpy()
    means = np.full(len(out), np.nan, dtype=float)
    stds = np.full(len(out), np.nan, dtype=float)
    n_pts = np.zeros(len(out), dtype=int)
    used_pts_set: set[int] = set()

    for i, idx_list in enumerate(p_lists):
        if len(idx_list) > 0:
            arr_idx = np.array(idx_list, dtype=int)
            v = vals[arr_idx]
            means[i] = float(np.mean(v))
            stds[i] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            n_pts[i] = len(v)
            used_pts_set.update(idx_list)

    out["mp_mean_r22km"] = means
    out["mp_std_r22km"] = stds
    out["n_points_r22km"] = n_pts
    out["flux_r22km"] = out["mp_mean_r22km"] * out["area_km2"]
    return out, len(used_pts_set)


def _desenhar_continentes(ax: plt.Axes) -> None:
    with open(WORLD_GEOJSON, "r", encoding="utf-8") as f:
        gj = json.load(f)
    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        gtype = geom.get("type")
        coords = geom.get("coordinates", [])
        if gtype == "Polygon":
            for ring in coords:
                ax.add_patch(Polygon(ring, facecolor="#e5e5e5", edgecolor="#cccccc", linewidth=0.35, zorder=1))
        elif gtype == "MultiPolygon":
            for poly_coords in coords:
                for ring in poly_coords:
                    ax.add_patch(Polygon(ring, facecolor="#e5e5e5", edgecolor="#cccccc", linewidth=0.35, zorder=1))


def _plotar_painel_lei_potencia(
    ax: plt.Axes,
    df_valid: pd.DataFrame,
    pop_col: str,
    y_col: str,
    titulo: str,
    xlabel: str,
    ylabel: str,
    cor_reta: str = "#047857",
) -> dict[str, Any]:
    estilos_regiao = {
        "Am. Norte/Central": ("o", "C0"),
        "Am. do Sul": ("s", "C2"),
        "Europa": ("^", "C3"),
        "África": ("D", "C4"),
        "Ásia": ("P", "C5"),
        "Austrália/Oceania": ("v", "C1"),
    }

    sub = df_valid[(df_valid[pop_col] > 0) & (df_valid[y_col] > 0) & np.isfinite(df_valid[y_col])].copy()
    x = np.log10(sub[pop_col].to_numpy())
    y = np.log10(sub[y_col].to_numpy())

    for reg_key, (mk, cor) in estilos_regiao.items():
        grp = sub[sub["macroregiao"] == reg_key]
        if len(grp) > 0:
            ax.scatter(
                np.log10(grp[pop_col].to_numpy()),
                np.log10(grp[y_col].to_numpy()),
                edgecolors=cor,
                facecolors="none",
                marker=mk,
                s=28,
                linewidths=0.9,
                alpha=0.58,
                label=f"{reg_key} (n={len(grp)})",
                zorder=3,
            )

    sl, ic, rv, pv, se = stats.linregress(x, y)
    rho, prho = stats.spearmanr(x, y)

    xl = np.linspace(x.min() - 0.1, x.max() + 0.1, 100)
    yl = ic + sl * xl
    ax.plot(xl, yl, color=cor_reta, linewidth=2.0, zorder=7, label="Ajuste OLS (Lei de Potência)")

    n = len(x)
    t_val = float(stats.t.ppf(0.975, df=n - 2))
    s_err = float(np.sqrt(np.sum((y - (ic + sl * x)) ** 2) / (n - 2)))
    ci = t_val * s_err * np.sqrt(1.0 / n + (xl - np.mean(x)) ** 2 / np.sum((x - np.mean(x)) ** 2))
    ax.fill_between(xl, yl - ci, yl + ci, color=cor_reta, alpha=0.18, zorder=2)

    bins = np.linspace(x.min(), x.max(), 9)
    sub["bin"] = pd.cut(x, bins=bins, include_lowest=True)
    b_df = (
        sub.groupby("bin", observed=True)
        .agg(
            n_bin=(y_col, "count"),
            xm=(pop_col, lambda s: float(np.mean(np.log10(s)))),
            ym=(y_col, lambda s: float(np.mean(np.log10(s)))),
            ysem=(y_col, lambda s: float(np.std(np.log10(s), ddof=1) / np.sqrt(len(s))) if len(s) > 1 else 0.0),
        )
        .dropna()
    )
    b_df = b_df[b_df["n_bin"] >= 3]
    r2_bin_str = ""
    if len(b_df) >= 4:
        ax.errorbar(
            b_df["xm"].to_numpy(),
            b_df["ym"].to_numpy(),
            yerr=1.96 * b_df["ysem"].to_numpy(),
            fmt="o",
            color="#111111",
            ecolor="#111111",
            markersize=5.2,
            elinewidth=1.2,
            capsize=2.6,
            zorder=8,
            label="Média por faixa log ± IC 95%",
        )
        sl_b, _, rv_b, pv_b, _ = stats.linregress(b_df["xm"].to_numpy(), b_df["ym"].to_numpy())
        p_b_str = "p < 0{,}001" if pv_b < 0.001 else f"p = {_ltx_num(pv_b, 3, False)}"
        r2_bin_str = f"\nMédias log-bins: $R^2_{{\\mathrm{{bins}}}} = {_ltx_num(rv_b**2 * 100, 1, False)}\\%$ ($r = {_ltx_num(rv_b, 2, True)}$, ${p_b_str}$)"

    p_str = "p < 0{,}0001" if pv < 0.0001 else f"p = {_ltx_num(pv, 4, False)}"
    prho_str = "p < 0{,}0001" if prho < 0.0001 else f"p = {_ltx_num(prho, 4, False)}"
    texto_stat = (
        f"$\\beta = {_ltx_num(sl, 2, True)} \\pm {_ltx_num(se, 2, False)}$\n"
        f"$R^2 = {_ltx_num(rv**2 * 100, 1, False)}\\%$ ($r = {_ltx_num(rv, 2, True)}$, ${p_str}$)\n"
        f"Spearman $\\rho = {_ltx_num(rho, 2, True)}$ (${prho_str}$)"
        f"{r2_bin_str}\n"
        f"$N = {n}$ bacias pareadas ($R = 22$ km)"
    )
    ax.text(
        0.04,
        0.96,
        texto_stat,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#cccccc", alpha=0.94),
        zorder=10,
    )

    y_span = max(y.max() - y.min(), 3.5)
    ax.set_ylim(y.min() - 0.36 * y_span, y.max() + 0.44 * y_span)
    ax.set_title(titulo, fontsize=10.1, pad=9)
    ax.set_xlabel(xlabel, fontsize=9.8)
    ax.set_ylabel(ylabel, fontsize=9.8)
    ax.legend(
        loc="lower right",
        ncol=2,
        fontsize=6.9,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.94,
    )

    return {
        "N": n,
        "beta": sl,
        "se": se,
        "r": rv,
        "r2_pct": rv**2 * 100.0,
        "p_value": pv,
        "spearman_rho": rho,
        "spearman_p": prho,
    }


def gerar_figura_r22km_mar_territorial(
    ow_all: pd.DataFrame,
    ow_sea: pd.DataFrame,
    df_main_paired: pd.DataFrame,
    n_pts_main: int,
) -> Path:
    """
    Gera a figura 2x2 do protocolo R = 22 km (mar territorial, apenas pontos no mar,
    com divisao do globo via `externos/GebcoCutter`, bacias via `externos/IPBA`
    e manchas urbanas via `externos/CCA`):
      - Painel (a): Mapa-mundi do filtro de mar (`h <= 0`) e fozes IPBA ativas a 22 km.
      - Painel (b): Bacias IPBA (Area >= 100 km²): Pop. da bacia vs. Media em 22 km.
      - Painel (c): Bacias c/ Mancha Urbana CCA (Area >= 100 km²): Pop. mancha vs. Media em 22 km.
      - Painel (d): Bacias IPBA (Area >= 100 km²): Pop. da bacia vs. Fluxo (Media_22km * Area).
    """
    aplicar_estilo()

    fig, axes = plt.subplots(2, 2, figsize=(14.6, 9.9), gridspec_kw={"wspace": 0.23, "hspace": 0.29})
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    eixo_mapa(ax_a, (-180, 180, -60, 82))
    _desenhar_continentes(ax_a)

    ow_land = ow_all[~ow_all["is_sea"]]
    sinks_ativos = df_main_paired[
        (df_main_paired["mp_mean_r22km"] > 0) & (df_main_paired["total_population"] > 0)
    ]

    ax_a.scatter(
        ow_sea["lon"],
        ow_sea["lat"],
        s=3.5,
        color="#93c5fd",
        alpha=0.45,
        linewidths=0,
        zorder=3,
        label=f"Pontos no mar GEBCO h ≤ 0 (n = {fmt_int_pt(len(ow_sea))})",
    )
    ax_a.scatter(
        ow_land["lon"],
        ow_land["lat"],
        s=12.0,
        color="#f59e0b",
        marker="x",
        linewidths=0.8,
        alpha=0.90,
        zorder=6,
        label=f"Descartados rio acima / terra h > 0 (n = {fmt_int_pt(len(ow_land))})",
    )
    ax_a.scatter(
        sinks_ativos["longitude"],
        sinks_ativos["latitude"],
        s=14.0,
        edgecolors="#b91c1c",
        facecolors="none",
        marker="o",
        linewidths=0.85,
        alpha=0.88,
        zorder=8,
        label=f"Fozes GebcoCutter+IPBA (A ≥ 100 km², R ≤ 22 km: N = {fmt_int_pt(len(sinks_ativos))}, {fmt_int_pt(n_pts_main)} pts)",
    )

    ax_a.set_title("Filtro de mar (h ≤ 0) e fozes GebcoCutter+IPBA em R = 22 km", fontsize=10.1, pad=9)
    leg_a = ax_a.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=1,
        fontsize=7.3,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.94,
        markerscale=1.4,
    )
    for h in leg_a.legend_handles:
        h.set_alpha(1.0)
    rotulo_painel(ax_a, "(a)", dentro=True, x=0.02, y=0.96)

    _plotar_painel_lei_potencia(
        ax_b,
        df_main_paired,
        pop_col="total_population",
        y_col="mp_mean_r22km",
        titulo="Bacias IPBA (A ≥ 100 km²): População da bacia vs. Média em R = 22 km",
        xlabel=r"$\log_{10} P_{\mathrm{bacia}}$ (Habitantes na bacia hidrográfica)",
        ylabel=r"$\log_{10} \bar{C}_{22\mathrm{km}}$ (Média no mar em $\mathrm{pieces/m^3}$)",
        cor_reta="#1d4ed8",
    )
    rotulo_painel(ax_b, "(b)", dentro=False, x=-0.13, y=1.03)

    _plotar_painel_lei_potencia(
        ax_c,
        df_main_paired,
        pop_col="urban_population",
        y_col="mp_mean_r22km",
        titulo="Bacias c/ Mancha Urbana CCA (A ≥ 100 km²): Pop. mancha vs. Média 22 km",
        xlabel=r"$\log_{10} P_{\mathrm{mancha}}$ (População urbana CCA na bacia)",
        ylabel=r"$\log_{10} \bar{C}_{22\mathrm{km}}$ (Média no mar em $\mathrm{pieces/m^3}$)",
        cor_reta="#047857",
    )
    rotulo_painel(ax_c, "(c)", dentro=False, x=-0.13, y=1.03)

    _plotar_painel_lei_potencia(
        ax_d,
        df_main_paired,
        pop_col="total_population",
        y_col="flux_r22km",
        titulo=r"Bacias IPBA (A ≥ 100 km²): População vs. Fluxo ($\bar{C}_{22\mathrm{km}} \times A_{\mathrm{bacia}}$)",
        xlabel=r"$\log_{10} P_{\mathrm{bacia}}$ (Habitantes na bacia hidrográfica)",
        ylabel=r"$\log_{10} (\bar{C}_{22\mathrm{km}} \cdot A_{\mathrm{bacia}})$ (Índice de fluxo exportado)",
        cor_reta="#b91c1c",
    )
    rotulo_painel(ax_d, "(d)", dentro=False, x=-0.13, y=1.03)

    out_base = REPO_ROOT / "relatorios" / "figuras" / "noaa_r22km_mar_territorial_lei_potencia"
    salvar(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".png")


if __name__ == "__main__":
    print("1. Carregando Agua Oceanica (malha padrao 0,02-5,0 mm) e aplicando filtro de mar (GEBCO h <= 0)...")
    ow_all, ow_sea = carregar_ocean_water_filtrado_mar()
    n_land = len(ow_all) - len(ow_sea)
    print(f"   Total: {len(ow_all)} | No mar (h <= 0): {len(ow_sea)} | Descartados rio acima/continente (h > 0): {n_land}")

    print("2. Segmentando o globo com externos/GebcoCutter e rodando externos/IPBA + externos/CCA...")
    df_mundo_100 = carregar_ou_gerar_bacias_globais_externos(ow_sea)
    df_mundo_a100 = df_mundo_100[df_mundo_100["area_km2"] >= 100.0].copy().reset_index(drop=True)
    df_mundo_95g = df_mundo_100[df_mundo_100["in_top_95pct_global"]].copy().reset_index(drop=True)
    print(
        f"   Total bacias (porcoes ativas): {len(df_mundo_100)} | "
        f"Area >= 100 km2: {len(df_mundo_a100)} | Top 95% global: {len(df_mundo_95g)}"
    )

    print("3. Calculando media simples nos sinks em R = 22 km (mar territorial, apenas pontos no mar)...")
    df_a100_paired, n_pts_a100 = associar_media_r22km_mar(df_mundo_a100, ow_sea, r_km=22.0)
    df_95g_paired, n_pts_95g = associar_media_r22km_mar(df_mundo_95g, ow_sea, r_km=22.0)
    df_100_paired, n_pts_100 = associar_media_r22km_mar(df_mundo_100, ow_sea, r_km=22.0)

    positivos_a100 = df_a100_paired[(df_a100_paired["mp_mean_r22km"] > 0) & (df_a100_paired["total_population"] > 0)].copy()
    positivos_a100_urb = df_a100_paired[(df_a100_paired["mp_mean_r22km"] > 0) & (df_a100_paired["urban_population"] > 0)].copy()
    positivos_95g = df_95g_paired[(df_95g_paired["mp_mean_r22km"] > 0) & (df_95g_paired["total_population"] > 0)].copy()
    positivos_100_urb = df_100_paired[(df_100_paired["mp_mean_r22km"] > 0) & (df_100_paired["urban_population"] > 0)].copy()

    out_csv_a100 = REPO_ROOT / "relatorios" / "bacias_pareadas_r22km_mar_ipba_area100km2.csv"
    out_csv_95 = REPO_ROOT / "relatorios" / "bacias_pareadas_r22km_mar_ipba_top95.csv"
    out_csv_100 = REPO_ROOT / "relatorios" / "bacias_pareadas_r22km_mar_urbanas_100pct.csv"
    positivos_a100.to_csv(out_csv_a100, index=False)
    positivos_95g.to_csv(out_csv_95, index=False)
    positivos_100_urb.to_csv(out_csv_100, index=False)

    print(f"   [Area >= 100 km2] Fozes c/ media > 0 e pop_total > 0: {len(positivos_a100)} | c/ pop_urbana > 0: {len(positivos_a100_urb)} (pts unicos mar: {n_pts_a100})")
    print(f"   [Top 95% Global] Fozes c/ media > 0 e pop_total > 0: {len(positivos_95g)} (pts unicos mar: {n_pts_95g})")
    print(f"   [100% Sinks Urbanos CCA] Bacias c/ media > 0: {len(positivos_100_urb)} (pts unicos mar: {n_pts_100})")
    print("   Distribuicao por porcao de terra (Area >= 100 km2, N = " + str(len(positivos_a100)) + "):")
    print(positivos_a100["landmass_name"].value_counts().head(15).to_string())

    print("\n4. Gerando figura cientifica R = 22 km (externos/GebcoCutter + IPBA + CCA)...")
    fig_path = gerar_figura_r22km_mar_territorial(ow_all, ow_sea, df_a100_paired, n_pts_a100)
    print(f"   Figura salva em: {fig_path}")
