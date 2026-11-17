"""
scripts/analise_noaa_distancia_fozes.py
---------------------------------------
Analise espacial dos 14.773 pontos de coleta `Ocean water` (malha 0,02-5,0 mm,
pieces/m3) da base NOAA NCEI em relacao as 5.094 fozes globais do IPBA:
  - Painel (a): Mapa-mundi distinguindo coletas perto de fozes (<= 25 km),
    em plataforma/offshore (25-200 km) e em mar aberto / giros (> 200 km).
  - Painel (b): Histograma em escala logaritmica da distancia a foz mais
    proxima + curva acumulada (CDF) com os limiares de raio (15, 25, 50, 200 km)
    e mediana (454,8 km).

Gera PNG (300 DPI) e PDF vetorial em `relatorios/figuras/`.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch, Polygon
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


def fmt_int_pt(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def fmt_dec_pt(val: float, dec: int = 1) -> str:
    return f"{val:.{dec}f}".replace(".", ",")


def to_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    return np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])


def carregar_ocean_water_com_distancias() -> tuple[pd.DataFrame, int]:
    noaa_path = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
    with open(noaa_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    df = pd.DataFrame([ft["properties"] for ft in gj["features"]])
    df["lat"] = pd.to_numeric(df["Latitude__degree_"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude_degree_"], errors="coerce")
    df["mesh"] = pd.to_numeric(df["Mesh_size__mm_"], errors="coerce")

    ow = df[(df["Medium"] == "Ocean water") & (df["mesh"] >= 0.02) & (df["mesh"] <= 5.0)].copy()

    sinks_files = sorted(glob.glob(str(REPO_ROOT / "relatorios" / "ilhas" / "sinks_raios_*.csv")))
    sinks = pd.concat([pd.read_csv(p) for p in sinks_files], ignore_index=True)

    tree = cKDTree(to_xyz(sinks["latitude"].to_numpy(), sinks["longitude"].to_numpy()))
    dists_chord, _ = tree.query(to_xyz(ow["lat"].to_numpy(), ow["lon"].to_numpy()), k=1)
    ow["dist_foz_km"] = 2.0 * 6371.0 * np.arcsin(np.clip(dists_chord / 2.0, 0.0, 1.0))
    return ow.reset_index(drop=True), len(sinks)


def desenhar_continentes(ax: plt.Axes) -> None:
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


def gerar_figura(ow: pd.DataFrame, n_sinks: int) -> Path:
    aplicar_estilo()

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14.8, 5.1),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.20},
    )

    n_tot = len(ow)
    m_mar = ow["dist_foz_km"] > 200.0
    m_plat = (ow["dist_foz_km"] > 25.0) & (ow["dist_foz_km"] <= 200.0)
    m_foz = ow["dist_foz_km"] <= 25.0

    n_mar = int(m_mar.sum())
    n_plat = int(m_plat.sum())
    n_foz = int(m_foz.sum())
    n_15 = int((ow["dist_foz_km"] <= 15.0).sum())
    n_50 = int((ow["dist_foz_km"] <= 50.0).sum())
    med_km = float(ow["dist_foz_km"].median())

    # =========================================================================
    # Painel (a): Mapa-mundi de Ocean water (0,02-5,0 mm) por distancia a foz
    # =========================================================================
    eixo_mapa(ax1, (-180, 180, -60, 82))
    desenhar_continentes(ax1)

    ax1.scatter(
        ow.loc[m_mar, "lon"],
        ow.loc[m_mar, "lat"],
        s=4.0,
        color="#2b83ba",
        alpha=0.45,
        linewidths=0,
        zorder=4,
        label=f"Mar aberto / giros (> 200 km): {fmt_int_pt(n_mar)} ({fmt_dec_pt(n_mar / n_tot * 100)}%)",
    )
    ax1.scatter(
        ow.loc[m_plat, "lon"],
        ow.loc[m_plat, "lat"],
        s=5.5,
        color="#f59e0b",
        alpha=0.65,
        linewidths=0,
        zorder=6,
        label=f"Plataforma / offshore (25–200 km): {fmt_int_pt(n_plat)} ({fmt_dec_pt(n_plat / n_tot * 100)}%)",
    )
    ax1.scatter(
        ow.loc[m_foz, "lon"],
        ow.loc[m_foz, "lat"],
        s=9.5,
        color="#d7191c",
        alpha=0.85,
        linewidths=0,
        zorder=8,
        label=f"Perto de fozes (≤ 25 km): {fmt_int_pt(n_foz)} ({fmt_dec_pt(n_foz / n_tot * 100)}%)",
    )

    ax1.set_title(
        f"Localização global: Água oceânica 0,02–5,0 mm (n = {fmt_int_pt(n_tot)})",
        fontsize=10.8,
        pad=10,
    )
    leg = ax1.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=1,
        fontsize=8.3,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
        markerscale=1.8,
    )
    for h in leg.legend_handles:
        h.set_alpha(1.0)
    rotulo_painel(ax1, "(a)", dentro=True, x=0.02, y=0.96)

    # =========================================================================
    # Painel (b): Histograma log da distancia ate a foz + CDF acumulada
    # =========================================================================
    d = np.clip(ow["dist_foz_km"].to_numpy(), 0.25, None)
    bins = np.logspace(np.log10(0.25), np.log10(8000.0), 34)
    counts, edges = np.histogram(d, bins=bins)
    centers = np.sqrt(edges[:-1] * edges[1:])
    widths = np.diff(edges)

    bar_colors = []
    for c in centers:
        if c <= 25.0:
            bar_colors.append("#d7191c")
        elif c <= 200.0:
            bar_colors.append("#f59e0b")
        else:
            bar_colors.append("#2b83ba")

    ax2.bar(
        edges[:-1],
        counts,
        width=widths,
        align="edge",
        color=bar_colors,
        edgecolor="black",
        linewidth=0.45,
        alpha=0.88,
        zorder=3,
    )

    ax2.set_xscale("log")
    ax2.set_xlim(0.25, 8000.0)
    ax2.set_ylim(0, max(counts) * 1.25)
    ax2.set_xticks([1, 10, 25, 100, 200, 1000, 5000])
    ax2.set_xticklabels(["1", "10", "25", "100", "200", "1.000", "5.000"], fontsize=8.5)
    ax2.minorticks_off()
    ax2.set_xlabel("Distância até a foz IPBA mais próxima (km, escala log)", fontsize=10.3)
    ax2.set_ylabel("Pontos de coleta por faixa de distância", fontsize=10.3)
    ax2.set_title(f"Distância às {fmt_int_pt(n_sinks)} fozes globais (mediana = {fmt_dec_pt(med_km)} km)", fontsize=10.8, pad=10)

    # Linhas verticais de corte
    ax2.axvline(15.0, color="#7f0000", linestyle=":", linewidth=1.2, zorder=5)
    ax2.axvline(25.0, color="#d7191c", linestyle="--", linewidth=1.4, zorder=5)
    ax2.axvline(200.0, color="#1f4e78", linestyle="--", linewidth=1.4, zorder=5)
    ax2.axvline(med_km, color="black", linestyle="-.", linewidth=1.2, zorder=5)

    ymax = ax2.get_ylim()[1]
    ax2.text(13.0, ymax * 0.96, f"15 km\n({fmt_dec_pt(n_15 / n_tot * 100)}%)", ha="right", va="top", fontsize=7.6, color="#7f0000")
    ax2.text(28.0, ymax * 0.81, f"25 km\n({fmt_dec_pt(n_foz / n_tot * 100)}%)", ha="left", va="top", fontsize=7.6, color="#b30000")
    ax2.text(178.0, ymax * 0.96, f"200 km\n({fmt_dec_pt((n_foz + n_plat) / n_tot * 100)}%)\nacum.", ha="right", va="top", fontsize=7.6, color="#1f4e78")
    ax2.text(med_km * 1.12, ymax * 0.96, f"Mediana\n{med_km:.0f} km", ha="left", va="top", fontsize=7.6, color="black")

    # Eixo secundario: Curva acumulada (%)
    ax2_cdf = ax2.twinx()
    d_sorted = np.sort(d)
    cdf_pct = np.arange(1, len(d_sorted) + 1) / len(d_sorted) * 100.0
    ax2_cdf.plot(d_sorted, cdf_pct, color="#111111", linewidth=1.6, zorder=7, label="% acumulada")
    ax2_cdf.set_ylim(0, 108)
    ax2_cdf.set_ylabel("Percentual acumulado (%)", fontsize=10.0)
    ax2_cdf.tick_params(direction="out", length=4.0, width=0.8)

    # Destacar pontos da CDF em 25 km e 200 km
    ax2_cdf.scatter([25.0, 200.0], [n_foz / n_tot * 100.0, (n_foz + n_plat) / n_tot * 100.0], color=["#d7191c", "#1f4e78"], s=28, zorder=8)

    # Legenda compacta abaixo do painel (b)
    leg_b = [
        Patch(facecolor="#d7191c", edgecolor="black", linewidth=0.45, label=f"≤ 25 km ({n_foz / n_tot * 100:.1f}%)".replace(".", ",")),
        Patch(facecolor="#f59e0b", edgecolor="black", linewidth=0.45, label=f"25–200 km ({n_plat / n_tot * 100:.1f}%)".replace(".", ",")),
        Patch(facecolor="#2b83ba", edgecolor="black", linewidth=0.45, label=f"> 200 km ({n_mar / n_tot * 100:.1f}%)".replace(".", ",")),
        plt.Line2D([0], [0], color="#111111", lw=1.6, label="Curva acumulada (%)"),
    ]
    ax2.legend(
        handles=leg_b,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=2,
        fontsize=8.0,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
    )
    rotulo_painel(ax2, "(b)", dentro=False, x=-0.12, y=1.03)

    out_base = REPO_ROOT / "relatorios" / "figuras" / "noaa_ocean_water_distancia_fozes"
    salvar(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".png")


if __name__ == "__main__":
    ow_df, n_sinks_tot = carregar_ocean_water_com_distancias()
    png_path = gerar_figura(ow_df, n_sinks_tot)
    print(f"Figura salva em: {png_path}")
