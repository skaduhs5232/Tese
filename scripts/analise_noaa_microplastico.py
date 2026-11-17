"""
scripts/analise_noaa_microplastico.py
-------------------------------------
Analise exploratoria completa de TODA a base NOAA NCEI Marine Microplastics
(29.076 registros, sem filtros aplicados):
  - Tipos de lixo / compartimentos (`Medium`) e metodos (`Sampling_Method`)
  - Quantidade de pontos de coleta (registros e coordenadas unicas)
  - Unidades de medida (`Unit`)
  - Corte pela definicao fisica e operacional de microplastico (`Mesh_size__mm_`)

Gera figura em PNG (300 DPI) e PDF vetorial em `relatorios/figuras/`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from geracao_graficos.estilo_erneson import aplicar_estilo, rotulo_painel, salvar


def carregar_noaa_completo() -> pd.DataFrame:
    noaa_path = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
    with open(noaa_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    df = pd.DataFrame([ft["properties"] for ft in gj["features"]])
    df["lat"] = pd.to_numeric(df["Latitude__degree_"], errors="coerce")
    df["lon"] = pd.to_numeric(df["Longitude_degree_"], errors="coerce")
    df["mesh_mm"] = pd.to_numeric(df["Mesh_size__mm_"], errors="coerce")
    df["mp_val"] = pd.to_numeric(df["Microplastics_measurement"], errors="coerce")
    df["nurdle_val"] = pd.to_numeric(df["Standardized_Nurdle__Amount"], errors="coerce")
    df["valor"] = np.where(df["Medium"] == "Nurdle Patrol", df["nurdle_val"], df["mp_val"])

    def classificar_corte(row: pd.Series) -> str:
        m = row["mesh_mm"]
        if pd.isna(m):
            return "sem_malha"
        if m < 0.001:
            return "submicro"
        if m < 0.02:
            return "filtro_fino"
        if m <= 5.0:
            return "malha_padrao"
        return "acima_5mm"

    df["corte_mp"] = df.apply(classificar_corte, axis=1)
    return df


def fmt_int_pt(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def fmt_pct_pt(val: float) -> str:
    return f"{val:.1f}%".replace(".", ",")


def gerar_figura(df: pd.DataFrame) -> Path:
    aplicar_estilo()

    fig = plt.figure(figsize=(14.4, 5.9))
    outer = GridSpec(1, 2, figure=fig, width_ratios=[1.02, 1.24], wspace=0.24)

    ax1 = fig.add_subplot(outer[0, 0])
    inner_b = GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer[0, 1], width_ratios=[4.3, 1.15], wspace=0.05
    )
    ax2 = fig.add_subplot(inner_b[0, 0])
    ax2_nan = fig.add_subplot(inner_b[0, 1], sharey=ax2)

    # =========================================================================
    # Painel (a): Toda a base NOAA por compartimento, unidade e corte MP
    # =========================================================================
    grupos_ordem = [
        ("Ocean water", "pieces/m3", "Água oceânica\n(Ocean water)\n[pieces/m³]"),
        ("Nurdle Patrol", "pieces/10 mins", "Pellets industriais\n(Nurdle Patrol)\n[pieces/10 mins]"),
        ("Beach", "pieces/m3", "Areia de praia\n(Beach)\n[pieces/m³]"),
        ("Ocean sediment", "pieces kg-1 d.w.", "Sedimento marinho\n(Ocean sediment)\n[pieces kg⁻¹ d.w.]"),
    ]

    categorias_corte = [
        ("malha_padrao", "Microplástico — malha padrão (0,02–5,0 mm)", "#1b7837"),
        ("filtro_fino", "Microplástico — filtro fino (0,001–<0,02 mm)", "#2b83ba"),
        ("submicro", "Submicrométrico / nano (< 0,001 mm)", "#e66101"),
        ("sem_malha", "Sem malha / coleta visual (NaN)", "#7f7f7f"),
    ]

    y_pos = np.arange(len(grupos_ordem))[::-1]
    bar_h = 0.56

    for idx, (med, unit, _) in enumerate(grupos_ordem):
        sub = df[(df["Medium"] == med) & (df["Unit"] == unit)]
        y = y_pos[idx]
        left = 0.0
        for cat_key, _, cor in categorias_corte:
            cnt = int((sub["corte_mp"] == cat_key).sum())
            if cnt > 0:
                ax1.barh(
                    y,
                    cnt,
                    left=left,
                    height=bar_h,
                    color=cor,
                    edgecolor="black",
                    linewidth=0.6,
                    zorder=3,
                )
                left += cnt

        total = len(sub)
        n_coords = sub[["lat", "lon"]].drop_duplicates().shape[0]
        n_mp_def = int(sub["corte_mp"].isin(["malha_padrao", "filtro_fino"]).sum())
        pct_mp = (n_mp_def / total * 100.0) if total > 0 else 0.0

        if med == "Nurdle Patrol":
            texto_resumo = (
                f"{fmt_int_pt(total)} coletas ({fmt_int_pt(n_coords)} locais)\n"
                f"0% c/ malha (coleta visual)"
            )
        else:
            texto_resumo = (
                f"{fmt_int_pt(total)} coletas ({fmt_int_pt(n_coords)} locais)\n"
                f"{fmt_pct_pt(pct_mp)} na faixa 1 µm–5 mm"
            )

        ax1.text(
            total + 320,
            y,
            texto_resumo,
            va="center",
            ha="left",
            fontsize=8.4,
            color="#111111",
        )

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([g[2] for g in grupos_ordem], fontsize=9.1)
    ax1.set_xlabel("Quantidade de pontos de coleta (registros NOAA)", fontsize=10.3)
    ax1.set_xlim(0, 28500)
    ax1.set_ylim(-0.6, 3.6)
    ax1.set_xticks([0, 5000, 10000, 15000, 20000, 25000])
    ax1.set_xticklabels(["0", "5.000", "10.000", "15.000", "20.000", "25.000"])
    ax1.set_title(f"Tipos de lixo, unidades e corte (n = {fmt_int_pt(len(df))})", fontsize=10.8, pad=10)

    leg_handles_a = [
        Patch(facecolor=cor, edgecolor="black", linewidth=0.6, label=lbl)
        for _, lbl, cor in categorias_corte
    ]
    ax1.legend(
        handles=leg_handles_a,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        fontsize=7.9,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
    )
    rotulo_painel(ax1, "(a)", dentro=False, x=-0.16, y=1.03)

    # =========================================================================
    # Painel (b) parte esquerda: Espectro continuo de malha ate 5,0 mm (sem Meso/Macro)
    # =========================================================================
    bin_edges = np.array([0.0003, 0.001, 0.005, 0.02, 0.08, 0.25, 0.45, 1.5, 5.0])

    ax2.axvspan(0.00025, 0.001, color="#fde0c2", alpha=0.45, zorder=0)
    ax2.axvspan(0.001, 0.02, color="#d1e5f0", alpha=0.50, zorder=0)
    ax2.axvspan(0.02, 5.0, color="#d9f0d3", alpha=0.50, zorder=0)

    ax2.axvline(0.001, color="#b35806", linestyle="--", linewidth=1.4, zorder=4)
    ax2.axvline(0.02, color="#2166ac", linestyle=":", linewidth=1.4, zorder=4)
    ax2.axvline(5.0, color="#1b7837", linestyle="--", linewidth=1.6, zorder=4)

    media_com_malha = [
        ("Ocean water", "Água oceânica (pieces/m³)", "#2b83ba", -0.24),
        ("Beach", "Areia de praia (pieces/m³)", "#d7191c", 0.0),
        ("Ocean sediment", "Sedimento marinho (pieces kg⁻¹ d.w.)", "#7b3294", +0.24),
    ]

    log_edges = np.log10(bin_edges)
    log_widths = np.diff(log_edges)
    w_bar = log_widths * 0.24

    for med, lbl, cor, offset_frac in media_com_malha:
        vals = df.loc[(df["Medium"] == med) & (df["mesh_mm"].notna()), "mesh_mm"].to_numpy()
        counts, _ = np.histogram(vals, bins=bin_edges)
        log_centers = 0.5 * (log_edges[:-1] + log_edges[1:]) + offset_frac * log_widths

        mask_nz = counts > 0
        if np.any(mask_nz):
            x_left = 10 ** (log_centers[mask_nz] - w_bar[mask_nz] / 2)
            x_right = 10 ** (log_centers[mask_nz] + w_bar[mask_nz] / 2)
            ax2.bar(
                x_left,
                counts[mask_nz],
                width=(x_right - x_left),
                align="edge",
                color=cor,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.92,
                zorder=3,
            )

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlim(0.00025, 5.0)
    ax2.set_ylim(0.8, 220000)
    ax2.set_xticks([0.001, 0.02, 0.1, 0.333, 1.0, 5.0])
    ax2.set_xticklabels(["0,001\n(1 µm)", "0,02\n(20 µm)", "0,1", "0,333", "1,0", "5,0\n(5 mm)"], fontsize=8.4)
    ax2.minorticks_off()
    ax2.set_xlabel("Abertura de malha / filtro (mm, escala log)", fontsize=10.3)
    ax2.set_ylabel("Pontos de coleta (escala log)", fontsize=10.3)

    n_sub = int((df["corte_mp"] == "submicro").sum())
    n_fino = int((df["corte_mp"] == "filtro_fino").sum())
    n_pad = int((df["corte_mp"] == "malha_padrao").sum())
    n_nan = int((df["corte_mp"] == "sem_malha").sum())

    ax2.text(0.00050, 145000, f"Nano/\nSubmicro\nn = {fmt_int_pt(n_sub)}", ha="center", va="top", fontsize=7.8, color="#7f3b08")
    ax2.text(0.0045, 145000, f"Filtro fino\n(1–20 µm)\nn = {fmt_int_pt(n_fino)}", ha="center", va="top", fontsize=7.8, color="#053061")
    ax2.text(0.32, 145000, f"Microplástico padrão (20 µm–5 mm)\nn = {fmt_int_pt(n_pad)}  (pico em 0,33–0,335 mm)", ha="center", va="top", fontsize=8.0, color="#00441b")

    rotulo_painel(ax2, "(b)", dentro=False, x=-0.13, y=1.03)

    # =========================================================================
    # Painel (b) parte direita: Bloco discreto "Sem malha (NaN)" (sem filtrar!)
    # =========================================================================
    ax2_nan.axvspan(-0.55, 0.55, color="#f2f2f2", alpha=0.85, zorder=0)

    nan_series = [
        ("Nurdle Patrol", "#7f7f7f", -0.25),
        ("Ocean sediment", "#7b3294", 0.0),
        ("Beach", "#d7191c", +0.25),
    ]
    for med, cor, xpos in nan_series:
        cnt_nan = int(((df["Medium"] == med) & (df["corte_mp"] == "sem_malha")).sum())
        if cnt_nan > 0:
            ax2_nan.bar(
                xpos,
                cnt_nan,
                width=0.22,
                color=cor,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.92,
                zorder=3,
            )
            ax2_nan.text(
                xpos,
                cnt_nan * 1.18,
                fmt_int_pt(cnt_nan),
                ha="center",
                va="bottom",
                fontsize=7.3,
                color="#111111",
            )

    ax2_nan.set_xlim(-0.55, 0.55)
    ax2_nan.set_xticks([0.0])
    ax2_nan.set_xticklabels(["Sem malha\n(NaN)"], fontsize=8.4)
    ax2_nan.tick_params(axis="y", left=False, labelleft=False)
    ax2_nan.minorticks_off()
    ax2_nan.text(
        0.0,
        145000,
        f"Sem malha\n(coleta visual / NaN)\nn = {fmt_int_pt(n_nan)}",
        ha="center",
        va="top",
        fontsize=7.5,
        color="#333333",
    )

    # Titulo centralizado sobre os dois sub-eixos do Painel (b)
    fig.canvas.draw()
    pos_l = ax2.get_position()
    pos_r = ax2_nan.get_position()
    cx_b = 0.5 * (pos_l.x0 + pos_r.x1)
    fig.text(
        cx_b,
        pos_l.y1 + 0.025,
        f"Base completa por corte de microplástico (n = {fmt_int_pt(len(df))})",
        ha="center",
        va="bottom",
        fontsize=10.8,
    )

    # Legenda conjunta do Painel (b)
    leg_handles_b = [
        Patch(facecolor="#2b83ba", edgecolor="black", linewidth=0.5, label="Água oceânica (pieces/m³)"),
        Patch(facecolor="#7f7f7f", edgecolor="black", linewidth=0.5, label="Pellets industriais / Nurdle Patrol (pieces/10 mins)"),
        Patch(facecolor="#d7191c", edgecolor="black", linewidth=0.5, label="Areia de praia (pieces/m³)"),
        Patch(facecolor="#7b3294", edgecolor="black", linewidth=0.5, label="Sedimento marinho (pieces kg⁻¹ d.w.)"),
    ]
    ax2.legend(
        handles=leg_handles_b,
        loc="upper center",
        bbox_to_anchor=(0.64, -0.16),
        ncol=2,
        fontsize=7.9,
        frameon=True,
        facecolor="white",
        edgecolor="#cccccc",
        framealpha=0.95,
    )

    out_base = REPO_ROOT / "relatorios" / "figuras" / "noaa_analise_tipos_unidades_microplastico"
    salvar(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".png")


if __name__ == "__main__":
    df_noaa = carregar_noaa_completo()
    png_path = gerar_figura(df_noaa)
    print(f"Figura salva em: {png_path}")
