"""
scripts/plotar_sobreposicao_e_justificativa_r.py
------------------------------------------------
Gera dois graficos independentes, diretos e autoexplicativos a partir dos CSVs
oficiais da branch `sinksRaioNoaa`:
  1. `relatorios/figuras/noaa_r_vs_quantidade_sobreposicoes.(png|pdf)`:
     Mostra o raio R (1 a 22 km) pela quantidade de sobreposicoes:
       (a) Quantidade de sinks/circulos com sobreposicao vs. exclusivos;
       (b) Quantidade media e maxima de circulos sobrepostos por ponto da NOAA.
  2. `relatorios/figuras/noaa_r_e_justificativa_melhor_escolha.(png|pdf)`:
     Mostra o valor de R (1 a 22 km) e a justificativa completa da escolha de R* = 5 km:
       (a) Zonas estatisticas de R² (%) (insuficiente -> pico otimo R*=5 -> diluicao);
       (b) Evolucao do expoente beta +- erro padrao (achatamento por sobreposicao);
       (c) Curva de Pareto Cobertura (N) vs. Qualidade (R²);
       (d) Quadro-sintese comparando R* = 5 km, R = 13 km e R = 22 km.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from geracao_graficos.estilo_erneson import aplicar_estilo, rotulo_painel, salvar


def fmt_int_pt(n: int) -> str:
    return f"{int(n):,}".replace(",", ".")


def fmt_dec_pt(v: float, dec: int = 1) -> str:
    return f"{v:.{dec}f}".replace(".", ",")


def gerar_grafico_r_vs_sobreposicoes(df_pre: pd.DataFrame) -> Path:
    """Gera o grafico 1: numero de R (1 a 22 km) pela quantidade de sobreposicoes."""
    aplicar_estilo()
    sub = df_pre[df_pre["r_km"].between(1, 22)].sort_values("r_km").copy()

    raios = sub["r_km"].to_numpy(dtype=int)
    n_comp = sub["n_sinks_compartilhados"].to_numpy(dtype=int)
    n_excl = sub["n_sinks_exclusivos"].to_numpy(dtype=int)
    n_ativ = sub["n_sinks_ativos"].to_numpy(dtype=int)
    pct_comp = sub["pct_sinks_compartilhados"].to_numpy(dtype=float)
    media_usos = sub["media_usos_por_ponto"].to_numpy(dtype=float)
    max_usos = sub["max_usos_ponto"].to_numpy(dtype=int)
    n_pts = sub["n_pontos_unicos_capturados"].to_numpy(dtype=int)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14.4, 5.4),
        gridspec_kw={"wspace": 0.24},
    )

    # --- Painel (a): Quantidade de sinks com sobreposicao vs exclusivos ---
    ax1.bar(
        raios,
        n_excl,
        width=0.72,
        color="#10b981",
        edgecolor="#065f46",
        linewidth=0.6,
        zorder=3,
        label="Círculos exclusivos (sem sobreposição)",
    )
    ax1.bar(
        raios,
        n_comp,
        bottom=n_excl,
        width=0.72,
        color="#ef4444",
        edgecolor="#7f1d1d",
        linewidth=0.6,
        alpha=0.88,
        zorder=3,
        label="Círculos com sobreposição (dividem pontos)",
    )
    ax1.axvline(5.0, color="#047857", linestyle="--", linewidth=1.2, alpha=0.85, zorder=4)

    anot_a = [
        (3, (3.2, 2350), "#7f1d1d"),
        (5, (5.5, 4200), "#065f46"),
        (13, (11.2, 8300), "#1e3a8a"),
        (22, (17.5, 11700), "#7f1d1d"),
    ]
    for r_k, (tx, ty), cor_b in anot_a:
        idx = int(np.where(raios == r_k)[0][0])
        ax1.annotate(
            f"R={r_k} km\n{fmt_int_pt(n_comp[idx])} sobrep. ({fmt_dec_pt(pct_comp[idx], 1)}%)",
            xy=(r_k, n_ativ[idx]),
            xytext=(tx, ty),
            fontsize=7.3,
            fontweight="bold" if r_k in (5, 22) else "normal",
            color=cor_b,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=cor_b, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=cor_b, lw=0.9),
            zorder=8,
        )

    ax1.set_xticks(range(2, 23, 2))
    ax1.set_xlim(0.2, 22.8)
    ax1.set_ylim(0, 13400)
    ax1.set_xlabel("Raio fixo R aplicado em cada foz (km)", fontsize=10.0)
    ax1.set_ylabel("Quantidade de círculos/sinks ativos", fontsize=10.0)
    ax1.set_title("Quantidade de círculos com sobreposição por raio R", fontsize=10.6, pad=10)
    ax1.legend(loc="upper left", fontsize=7.8, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)
    rotulo_painel(ax1, "(a)", dentro=False, x=-0.12, y=1.03)

    # --- Painel (b): Media e maximo de sobreposicoes por ponto NOAA ---
    bar_cols = [
        "#059669" if r == 5 else ("#2563eb" if r == 13 else ("#dc2626" if r == 22 else "#93c5fd"))
        for r in raios
    ]
    bar_eds = [
        "#064e3b" if r == 5 else ("#1e3a8a" if r == 13 else ("#7f1d1d" if r == 22 else "#1d4ed8"))
        for r in raios
    ]
    bars_b = ax2.bar(raios, media_usos, width=0.70, zorder=3)
    for b, c, ec in zip(bars_b, bar_cols, bar_eds):
        b.set_facecolor(c)
        b.set_edgecolor(ec)
        b.set_linewidth(0.85)

    for r_k, m_u in zip(raios, media_usos):
        if r_k in (1, 3, 5, 8, 10, 13, 16, 19, 22):
            peso = "bold" if r_k in (5, 13, 22) else "normal"
            ax2.text(
                r_k,
                m_u + 0.32,
                f"{fmt_dec_pt(m_u, 1)}x",
                ha="center",
                va="bottom",
                fontsize=7.3,
                fontweight=peso,
                color="#1e293b",
            )

    ax2_r = ax2.twinx()
    ax2_r.plot(
        raios,
        max_usos,
        color="#475569",
        marker="o",
        markersize=3.8,
        linestyle="--",
        linewidth=1.4,
        zorder=5,
        label="Máximo de sobreposições em 1 ponto",
    )
    ax2_r.set_ylabel("Máximo de círculos sobre o mesmo ponto", fontsize=9.8, color="#334155")
    ax2_r.tick_params(axis="y", labelcolor="#334155", direction="out", length=4.0, width=0.8)
    ax2_r.set_ylim(0, 105)

    ax2.annotate(
        f"Em R = 5 km: média {fmt_dec_pt(media_usos[4], 2)}x por ponto\n(máx. {max_usos[4]} círculos; {fmt_int_pt(n_pts[4])} pts)",
        xy=(5, media_usos[4] + 1.0),
        xytext=(7.6, 12.6),
        fontsize=7.3,
        fontweight="bold",
        color="#065f46",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#ecfdf5", edgecolor="#059669", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.0),
        zorder=9,
    )
    ax2.annotate(
        f"Em R = 22 km: {fmt_dec_pt(media_usos[21], 2)}x por ponto\n(máx. {max_usos[21]} círculos; {fmt_int_pt(n_pts[21])} pts)",
        xy=(22, media_usos[21] + 1.1),
        xytext=(15.5, 19.6),
        fontsize=7.3,
        fontweight="bold",
        color="#7f1d1d",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="#fef2f2", edgecolor="#dc2626", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#dc2626", lw=1.0),
        zorder=9,
    )

    ax2.set_xticks(range(2, 23, 2))
    ax2.set_xlim(0.2, 22.8)
    ax2.set_ylim(0, 22.5)
    ax2.set_xlabel("Raio fixo R aplicado em cada foz (km)", fontsize=10.0)
    ax2.set_ylabel("Média de círculos sobrepostos por ponto usado", fontsize=10.0)
    ax2.set_title("Multiplicação de uso dos pontos por sobreposição", fontsize=10.6, pad=10)

    leg_b = [
        Patch(facecolor="#93c5fd", edgecolor="#1d4ed8", lw=0.7, label="Média de círculos por ponto (x)"),
        ax2_r.get_lines()[0],
    ]
    ax2.legend(handles=leg_b, loc="upper left", fontsize=7.5, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)
    rotulo_painel(ax2, "(b)", dentro=False, x=-0.12, y=1.03)

    out_base = REPO_ROOT / "relatorios" / "figuras" / "noaa_r_vs_quantidade_sobreposicoes"
    salvar(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".png")


def gerar_grafico_r_e_justificativa(
    df_sens: pd.DataFrame,
    df_pre: pd.DataFrame,
) -> Path:
    """Gera o grafico 2: valor de R (1..22 km) e a justificativa completa da escolha."""
    aplicar_estilo()

    sub_c = df_sens[
        (df_sens["regime"] == "2_Bacias_Area_ge_100km2_Todas")
        & (df_sens["pop_col"] == "urban_population")
        & (df_sens["metrica"] == "concentracao")
        & (df_sens["r_km"].between(1, 22))
    ].sort_values("r_km").copy()

    sub_f = df_sens[
        (df_sens["regime"] == "2_Bacias_Area_ge_100km2_Todas")
        & (df_sens["pop_col"] == "urban_population")
        & (df_sens["metrica"] == "fluxo")
        & (df_sens["r_km"].between(1, 22))
    ].sort_values("r_km").copy()

    sub_pre = df_pre[df_pre["r_km"].between(1, 22)].sort_values("r_km").copy()

    raios = sub_c["r_km"].to_numpy(dtype=int)
    n_c = sub_c["n_bacias_estimadas"].to_numpy(dtype=int)
    r2_c = np.where(np.isfinite(sub_c["r2_pct"].to_numpy()), sub_c["r2_pct"].to_numpy(), 0.0)
    r2_f = np.where(np.isfinite(sub_f["r2_pct"].to_numpy()), sub_f["r2_pct"].to_numpy(), 0.0)
    beta_c = sub_c["beta"].to_numpy(dtype=float)
    se_c = sub_c["se"].to_numpy(dtype=float)
    beta_f = sub_f["beta"].to_numpy(dtype=float)
    se_f = sub_f["se"].to_numpy(dtype=float)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(14.2, 9.2),
        gridspec_kw={"wspace": 0.25, "hspace": 0.32},
    )
    ax_a, ax_b, ax_c, ax_d = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # --- Painel (a): As 3 zonas do raio R e o pico absoluto em R* = 5 km ---
    ax_a.axvspan(0.4, 3.5, color="#e5e7eb", alpha=0.55, zorder=1)
    ax_a.axvspan(4.5, 5.5, color="#d1fae5", alpha=0.65, zorder=1)
    ax_a.axvspan(5.5, 22.6, color="#eff6ff", alpha=0.55, zorder=1)

    bar_cols_a = [
        "#059669" if r == 5 else ("#2563eb" if r == 13 else ("#dc2626" if r == 22 else ("#d1d5db" if r <= 3 else "#93c5fd")))
        for r in raios
    ]
    bars_a = ax_a.bar(raios, r2_c, width=0.68, edgecolor="#334155", linewidth=0.6, zorder=3, label="Concentração Média (R² %)")
    for b, c in zip(bars_a, bar_cols_a):
        b.set_facecolor(c)

    ax_a.plot(
        raios[raios >= 3],
        r2_f[raios >= 3],
        color="#7c3aed",
        marker="D",
        markersize=4.0,
        linewidth=1.5,
        zorder=5,
        label="Fluxo C̄ × A (R² %)",
    )

    ax_a.text(2.0, 22.0, "Zona 1\nAmostra\ninsufic.\n(N ≤ 15)", ha="center", va="center", fontsize=7.3, color="#4b5563",
              bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#9ca3af", alpha=0.9))
    ax_a.annotate(
        "Zona 2 — ESCOLHIDO R* = 5 km\nMaior R² (33,3% conc. / 42,9% fluxo)\ncom amostra válida (N = 49 ≥ 30)",
        xy=(5.35, 33.5),
        xytext=(10.2, 37.8),
        fontsize=7.4,
        fontweight="bold",
        color="#065f46",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.26", facecolor="#ecfdf5", edgecolor="#059669", lw=1.2),
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.3),
        zorder=8,
    )
    ax_a.text(16.6, 25.0, "Zona 3 (R = 6..22 km)\nGanha cobertura (N → 314),\nmas cai o R² pela sobreposição",
              ha="center", va="center", fontsize=7.4, color="#1e3a8a",
              bbox=dict(boxstyle="round,pad=0.24", facecolor="white", edgecolor="#93c5fd", alpha=0.94))

    ax_a.set_xticks(range(2, 23, 2))
    ax_a.set_xlim(0.3, 22.7)
    ax_a.set_ylim(0, 56)
    ax_a.set_xlabel("Raio fixo R (km)", fontsize=9.8)
    ax_a.set_ylabel(r"Qualidade da Lei de Potência $R^2$ (%)", fontsize=9.8)
    ax_a.set_title("1. Pico estatístico em R* = 5 km (Concentração e Fluxo)", fontsize=10.1, pad=9)
    ax_a.legend(loc="upper right", fontsize=7.2, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)
    rotulo_painel(ax_a, "(a)", dentro=False, x=-0.12, y=1.03)

    # --- Painel (b): Expoente beta +- erro padrao ao longo de R = 3..22 km ---
    mask_sig = raios >= 4
    ax_b.axhline(1.0, color="#64748b", linestyle=":", linewidth=1.1, zorder=2, label=r"Escala linear ($\beta = 1{,}0$)")
    ax_b.axvline(5.0, color="#059669", linestyle="-.", linewidth=1.2, alpha=0.8, zorder=2)

    ax_b.errorbar(
        raios[mask_sig],
        beta_f[mask_sig],
        yerr=se_f[mask_sig],
        fmt="D-",
        color="#7c3aed",
        ecolor="#c4b5fd",
        markersize=4.4,
        capsize=2.5,
        linewidth=1.5,
        zorder=4,
        label=r"Fluxo ($\bar{C} \times A$): $\beta$ até +1,03 em 5 km",
    )
    ax_b.errorbar(
        raios[mask_sig],
        beta_c[mask_sig],
        yerr=se_c[mask_sig],
        fmt="o-",
        color="#0369a1",
        ecolor="#bae6fd",
        markersize=4.8,
        capsize=2.5,
        linewidth=1.7,
        zorder=5,
        label=r"Concentração ($\bar{C}$): $\beta$ de +0,70 (5 km) → +0,37 (22 km)",
    )
    ax_b.scatter([5], [beta_c[4]], color="#059669", s=75, zorder=7)
    ax_b.scatter([5], [beta_f[4]], color="#059669", marker="D", s=70, zorder=7)

    ax_b.annotate(
        "Em R* = 5 km o expoente\natinge o máximo real\n(sem achatamento)",
        xy=(5.2, 0.70),
        xytext=(12.2, 1.24),
        fontsize=7.4,
        fontweight="bold",
        color="#065f46",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.24", facecolor="#ecfdf5", edgecolor="#059669", alpha=0.95),
        arrowprops=dict(arrowstyle="->", color="#059669", lw=1.1),
        zorder=8,
    )

    ax_b.set_xticks(range(4, 23, 2))
    ax_b.set_xlim(3.2, 22.8)
    ax_b.set_ylim(0.15, 1.45)
    ax_b.set_xlabel("Raio fixo R (km)", fontsize=9.8)
    ax_b.set_ylabel(r"Expoente da Lei de Potência $\beta \pm \mathrm{EP}$", fontsize=9.8)
    ax_b.set_title("2. Raio grande achata β (de +0,70 para +0,37)", fontsize=10.1, pad=9)
    ax_b.legend(loc="lower left", fontsize=7.2, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)
    rotulo_painel(ax_b, "(b)", dentro=False, x=-0.12, y=1.03)

    # --- Painel (c): Trajetoria de Pareto N vs R² ---
    m_p = raios >= 3
    ax_c.plot(n_c[m_p], r2_c[m_p], color="#94a3b8", linewidth=1.4, zorder=2)
    for r_k, nv, r2v in zip(raios[m_p], n_c[m_p], r2_c[m_p]):
        cor_p = "#059669" if r_k == 5 else ("#2563eb" if r_k == 13 else ("#dc2626" if r_k == 22 else ("#9ca3af" if r_k <= 3 else "#475569")))
        sz = 95 if r_k in (5, 13, 22) else 34
        ax_c.scatter([nv], [r2v], color=cor_p, edgecolor="black", linewidth=0.7, s=sz, zorder=5)

    ax_c.axvline(30.0, color="#9ca3af", linestyle=":", linewidth=1.1, zorder=1)
    ax_c.text(34.0, 26.5, "Mín. estatístico\nN = 30", fontsize=7.0, color="#4b5563")

    anot_pareto = [
        (3, (74, 4.0), "R=3 (N=15, n.s.)", "#4b5563"),
        (5, (118, 33.8), "R* = 5 km (ÓTIMO)\nN = 49, R² = 33,3%", "#065f46"),
        (13, (175, 21.0), "R = 13 km (Interm.)\nN = 157, R² = 13,3%", "#1e3a8a"),
        (22, (255, 15.0), "R = 22 km (Mar territ.)\nN = 314, R² = 7,9%", "#7f1d1d"),
    ]
    for r_pt, (tx, ty), label_p, cor_p in anot_pareto:
        idx = int(np.where(raios == r_pt)[0][0])
        ax_c.annotate(
            label_p,
            xy=(n_c[idx], r2_c[idx]),
            xytext=(tx, ty),
            fontsize=7.3,
            fontweight="bold" if r_pt in (5, 13, 22) else "normal",
            color=cor_p,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=cor_p, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=cor_p, lw=1.0),
            zorder=8,
        )

    ax_c.set_xlim(0, 340)
    ax_c.set_ylim(0, 40)
    ax_c.set_xlabel("Cobertura: número de bacias urbanas estimadas (N)", fontsize=9.8)
    ax_c.set_ylabel(r"Qualidade do ajuste $R^2$ (%)", fontsize=9.8)
    ax_c.set_title("3. Curva de compromisso (Pareto): joelho ótimo em R* = 5 km", fontsize=10.1, pad=9)
    rotulo_painel(ax_c, "(c)", dentro=False, x=-0.12, y=1.03)

    # --- Painel (d): Sintese objetiva comparando R*=5 vs R=13 vs R=22 km ---
    ax_d.axis("off")
    ax_d.set_title("4. Justificativa comparada dos 3 raios-chave", fontsize=10.1, pad=9)

    col_labels = ["Critério", "R* = 5 km\n(ESCOLHIDO)", "R = 13 km\n(Intermediário)", "R = 22 km\n(Mar territorial)"]
    r5_pre = sub_pre[sub_pre["r_km"] == 5].iloc[0]
    r13_pre = sub_pre[sub_pre["r_km"] == 13].iloc[0]
    r22_pre = sub_pre[sub_pre["r_km"] == 22].iloc[0]

    cell_text = [
        ["Bacias urbanas (N)", "49", "157 (300 tot.)", "314 (624 tot.)"],
        ["R² Conc. média", "33,3% (pico)", "13,3%", "7,9% (14,1% tot.)"],
        ["R² Fluxo (C̄ × A)", "42,9% (pico)", "20,6%", "17,0% (25,3% tot.)"],
        ["Expoente β (Conc.)", "+0,70 ± 0,14", "+0,52 ± 0,11", "+0,37 ± 0,07"],
        ["Expoente β (Fluxo)", "+1,03 (≈ 1:1)", "+0,73", "+0,61"],
        [
            "Sobreposição/ponto",
            f"{fmt_dec_pt(float(r5_pre['media_usos_por_ponto']), 2)}x (baixa)",
            f"{fmt_dec_pt(float(r13_pre['media_usos_por_ponto']), 2)}x (média)",
            f"{fmt_dec_pt(float(r22_pre['media_usos_por_ponto']), 2)}x (alta)",
        ],
        ["Papel na tese", "Melhor sinal físico\ne lei de potência", "Compromisso N=300\nsem perder R²", "Máxima cobertura\nglobal (R²_bins 86%)"],
    ]

    tbl = ax_d.table(
        cellText=cell_text,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        bbox=[0.0, 0.04, 1.0, 0.90],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.0)

    for (r_i, c_i), cell in tbl.get_celld().items():
        cell.set_edgecolor("#cbd5e1")
        cell.set_linewidth(0.8)
        if r_i == 0:
            cell.set_text_props(weight="bold", color="#ffffff")
            if c_i == 1:
                cell.set_facecolor("#059669")
            elif c_i == 2:
                cell.set_facecolor("#2563eb")
            elif c_i == 3:
                cell.set_facecolor("#dc2626")
            else:
                cell.set_facecolor("#334155")
        else:
            if c_i == 0:
                cell.set_text_props(weight="bold", color="#1e293b")
                cell.set_facecolor("#f8fafc")
            elif c_i == 1:
                cell.set_facecolor("#ecfdf5")
                cell.set_text_props(weight="bold" if r_i in (2, 3, 7) else "normal", color="#065f46")
            else:
                cell.set_facecolor("#ffffff")
                cell.set_text_props(color="#1e293b")

    rotulo_painel(ax_d, "(d)", dentro=False, x=-0.04, y=1.03)

    out_base = REPO_ROOT / "relatorios" / "figuras" / "noaa_r_e_justificativa_melhor_escolha"
    salvar(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".png")


def gerar_grafico_r_vs_bacias_contempladas(df_sens: pd.DataFrame) -> Path:
    """
    Gera um grafico focado em explicar o porque da escolha de R* relacionando
    o valor de R (1 a 22 km) diretamente com a quantidade de bacias contempladas
    com pontos de coleta no mar (`n_bacias_com_pts_mar`, A >= 100 km²).
    """
    aplicar_estilo()

    sub_tot = df_sens[
        (df_sens["regime"] == "2_Bacias_Area_ge_100km2_Todas")
        & (df_sens["pop_col"] == "total_population")
        & (df_sens["metrica"] == "concentracao")
        & (df_sens["r_km"].between(1, 22))
    ].sort_values("r_km").copy()

    sub_urb = df_sens[
        (df_sens["regime"] == "2_Bacias_Area_ge_100km2_Todas")
        & (df_sens["pop_col"] == "urban_population")
        & (df_sens["metrica"] == "concentracao")
        & (df_sens["r_km"].between(1, 22))
    ].sort_values("r_km").copy()

    raios = sub_tot["r_km"].to_numpy(dtype=int)
    n_tot_pts = sub_tot["n_bacias_com_pts_mar"].to_numpy(dtype=int)
    n_urb_pts = sub_urb["n_bacias_com_pts_mar"].to_numpy(dtype=int)
    n_tot_est = sub_tot["n_bacias_estimadas"].to_numpy(dtype=int)
    n_urb_est = sub_urb["n_bacias_estimadas"].to_numpy(dtype=int)
    r2_tot = np.where(np.isfinite(sub_tot["r2_pct"].to_numpy()), sub_tot["r2_pct"].to_numpy(), np.nan)
    r2_urb = np.where(np.isfinite(sub_urb["r2_pct"].to_numpy()), sub_urb["r2_pct"].to_numpy(), np.nan)

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(14.8, 5.7),
        gridspec_kw={"wspace": 0.23},
    )

    # --- Painel (a): R (1..22 km) x Bacias contempladas com pontos de coleta ---
    ax1.axvspan(0.4, 3.5, color="#f1f5f9", alpha=0.85, zorder=1)
    ax1.axvspan(4.5, 5.5, color="#d1fae5", alpha=0.55, zorder=1)

    cols_tot = [
        "#6ee7b7" if r == 5 else ("#93c5fd" if r == 13 else ("#fca5a5" if r == 22 else ("#e2e8f0" if r <= 3 else "#bfdbfe")))
        for r in raios
    ]
    eds_tot = [
        "#047857" if r == 5 else ("#1d4ed8" if r == 13 else ("#b91c1c" if r == 22 else ("#94a3b8" if r <= 3 else "#3b82f6")))
        for r in raios
    ]
    cols_urb = [
        "#059669" if r == 5 else ("#1d4ed8" if r == 13 else ("#dc2626" if r == 22 else ("#94a3b8" if r <= 3 else "#1e40af")))
        for r in raios
    ]

    b_tot = ax1.bar(raios, n_tot_pts, width=0.76, zorder=2)
    for b, c, ec in zip(b_tot, cols_tot, eds_tot):
        b.set_facecolor(c)
        b.set_edgecolor(ec)
        b.set_linewidth(0.8)

    b_urb = ax1.bar(raios, n_urb_pts, width=0.44, zorder=4)
    for b, c in zip(b_urb, cols_urb):
        b.set_facecolor(c)
        b.set_edgecolor("#0f172a")
        b.set_linewidth(0.55)

    ax1.axhline(30.0, color="#64748b", linestyle=":", linewidth=1.1, zorder=5)
    ax1.text(
        0.55,
        44.0,
        "Mín. estatístico (N = 30)",
        ha="left",
        va="bottom",
        fontsize=6.9,
        color="#334155",
        bbox=dict(boxstyle="round,pad=0.14", facecolor="white", edgecolor="none", alpha=0.92),
        zorder=6,
    )
    ax1.axhline(300.0, color="#0369a1", linestyle="--", linewidth=1.1, alpha=0.85, zorder=5)
    ax1.text(0.8, 309.0, "Meta de ~300 bacias citada nos áudios", ha="left", va="bottom", fontsize=7.2, color="#0369a1")

    anot_a = [
        (3, (3.1, 165), f"R=1–3 km (insuf.)\n22 tot / 16 urb.", "#475569"),
        (5, (6.0, 245), f"R* = 5 km (ESCOLHIDO)\n{n_tot_pts[4]} IPBA / {n_urb_pts[4]} urb.\n(R² = 33,3%)", "#065f46"),
        (13, (11.2, 470), f"R = 13 km\n{n_tot_pts[12]} IPBA / {n_urb_pts[12]} urb.", "#1e3a8a"),
        (22, (17.5, 675), f"R = 22 km (Mar territ.)\n{n_tot_pts[21]} IPBA / {n_urb_pts[21]} urb.", "#7f1d1d"),
    ]
    for r_pt, (tx, ty), texto_a, cor_a in anot_a:
        idx = int(np.where(raios == r_pt)[0][0])
        ax1.annotate(
            texto_a,
            xy=(r_pt, n_tot_pts[idx]),
            xytext=(tx, ty),
            fontsize=7.3,
            fontweight="bold" if r_pt in (5, 13, 22) else "normal",
            color=cor_a,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=cor_a, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=cor_a, lw=0.95),
            zorder=8,
        )

    ax1.set_xticks(range(2, 23, 2))
    ax1.set_xlim(0.2, 22.8)
    ax1.set_ylim(0, 790)
    ax1.set_xlabel("Raio fixo R em todas as fozes (km)", fontsize=10.0)
    ax1.set_ylabel("Bacias contempladas com pontos de coleta (N)", fontsize=10.0)
    ax1.set_title("Cobertura: R (1–22 km) × Bacias contempladas", fontsize=10.6, pad=10)
    leg_a = [
        Patch(facecolor="#bfdbfe", edgecolor="#3b82f6", lw=0.8, label="Todas as bacias IPBA com pontos no mar (≥ 100 km²)"),
        Patch(facecolor="#1e40af", edgecolor="#0f172a", lw=0.6, label="Bacias c/ Mancha Urbana CCA contempladas"),
        Patch(facecolor="#059669", edgecolor="#047857", lw=0.8, label="R* = 5 km (ESCOLHIDO — melhor lei de potência)"),
    ]
    ax1.legend(handles=leg_a, loc="upper left", fontsize=7.2, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)
    rotulo_painel(ax1, "(a)", dentro=False, x=-0.12, y=1.03)

    # --- Painel (b): Por que R*=5 km? Relacao Cobertura (bacias contempladas) x R² ---
    m3 = raios >= 3
    ax2.plot(n_urb_pts[m3], r2_urb[m3], color="#059669", marker="o", markersize=4.5, linewidth=1.8, zorder=4,
             label="Mancha Urbana CCA (R = 3..22 km)")
    ax2.plot(n_tot_pts[m3], r2_tot[m3], color="#2563eb", marker="s", markersize=4.0, linewidth=1.5, linestyle="--", zorder=3,
             label="População Total IPBA (R = 3..22 km)")

    ax2.axvline(30.0, color="#64748b", linestyle=":", linewidth=1.1, zorder=2)
    ax2.scatter([n_urb_pts[4]], [r2_urb[4]], color="#059669", edgecolor="#064e3b", linewidth=1.4, s=110, zorder=7)
    ax2.scatter([n_tot_pts[3]], [r2_tot[3]], color="#1d4ed8", edgecolor="#1e3a8a", linewidth=1.2, s=85, zorder=7)
    ax2.scatter([n_urb_pts[12], n_tot_pts[12]], [r2_urb[12], r2_tot[12]], color="#2563eb", edgecolor="black", s=65, zorder=6)
    ax2.scatter([n_urb_pts[21], n_tot_pts[21]], [r2_urb[21], r2_tot[21]], color="#dc2626", edgecolor="black", s=70, zorder=6)

    anot_b = [
        (3, n_urb_pts[2], r2_urb[2], (95, 5.2), "R=3 (16 urb., n.s.)", "#475569"),
        (5, n_urb_pts[4], r2_urb[4], (210, 35.5), f"ESCOLHIDO R* = 5 km\n{n_urb_pts[4]} bacias urb. ({n_urb_est[4]} pos.)\nPico R² = 33,3% (β = +0,70)", "#065f46"),
        (4, n_tot_pts[3], r2_tot[3], (185, 27.2), f"R=4 km (Total: {n_tot_pts[3]} bacias)\nPico Pop. Total: R² = 26,9%", "#1e3a8a"),
        (13, n_tot_pts[12], r2_tot[12], (380, 21.8), f"R=13 km ({n_tot_pts[12]} IPBA / {n_tot_est[12]} pos.)\nR² = 15,1% tot. / 13,3% urb.", "#1e3a8a"),
        (22, n_urb_pts[21], r2_urb[21], (340, 4.3), f"R=22 km Urb. ({n_urb_pts[21]} bacias): R²=7,9%", "#7f1d1d"),
        (22, n_tot_pts[21], r2_tot[21], (575, 19.5), f"R=22 km Total\n({n_tot_pts[21]} IPBA / {n_tot_est[21]} pos.)\nR² = 14,1%", "#7f1d1d"),
    ]
    for _, x_p, y_p, (tx, ty), txt_b, cor_b in anot_b:
        ax2.annotate(
            txt_b,
            xy=(x_p, y_p),
            xytext=(tx, ty),
            fontsize=7.0,
            fontweight="bold" if "ESCOLHIDO" in txt_b else "normal",
            color=cor_b,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor=cor_b, alpha=0.95),
            arrowprops=dict(arrowstyle="->", color=cor_b, lw=0.9),
            zorder=8,
        )

    ax2.text(
        485,
        35.5,
        "Por que R* = 5 km?\n"
        "• R ≤ 3 km: poucas bacias (< 30).\n"
        "• R* = 5 km: 1º ponto c/ amostra\n"
        "  sólida antes da mistura costeira.\n"
        "• R > 5 km: +bacias, −qualidade.",
        fontsize=7.3,
        color="#1e293b",
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.30", facecolor="#f8fafc", edgecolor="#94a3b8", alpha=0.96),
        zorder=9,
    )

    ax2.set_xlim(0, 745)
    ax2.set_ylim(0, 43.0)
    ax2.set_xlabel("Bacias contempladas com pontos de coleta no mar (N)", fontsize=10.0)
    ax2.set_ylabel(r"Qualidade da Lei de Potência $R^2$ (%)", fontsize=10.0)
    ax2.set_title("Por que R* = 5 km? Relação Cobertura × Qualidade", fontsize=10.6, pad=10)
    ax2.legend(loc="upper left", fontsize=7.3, frameon=True, facecolor="white", edgecolor="#cccccc", framealpha=0.95)
    rotulo_painel(ax2, "(b)", dentro=False, x=-0.12, y=1.03)

    out_base = REPO_ROOT / "relatorios" / "figuras" / "noaa_r_vs_bacias_contempladas_e_escolha"
    salvar(fig, out_base)
    plt.close(fig)
    return out_base.with_suffix(".png")


if __name__ == "__main__":
    pre_csv = REPO_ROOT / "relatorios" / "pre_ipba_estatisticas_raios_sinks.csv"
    sens_csv = REPO_ROOT / "relatorios" / "lei_escala_sensibilidade_raio_fixo.csv"

    df_pre = pd.read_csv(pre_csv)
    df_sens = pd.read_csv(sens_csv)

    p1 = gerar_grafico_r_vs_sobreposicoes(df_pre)
    p2 = gerar_grafico_r_e_justificativa(df_sens, df_pre)
    p3 = gerar_grafico_r_vs_bacias_contempladas(df_sens)
    print(f"Graficos gerados com sucesso:\n  1. {p1}\n  2. {p2}\n  3. {p3}")
