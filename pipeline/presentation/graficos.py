"""Componentes gráficos padronizados (lei de potência, mosaico, varredura e calibração Zipf)."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

from geracao_graficos.estilo_erneson import (
    anotar_expoente,
    aplicar_estilo,
    colorbar_direita,
    eixos_log10,
    referencia_inclinacao_1,
    reta_ols,
    rotulo_canto,
    rotulo_painel,
    salvar,
    scatter_vazado,
)

from ..config import CORES, EIXOS_X, FIGURAS_DIR, UNIDADES
from ..domain.scaling import ajustar_lei_potencia, filtrar_pop, medias_bins


def plotar_lei_potencia(
    ax: plt.Axes,
    df: pd.DataFrame,
    x_col: str,
    titulo: str,
    xlabel: str,
    ylabel: str,
    cor: str = "#0284c7",
) -> None:
    """Nuvem log-log, reta OLS, médias por faixa (pontos pretos) e caixa de estatísticas."""
    res = ajustar_lei_potencia(df, x_col)
    ax.set(title=titulo, xlabel=xlabel, ylabel=ylabel)
    if res["N"] < 4:
        ax.text(0.5, 0.5, f"N = {res['N']} (insuficiente)", ha="center", va="center", transform=ax.transAxes)
        return

    sub = df[(df[x_col] > 0) & (df["mp_mean"] > 0)]
    x, y = np.log10(sub[x_col].to_numpy()), np.log10(sub["mp_mean"].to_numpy())
    ax.scatter(x, y, s=22, facecolors="none", edgecolors=cor, alpha=0.7, label=f"Bacias (N = {res['N']})")
    xl = np.array([x.min(), x.max()])
    ax.plot(xl, res["intercept"] + res["beta"] * xl, color=cor, lw=2, label="Ajuste na nuvem")

    if res["N"] >= 20:
        b = medias_bins(x, y)
        ax.plot(b["xm"], b["ym"], "ko", ms=5, label="Média por faixa")

    bins_txt = (
        f"\nR²_bins = {res['r2_bins_pct']:.1f}% | β_bins (pond) = {res['beta_bins_w']:+.2f}"
        if np.isfinite(res["r2_bins_pct"])
        else ""
    )
    texto = (
        f"β = {res['beta']:+.2f} ± {res['se']:.2f}\n"
        f"R² = {res['r2_pct']:.1f}% (p = {res['p_value']:.1e})\n"
        f"ρ Spearman = {res['rho']:+.2f}{bins_txt}"
    )
    ax.text(
        0.03, 0.97, texto.replace(".", ","), transform=ax.transAxes, ha="left", va="top", fontsize=7.5,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#cbd5e1"),
    )
    span = max(y.max() - y.min(), 2.0)
    ax.set_ylim(y.min() - 0.15 * span, y.max() + 0.55 * span)
    ax.legend(loc="lower right", fontsize=7)


def plotar_mosaico(
    pareados: dict[str, pd.DataFrame],
    titulo: str,
    pop_min: float = 0,
    arquivo: str | None = None,
) -> None:
    """Linhas = conjuntos de pontos; colunas = População e Área da bacia."""
    aplicar_estilo()
    fig, axes = plt.subplots(len(pareados), 2, figsize=(12, 4.3 * len(pareados)), squeeze=False)
    for i, (nome, df) in enumerate(pareados.items()):
        for j, (x_col, nome_x) in enumerate(EIXOS_X.items()):
            ax = axes[i, j]
            plotar_lei_potencia(
                ax, filtrar_pop(df, pop_min), x_col, nome,
                f"log10({nome_x})", f"log10(Média {UNIDADES.get(nome, 'pieces/m³')})", CORES[i % len(CORES)],
            )
            rotulo_painel(ax, f"({'abcdefgh'[2 * i + j]})", x=-0.12, y=1.04)

    filtro = f"População ≥ {pop_min:,} hab.".replace(",", ".") if pop_min else "sem filtro de população"
    fig.suptitle(f"{titulo} — {filtro}")
    fig.tight_layout()
    if arquivo:
        FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(FIGURAS_DIR / f"{arquivo}.png", dpi=200, bbox_inches="tight")
    plt.show()


def plotar_varredura(tab: pd.DataFrame, min_n: int = 20) -> None:
    """Para cada conjunto: R² (%) vs raio R (População e Área) e N de bacias vs R."""
    aplicar_estilo()
    nomes = tab["Conjunto"].unique()
    fig, axes = plt.subplots(len(nomes), 2, figsize=(12, 3.2 * len(nomes)), sharex=True, squeeze=False)
    for i, nome in enumerate(nomes):
        for x_col, nome_x in EIXOS_X.items():
            s = tab[(tab["Conjunto"] == nome) & (tab["X"] == x_col)]
            axes[i, 0].plot(s["R_km"], s["r2_pct"].where(s["N"] >= min_n), "o-", label=nome_x)
        axes[i, 1].bar(s["R_km"], s["N"], color=CORES[i % len(CORES)])
        axes[i, 0].set(title=f"{nome} — R² na nuvem (N ≥ {min_n})", ylabel="R² (%)")
        axes[i, 1].set(title=f"{nome} — bacias com valor no sink", ylabel="N")
        axes[i, 0].legend(fontsize=8)
        for ax in axes[i]:
            for r in (5, 22):
                ax.axvline(r, color="gray", ls=":", lw=1)

    for ax in axes[-1]:
        ax.set_xlabel("Raio R no sink (km)")
        ax.set_xticks(range(1, 23))
    fig.tight_layout()
    plt.show()


def _heatmap_zipf(fig, ax, matriz, l_values, d_values, vmin, vmax, cbar_label, painel, ylabel=None):
    im = ax.imshow(matriz, cmap="plasma", origin="lower", norm=Normalize(vmin=vmin, vmax=vmax), aspect="auto", interpolation="nearest")
    colorbar_direita(fig, ax, im, cbar_label)
    ax.set_xticks(np.arange(len(l_values)))
    ax.set_xticklabels([f"{x:g}" for x in l_values])
    ax.set_yticks(np.arange(len(d_values)))
    ax.set_yticklabels([f"{int(y)}" for y in d_values])
    ax.set_xlabel(r"Cluster distance cutoff $\ell$ (km)")
    if ylabel:
        ax.set_ylabel(ylabel)
    rotulo_painel(ax, painel)


def plot_zipf_calibration(
    matrix_alpha: np.ndarray,
    matrix_r2: np.ndarray,
    d_values: Sequence[float],
    l_values: Sequence[float],
    opt_d: float,
    opt_l: float,
    clusters_df: pd.DataFrame,
    output_basepath: str | Path,
    region_name: str = "North & Central America",
) -> tuple[Path, Path]:
    """Painel de calibração Zipf (D* × L): heatmaps de alpha e R² + curva rank-size ótima."""
    aplicar_estilo()
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(11.5, 3.4))

    _heatmap_zipf(fig, ax_a, matrix_alpha, l_values, d_values, 0.6, 1.4, r"Zipf exponent $\alpha$", "a", r"Density cutoff $D^*$ (inhab./km$^2$)")
    ax_a.contour(matrix_alpha, levels=[1.0], colors="black", linewidths=1.2, origin="lower")

    _heatmap_zipf(fig, ax_b, matrix_r2, l_values, d_values, 0.7, 1.0, r"Coefficient of determination $R^2$", "b")
    opt_j = list(l_values).index(opt_l) if opt_l in l_values else 0
    opt_i = list(d_values).index(opt_d) if opt_d in d_values else 0
    ax_b.scatter(opt_j, opt_i, marker="*", s=140, edgecolors="white", facecolors="black", lw=1.2, zorder=50)

    top_pops = clusters_df["population"].values[:100]
    log_r = np.log10(np.arange(1, len(top_pops) + 1, dtype=float))
    log_p = np.log10(top_pops)
    scatter_vazado(ax_c, log_r, log_p, c="C0", marker="o", s=25, label="Urban clusters")
    ols = reta_ols(ax_c, log_r, log_p, color="black", lw=1.2)
    referencia_inclinacao_1(ax_c, log_r.min(), log_r.max(), log_p.max())
    eixos_log10(ax_c, xlabel=r"$\log_{10} r$ (rank)", ylabel=r"$\log_{10} P$ (population)")
    rotulo_painel(ax_c, "c")
    rotulo_canto(ax_c, region_name, canto="upper right", bbox=False)
    anotar_expoente(
        ax_c,
        f"$\\alpha = {abs(ols['slope']):.2f} \\pm {ols['stderr']:.2f}$\n$R^2 = {ols['r2']:.2f}$\n$D^* = {int(opt_d)}$ inhab./km$^2$\n$\\ell = {opt_l:g}$ km",
        canto="lower left",
    )
    plt.tight_layout()
    return salvar(fig, output_basepath)
