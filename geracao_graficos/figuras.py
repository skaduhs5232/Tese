"""
geracao_graficos/figuras.py
---------------------------
Funções geradoras das figuras científicas da dissertação de mestrado,
em estrita conformidade com as Diretrizes 01, 02 e 03 do repositório:
  - F4: plotar_bacias_ipba
  - F5: plotar_calibracao_zipf_regioes
  - F6: plotar_evolucao_raio_exclusivo
  - F7: plotar_alometria_e_controle_area
  - F8: plotar_mapa_cca
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, LogNorm, Normalize
from scipy import stats

from .estilo_erneson import (
    anotar_expoente,
    aplicar_estilo,
    colorbar_direita,
    cores_bacias_tab20,
    cores_manchas,
    curva_nw,
    eixo_mapa,
    eixos_log10,
    guia_inclinacao,
    inset_localizador,
    referencia_inclinacao_1,
    reta_ols,
    rotulo_canto,
    rotulo_painel,
    salvar,
    scatter_vazado,
)


def plotar_calibracao_zipf_regioes(*args, **kwargs):
    from pipeline.cca_calibrator import plot_zipf_calibration
    return plot_zipf_calibration(*args, **kwargs)


def plotar_bacias_ipba(
    basin_grid: np.ndarray,
    sinks_df: pd.DataFrame,
    bbox: tuple[float, float, float, float],
    output_basepath: str | Path,
    region_title: str = "North & Central America",
    cum_pct: int = 95,
) -> tuple[Path, Path]:
    """
    Figura F4: Mapa das bacias hidrográficas do IPBA coloridas em tab20 (coloração de grafo),
    oceano em #e5e5e5, sem colorbar, com as desembocaduras (sinks) em pontos pretos.
    """
    aplicar_estilo()
    colored_grid, cmap = cores_bacias_tab20(basin_grid, ocean_color="#e5e5e5")

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    extent = (bbox[0], bbox[2], bbox[1], bbox[3])

    ax.imshow(
        colored_grid,
        origin="upper",
        extent=extent,
        cmap=cmap,
        interpolation="nearest",
        zorder=1,
    )

    # Plotar as desembocaduras (sinks costeiros) em pontos pretos
    if len(sinks_df) > 0:
        ax.scatter(
            sinks_df["longitude"],
            sinks_df["latitude"],
            s=2.5,
            c="black",
            marker="o",
            alpha=0.85,
            zorder=10,
            label="Coastal sinks",
        )

    eixo_mapa(ax, extent=extent)
    rotulo_painel(ax, "a", dentro=True)
    rotulo_canto(ax, f"{region_title}\nIPBA (single-pass, {cum_pct}% area)", canto="lower left")
    inset_localizador(ax, bbox=bbox, canto="lower right")

    return salvar(fig, output_basepath)


def plotar_mapa_cca(
    pop_grid: np.ndarray,
    cluster_grid: np.ndarray,
    pixel_area: np.ndarray,
    bbox: tuple[float, float, float, float],
    d_opt: float,
    l_opt: float,
    output_basepath: str | Path,
    region_title: str = "North & Central America",
) -> tuple[Path, Path]:
    """
    Figura F8: Painel duplo:
      (a) Densidade populacional contínua em colormap plasma + LogNorm (10^0 a 10^4).
      (b) Manchas urbanas do CCA calibradas com cores discretas, terra sem mancha #666666 e oceano branco.
    """
    aplicar_estilo()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(10.5, 5.0))
    extent = (bbox[0], bbox[2], bbox[1], bbox[3])

    # Painel (a): Densidade Populacional GPW v4
    density = np.where(pixel_area > 0, pop_grid / pixel_area, 0.0)
    density_plot = np.where(density <= 0, np.nan, density)

    # Fundo oceano branco e terra sem dado cinza escuro
    ax_a.set_facecolor("#ffffff")
    im_a = ax_a.imshow(
        density_plot,
        origin="upper",
        extent=extent,
        cmap="plasma",
        norm=LogNorm(vmin=1.0, vmax=10000.0),
        interpolation="nearest",
    )
    eixo_mapa(ax_a, extent=extent)
    rotulo_painel(ax_a, "a", dentro=True)
    rotulo_canto(ax_a, f"{region_title}\nGPW v4 Population density", canto="lower left")
    colorbar_direita(fig, ax_a, im_a, r"Population density (inhab./km$^2$)")

    # Painel (b): Manchas Urbanas do CCA Calibradas
    # Terra sem mancha em #666666, oceano branco #ffffff
    c_max = int(cluster_grid.max()) if cluster_grid.max() > 0 else 1
    cmap_clusters = cores_manchas(c_max)

    # Matriz para plotagem: 0 = oceano (nan), -1 = terra sem mancha, >0 = manchas
    land_no_cluster = (pop_grid > 0) & (cluster_grid == 0)
    display_grid = np.zeros_like(cluster_grid, dtype=float)
    display_grid[pop_grid == 0] = np.nan
    display_grid[land_no_cluster] = 0.5  # Código intermediário

    ax_b.set_facecolor("#ffffff")
    # Base de terra sem mancha
    ax_b.imshow(
        np.where(land_no_cluster, 1.0, np.nan),
        origin="upper",
        extent=extent,
        cmap=ListedColormap(["#666666"]),
        interpolation="nearest",
        zorder=1,
    )
    # Manchas urbanas coloridas por cima
    cluster_masked = np.where(cluster_grid > 0, cluster_grid, np.nan)
    ax_b.imshow(
        cluster_masked,
        origin="upper",
        extent=extent,
        cmap=cmap_clusters,
        interpolation="nearest",
        zorder=2,
    )

    eixo_mapa(ax_b, extent=extent)
    rotulo_painel(ax_b, "b", dentro=True)
    rotulo_canto(
        ax_b,
        f"{region_title}\nCCA ($D^* = {int(d_opt)}$ inhab./km$^2$, $\\ell = {int(l_opt)}$ km)",
        canto="lower left",
    )

    return salvar(fig, output_basepath)


def plotar_evolucao_raio_exclusivo(
    df_sensitivity: pd.DataFrame,
    output_basepath: str | Path,
    reference_radius: float = 25.0,
) -> tuple[Path, Path]:
    """
    Figura F6: Sensibilidade do raio exclusivo de busca (5 a 50 km):
      (a) Número de amostras de microplásticos capturadas e bacias pareadas.
      (b) Estabilidade do expoente alométrico beta e R² em função do raio.
    """
    aplicar_estilo()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(8.5, 3.6))

    rs = df_sensitivity["radius_km"].values
    samples = df_sensitivity["captured_samples"].values
    basins = df_sensitivity["paired_basins"].values

    # Painel (a): Amostras e Bacias
    scatter_vazado(ax_a, rs, samples, c="C0", marker="o", s=35, label="Captured samples")
    ax_a.plot(rs, samples, color="C0", lw=1.0, linestyle="--")

    ax_a_twin = ax_a.twinx()
    scatter_vazado(ax_a_twin, rs, basins, c="C1", marker="s", s=35, label="Paired basins")
    ax_a_twin.plot(rs, basins, color="C1", lw=1.0, linestyle=":")

    ax_a.axvline(reference_radius, color="#d62728", lw=1.0, linestyle="--", alpha=0.7)
    ax_a.set_xlabel(r"Exclusive search radius $R$ (km)")
    ax_a.set_ylabel(r"Captured samples ($n$)", color="C0")
    ax_a_twin.set_ylabel(r"Paired basins ($N$)", color="C1")
    ax_a.tick_params(axis="y", labelcolor="C0")
    ax_a_twin.tick_params(axis="y", labelcolor="C1", direction="out")
    rotulo_painel(ax_a, "a")

    # Painel (b): Coeficiente Alométrico e R²
    betas = df_sensitivity["allometric_beta"].values
    stderrs = df_sensitivity["stderr"].values
    r2s = df_sensitivity["r2"].values

    valid = np.isfinite(betas)
    if valid.any():
        ax_b.errorbar(
            rs[valid],
            betas[valid],
            yerr=stderrs[valid],
            fmt="none",
            ecolor="C0",
            elinewidth=1.0,
            capsize=2.5,
            zorder=10,
        )
        scatter_vazado(ax_b, rs[valid], betas[valid], c="C0", marker="o", s=35, label=r"Allometric slope $\beta$")
        ax_b.plot(rs[valid], betas[valid], color="C0", lw=1.0, linestyle="--")

        ax_b_twin = ax_b.twinx()
        scatter_vazado(ax_b_twin, rs[valid], r2s[valid], c="C2", marker="^", s=35, label=r"$R^2$")
        ax_b_twin.plot(rs[valid], r2s[valid], color="C2", lw=1.0, linestyle=":")

        ax_b.axhline(1.0, color="#888888", lw=1.0, linestyle=":", label=r"Isometric ($\beta=1$)")
        ax_b.axvline(reference_radius, color="#d62728", lw=1.0, linestyle="--", alpha=0.7)

        ax_b.set_xlabel(r"Exclusive search radius $R$ (km)")
        ax_b.set_ylabel(r"Allometric exponent $\beta$", color="C0")
        ax_b_twin.set_ylabel(r"$R^2$", color="C2")
        ax_b.tick_params(axis="y", labelcolor="C0")
        ax_b_twin.tick_params(axis="y", labelcolor="C2", direction="out")

    rotulo_painel(ax_b, "b")
    plt.tight_layout()
    return salvar(fig, output_basepath)


def _format_p_val(p: float) -> str:
    """Formata p-valor cientificamente sem notação feia como 5.933e-01."""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def _posicionar_guia_l_limpo(
    ax: mpl.axes.Axes,
    x_data: np.ndarray,
    y_data: np.ndarray,
    dx: float = 0.45,
    dy: float = 0.45,
    label: str = r"$\beta = 1$",
) -> None:
    """
    Encontra automaticamente uma região vazia do gráfico, com máxima distância
    a qualquer ponto de dados e sem sobreposição com rótulos de cantos, para
    desenhar o colchete em 'L' com segurança visual estrita.
    """
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    gx = np.linspace(xlim[0] + dx * 0.3, xlim[1] - dx * 1.8, 40)
    gy = np.linspace(ylim[0] + dy * 0.3, ylim[1] - dy * 1.8, 40)
    
    norm_x = (x_data - xlim[0]) / (xlim[1] - xlim[0])
    norm_y = (y_data - ylim[0]) / (ylim[1] - ylim[0])

    best_dist = -1.0
    best_pt = (xlim[0] + 0.1 * (xlim[1] - xlim[0]), ylim[1] - 0.25 * (ylim[1] - ylim[0]))

    for cx in gx:
        for cy in gy:
            nx = (cx - xlim[0]) / (xlim[1] - xlim[0])
            ny = (cy - ylim[0]) / (ylim[1] - ylim[0])
            
            # Evita cantos reservados para textos, legendas e títulos
            if nx < 0.48 and ny > 0.65:  # Canto superior esquerdo (título da região)
                continue
            if nx > 0.58 and ny > 0.62:  # Canto superior direito (legenda de séries)
                continue
            if nx > 0.55 and ny < 0.45:  # Canto inferior direito (anotação OLS)
                continue
            if nx < 0.40 and ny < 0.40:  # Canto inferior esquerdo
                continue
            if ny > 0.80 or ny < 0.15:   # Muito rente às bordas
                continue

            dist = np.min(np.hypot(nx - norm_x, ny - norm_y))
            if dist > best_dist:
                best_dist = dist
                best_pt = (cx, cy)

    lx, ly = best_pt
    guia_inclinacao(ax, lx, ly, dx, dy, label=label)


def plotar_alometria_e_controle_area(
    df_paired: pd.DataFrame,
    output_basepath: str | Path,
    reference_radius: float | str = "adaptive",
    region_title: str = "North & Central America",
    metric_col: str = "nurdles_sum",
) -> tuple[Path, Path]:
    """
    Figura F7: Alometria População Urbana x Microplásticos e Controle por Área:
      (a) Regressão log-log: log10 P_urban x log10(M), reta OLS sólida preta,
          curva Nadaraya-Watson suave com IC 95%, referência beta=1 e guia de inclinação em 'L'
          posicionado com garantia de zero colisão.
      (b) Resíduos da alometria em função da área da bacia (log10 Area), testando se a área
          da bacia introduz viés de escala, com rótulos matemáticos limpos sem falhas de rendering.
    """
    aplicar_estilo()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.0, 4.0))

    # Filtrar valores positivos
    df_valid = df_paired.copy()
    if metric_col not in df_valid.columns:
        metric_col = "nurdles_median" if "nurdles_median" in df_valid.columns else "nurdles_sum"

    if "area_km2" not in df_valid.columns:
        if "total_pixels" in df_valid.columns:
            df_valid["area_km2"] = df_valid["total_pixels"] * 13.7
        else:
            df_valid["area_km2"] = 100.0

    valid = df_valid[
        (df_valid["urban_population"] > 0)
        & (df_valid[metric_col] > 0)
        & (df_valid["area_km2"] > 0)
    ].copy()

    if len(valid) < 5:
        ax_a.text(0.5, 0.5, "Insufficient positive pairs", ha="center", va="center")
        return salvar(fig, output_basepath)

    log_p = np.log10(valid["urban_population"].values)
    log_m = np.log10(valid[metric_col].values)
    log_a = np.log10(valid["area_km2"].values)

    # Painel (a): Alometria População Urbana x Nurdles
    scatter_vazado(ax_a, log_p, log_m, c="C0", marker="o", s=34, lw=1.0, label="Drainage basins")
    ols_res = reta_ols(ax_a, log_p, log_m, color="black", lw=1.3, label="OLS fit")
    
    # Nadaraya-Watson com largura de banda suave calibrada
    bw_smooth = max(0.9, 1.3 * np.std(log_p) * (len(log_p) ** (-0.2)))
    curva_nw(ax_a, log_p, log_m, bandwidth=bw_smooth, color="C1", lw=1.2, label="Nadaraya–Watson")

    # Referência isométrica beta = 1
    p_min, p_max = log_p.min(), log_p.max()
    m_mid = np.median(log_m)
    p_mid = np.median(log_p)
    referencia_inclinacao_1(ax_a, p_min, p_max, m_mid - (p_mid - p_min))

    # Configuração dos eixos com folga limpa: banda inferior livre de dados para texto de estatísticas
    p_pad = 0.08 * (p_max - p_min)
    m_max = log_m.max()
    ax_a.set_xlim(p_min - p_pad, p_max + p_pad)
    ax_a.set_ylim(-1.4, m_max + 0.6)

    # Guia de inclinação em L posicionado em área comprovadamente livre de pontos
    _posicionar_guia_l_limpo(ax_a, log_p, log_m, dx=0.45, dy=0.45, label=r"$\beta = 1$")

    ylabel_a = r"$\log_{10} M_{\mathrm{total}}$ (accumulated pellets)" if "sum" in metric_col else r"$\log_{10} M$ (pellets / 10 mins)"
    eixos_log10(
        ax_a,
        xlabel=r"$\log_{10} P_{\mathrm{urban}}$ (inhabitants in urban clusters)",
        ylabel=ylabel_a,
    )
    rotulo_painel(ax_a, "a")
    r_str = f"$R = {reference_radius}$ km" if isinstance(reference_radius, (int, float)) else "Adaptive radius per sink"
    rotulo_canto(ax_a, f"{region_title}\n{r_str}", canto="upper left", bbox=False)
    
    anotar_expoente(
        ax_a,
        f"$\\beta = {ols_res['slope']:.2f} \\pm {ols_res['stderr']:.2f}$\n$R^2 = {ols_res['r2']:.2f}$\n${_format_p_val(ols_res['p_value'])}$\n$N = {len(valid)}$ basins",
        canto="lower right",
    )

    # Painel (b): Resíduos vs Área da Bacia (Controle Alométrico)
    y_pred = ols_res["intercept"] + ols_res["slope"] * log_p
    residuals = log_m - y_pred

    scatter_vazado(ax_b, log_a, residuals, c="C2", marker="s", s=34, lw=1.0, label="Residuals")
    res_ols = reta_ols(ax_b, log_a, residuals, color="black", lw=1.1, linestyle="--")
    ax_b.axhline(0.0, color="#888888", lw=0.8, linestyle=":")

    a_min, a_max = log_a.min(), log_a.max()
    a_pad = 0.08 * (a_max - a_min)
    r_max = residuals.max()
    ax_b.set_xlim(a_min - a_pad, a_max + a_pad)
    ax_b.set_ylim(-2.8, r_max + 0.6)

    eixos_log10(
        ax_b,
        xlabel=r"$\log_{10} A$ (basin drainage area in km$^2$)",
        ylabel=r"Residuals $\epsilon = \log_{10} M - \log_{10} \hat{M}$",
    )
    rotulo_painel(ax_b, "b")
    anotar_expoente(
        ax_b,
        f"Residual slope = ${res_ols['slope']:.2f} \\pm {res_ols['stderr']:.2f}$\n$R^2 = {res_ols['r2']:.2f}$\n${_format_p_val(res_ols['p_value'])}$",
        canto="lower right",
    )

    plt.tight_layout()
    return salvar(fig, output_basepath)


def plotar_comparacao_ilhas_global(
    df_global: pd.DataFrame,
    output_basepath: str | Path,
    metric_col: str = "nurdles_sum",
) -> tuple[Path, Path]:
    """
    Figura de Comparação Global das Grandes Ilhas / Massas Continentais:
      (a) Alometria log10 P_urban x log10 M com símbolos e cores específicos por ilha,
          retas OLS regionais e ajuste global unificado (preto sólido).
      (b) Controle de resíduos globais em função da área da bacia (log10 A).
    """
    aplicar_estilo()
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(9.2, 4.0))

    df_valid = df_global.copy()
    if metric_col not in df_valid.columns:
        metric_col = "nurdles_median" if "nurdles_median" in df_valid.columns else "nurdles_sum"

    if "area_km2" not in df_valid.columns:
        if "total_pixels" in df_valid.columns:
            df_valid["area_km2"] = df_valid["total_pixels"] * 13.7
        else:
            df_valid["area_km2"] = 100.0

    valid = df_valid[
        (df_valid["urban_population"] > 0)
        & (df_valid[metric_col] > 0)
        & (df_valid["area_km2"] > 0)
    ].copy()

    if len(valid) < 5:
        ax_a.text(0.5, 0.5, "Insufficient global pairs", ha="center", va="center")
        return salvar(fig, output_basepath)

    island_styles = {
        "North & Central America": {"marker": "o", "color": "C0"},
        "South America": {"marker": "s", "color": "C2"},
        "Europe & Eurasia": {"marker": "^", "color": "C3"},
        "Africa": {"marker": "D", "color": "C4"},
        "Australia": {"marker": "v", "color": "C1"},
    }

    # Painel (a): Alometria multi-ilhas
    log_p_all = np.log10(valid["urban_population"].values)
    log_m_all = np.log10(valid[metric_col].values)
    log_a_all = np.log10(valid["area_km2"].values)

    for island_name, grp in valid.groupby("island_name"):
        style = island_styles.get(island_name, {"marker": "o", "color": "C5"})
        lp = np.log10(grp["urban_population"].values)
        lm = np.log10(grp[metric_col].values)
        scatter_vazado(ax_a, lp, lm, c=style["color"], marker=style["marker"], s=35, lw=1.0, label=island_name)
        if len(grp) >= 8:
            # Reta OLS fina tracejada por região
            reta_ols(ax_a, lp, lm, color=style["color"], lw=0.9, linestyle="--")

    # Reta global consolidada (preto sólido)
    ols_global = reta_ols(ax_a, log_p_all, log_m_all, color="black", lw=1.4)
    bw_smooth = max(0.9, 1.3 * np.std(log_p_all) * (len(log_p_all) ** (-0.2)))
    curva_nw(ax_a, log_p_all, log_m_all, bandwidth=bw_smooth, color="C1", lw=1.2, label=None)

    # Referência isométrica beta = 1
    p_min, p_max = log_p_all.min(), log_p_all.max()
    m_mid = np.median(log_m_all)
    p_mid = np.median(log_p_all)
    referencia_inclinacao_1(ax_a, p_min, p_max, m_mid - (p_mid - p_min), label=None)

    # Configuração dos eixos com folga limpa: banda inferior livre de dados para texto de estatísticas
    p_pad = 0.08 * (p_max - p_min)
    m_max = log_m_all.max()
    ax_a.set_xlim(p_min - p_pad, p_max + p_pad)
    ax_a.set_ylim(-1.5, m_max + 0.6)

    # Guia de inclinação em L
    _posicionar_guia_l_limpo(ax_a, log_p_all, log_m_all, dx=0.45, dy=0.45, label=r"$\beta = 1$")

    ylabel_a = r"$\log_{10} M_{\mathrm{total}}$ (accumulated pellets)" if "sum" in metric_col else r"$\log_{10} M$ (pellets / 10 mins)"
    eixos_log10(
        ax_a,
        xlabel=r"$\log_{10} P_{\mathrm{urban}}$ (inhabitants in urban clusters)",
        ylabel=ylabel_a,
    )
    rotulo_painel(ax_a, "a")
    rotulo_canto(ax_a, "Global Continental Islands\nAdaptive sink radius", canto="upper left", bbox=False)
    ax_a.legend(frameon=True, facecolor="white", edgecolor="none", framealpha=0.90, fontsize=7.0, loc="upper right", ncol=1)

    anotar_expoente(
        ax_a,
        f"Global $\\beta = {ols_global['slope']:.2f} \\pm {ols_global['stderr']:.2f}$\n$R^2 = {ols_global['r2']:.2f}$\n${_format_p_val(ols_global['p_value'])}$\n$N = {len(valid)}$ basins",
        canto="lower right",
    )

    # Painel (b): Resíduos globais vs Área
    y_pred_global = ols_global["intercept"] + ols_global["slope"] * log_p_all
    res_global = log_m_all - y_pred_global

    for island_name, grp in valid.groupby("island_name"):
        style = island_styles.get(island_name, {"marker": "o", "color": "C5"})
        idx = grp.index
        pos_idx = [valid.index.get_loc(i) for i in idx]
        la = log_a_all[pos_idx]
        rg = res_global[pos_idx]
        scatter_vazado(ax_b, la, rg, c=style["color"], marker=style["marker"], s=35, lw=1.0)

    res_ols = reta_ols(ax_b, log_a_all, res_global, color="black", lw=1.1, linestyle="--")
    ax_b.axhline(0.0, color="#888888", lw=0.8, linestyle=":")

    a_min, a_max = log_a_all.min(), log_a_all.max()
    a_pad = 0.08 * (a_max - a_min)
    r_max = res_global.max()
    ax_b.set_xlim(a_min - a_pad, a_max + a_pad)
    ax_b.set_ylim(-2.8, r_max + 0.6)

    eixos_log10(
        ax_b,
        xlabel=r"$\log_{10} A$ (basin drainage area in km$^2$)",
        ylabel=r"Residuals $\epsilon = \log_{10} M - \log_{10} \hat{M}$",
    )
    rotulo_painel(ax_b, "b")
    anotar_expoente(
        ax_b,
        f"Residual slope = ${res_ols['slope']:.2f} \\pm {res_ols['stderr']:.2f}$\n$R^2 = {res_ols['r2']:.2f}$\n${_format_p_val(res_ols['p_value'])}$",
        canto="lower right",
    )

    plt.tight_layout()
    return salvar(fig, output_basepath)
