"""
plot_global_allometry_correlation.py
------------------------------------
Generates the scientific figures for the global and regional allometric scaling between
drainage-basin population (urban clusters split across basins by pixel person counts) and
coastal microplastic abundance/load at the river mouth, plus the comparison of scenarios
A (baseline v1), B (v2 hydrology + legacy NOAA) and C (v2 hydrology + new NOAA) across
search radii R = 15, 25, 38, 50 km. Strictly follows `AGENTS.md` and `diretrizes/`:
  - 100% English text
  - Colormap: 'plasma'
  - Vertical colorbar on the right
  - Only exorheic major basins (P = 0.90) with real coastal stations at their mouth
    (open-ocean samples and endorheic basins excluded).
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
for p in (REPO_ROOT, THIS_DIR.parent):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from diretrizes.templates.scientific_plot_utils import (
    set_scientific_style,
    add_panel_label,
    add_vertical_colorbar,
    save_scientific_figure,
)
from calculos import ajuste_escala_log_log
from notebooks.executar_pipeline_global import carregar_noaa_limpo, parear_bacias_noaa, UNIDADES_INFO
from regioes_plot import REGIOES_PLOT

RESULTS_DIR = REPO_ROOT / "notebooks" / "results"
MARCADORES = ["o", "s", "^", "D", "v", "P"]
DLON_GRADE = DLAT_GRADE = 1.0 / 30.0
N_MIN_REGRESSAO = 5


def _texto_fit(fit, rotulo_n="coastal sinks"):
    return (
        f"$\\beta = {fit['beta']:.2f}$ $[{fit['ic95_inf']:.2f}, {fit['ic95_sup']:.2f}]$\n"
        f"Pearson $r = {fit['r']:.2f}$ ($p = {fit['p_r']:.1e}$)\n"
        f"Spearman $\\rho_s = {fit['rho']:.2f}$,  $n = {fit['n']}$ {rotulo_n}"
    )


def _plot_scaling_panel(ax, df_sub, x_col, y_col, title_en, xlabel_en, ylabel_en, letra, cores, show_legend=False):
    fit = ajuste_escala_log_log(df_sub[x_col].to_numpy(float), df_sub[y_col].to_numpy(float), n_boot=5000, semente=42)
    for reg, mk in zip(REGIOES_PLOT, MARCADORES):
        m = df_sub["territorio_id"] == reg["id"]
        if m.any():
            ax.scatter(df_sub.loc[m, x_col], df_sub.loc[m, y_col], color=cores[reg["id"]], marker=mk, s=26,
                       alpha=0.88, edgecolors="#222222", linewidths=0.45,
                       label=f"{reg['label']} ($n={int(m.sum())}$)", zorder=4)
    xs, ys = 10.0 ** np.array(fit["xs"]), 10.0 ** np.array(fit["ys"])
    ax.fill_between(xs, 10.0 ** np.array(fit["banda"][0]), 10.0 ** np.array(fit["banda"][1]),
                    color="#f72585", alpha=0.18, zorder=2)
    ax.plot(xs, ys, color="#111111", lw=1.5, zorder=5)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel_en)
    ax.set_ylabel(ylabel_en)
    ax.set_title(title_en, pad=8)
    ax.grid(True, which="major", ls=":", lw=0.5, color="#cccccc")
    add_panel_label(ax, letra, x=-0.16, y=1.04)
    ax.text(0.03, 0.96, _texto_fit(fit), transform=ax.transAxes, fontsize=6.0, va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor="#bbbbbb", alpha=0.92), zorder=8)
    if show_legend:
        ax.legend(loc="lower right", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=5.3)
    return fit


def _tabela_pareada_r50(df_major, df_noaa):
    df_par = parear_bacias_noaa(df_major, df_noaa, DLON_GRADE, DLAT_GRADE)
    linhas = []
    for un_slug, _, _ in UNIDADES_INFO:
        sub = df_par[(df_par[f"dist_km_{un_slug}"] <= 50.0) & (df_par[f"n_amostras_{un_slug}"] >= 1)
                     & (df_par["pop_total"] > 0) & (df_par[f"media_{un_slug}"] > 0)]
        if sub.empty:
            continue
        sub = (sub.sort_values([f"dist_km_{un_slug}", "area_km2"], ascending=[True, False])
               .drop_duplicates(subset=[f"site_id_{un_slug}"], keep="first"))
        med_m = float(sub[f"media_{un_slug}"].median())
        med_q = float(sub[f"qtd_x_area_{un_slug}"].median())
        linhas.append(pd.DataFrame({
            "territorio_id": sub["territorio_id"].to_numpy(),
            "global_basin_uid": sub["global_basin_uid"].to_numpy(),
            "pop_total": sub["pop_total"].to_numpy(float),
            "area_km2": sub["area_km2"].to_numpy(float),
            "mp_media_norm": sub[f"media_{un_slug}"].to_numpy(float) / med_m,
            "mp_x_area_norm": sub[f"qtd_x_area_{un_slug}"].to_numpy(float) / med_q,
        }))
    return pd.concat(linhas, ignore_index=True).drop_duplicates(subset=["global_basin_uid"], keep="first")


def _figura_cenarios(metricas):
    comp = metricas["comparativo"]
    cenarios = [
        ("A_baseline_v1", "A: baseline v1"),
        ("B_hidrologia_v2_noaa_legado", "B: v2 hydrology + legacy NOAA"),
        ("C_hidrologia_v2_noaa_nova", "C: v2 hydrology + new NOAA"),
    ]
    cenarios = [(k, l) for k, l in cenarios if k in comp]
    cores = plt.get_cmap("plasma")(np.linspace(0.12, 0.80, len(cenarios)))

    fig, (ax_b, ax_r) = plt.subplots(1, 2, figsize=(9.0, 3.6))
    fig.subplots_adjust(wspace=0.40)
    chave = "pop_total__vs__mp_x_area_norm"
    for i, ((k, lab), c) in enumerate(zip(cenarios, cores)):
        f = comp[k][chave]
        if f.get("beta") is None:
            continue
        ax_b.errorbar(i, f["beta"], yerr=[[f["beta"] - f["ic95_inf"]], [f["ic95_sup"] - f["beta"]]],
                      fmt="o", color=c, mec="#222222", ms=7, capsize=4, lw=1.3, zorder=4)
        ax_b.text(i + 0.08, f["beta"], f"$n={f['n']}$\n$r={f['r']:.2f}$", fontsize=6.3, va="center", ha="left")
    ax_b.axhline(0.0, color="#888888", lw=0.8, ls=":")
    rotulos_curtos = {
        "A_baseline_v1": "A\nbaseline v1",
        "B_hidrologia_v2_noaa_legado": "B\nv2 hydrology\n+ legacy NOAA",
        "C_hidrologia_v2_noaa_nova": "C\nv2 hydrology\n+ new NOAA",
    }
    ax_b.set_xticks(range(len(cenarios)))
    ax_b.set_xticklabels([rotulos_curtos[k] for k, _ in cenarios], fontsize=6.6)
    ax_b.set_xlim(-0.5, len(cenarios) - 0.1)
    ax_b.set_ylabel("Scaling exponent $\\beta$ (95% bootstrap CI)")
    ax_b.set_title("Normalized Load $(M\\times A)$ vs. $P$ at $R \\leq 25$ km", pad=8)
    ax_b.grid(True, axis="y", ls=":", lw=0.5, color="#cccccc")
    add_panel_label(ax_b, "a", x=-0.16, y=1.04)

    raios = [15, 25, 38, 50]
    for (un_slug, _, un_label), mk in zip(UNIDADES_INFO, ("o", "s", "^")):
        por_r = metricas["por_unidade_e_raio"].get(un_slug, {})
        rs, ns = [], []
        for r in raios:
            f = por_r.get(f"R_{r}km", {}).get("pop_total__vs__qtd_x_area", {})
            rs.append(f.get("r", np.nan) if f.get("n", 0) >= 3 else np.nan)
            ns.append(f.get("n", 0))
        ax_r.plot(raios, rs, marker=mk, lw=1.2, color=plt.get_cmap("plasma")(0.15 + 0.3 * ("o", "s", "^").index(mk)),
                  label=f"{un_label.replace('³', '$^3$')}\n$n$ = {' / '.join(str(n) for n in ns)}", zorder=4)
    ax_r.axhline(0.0, color="#888888", lw=0.8, ls=":")
    ax_r.set_xticks(raios)
    ax_r.set_ylim(-0.08, 0.82)
    ax_r.set_xlabel("Mouth-to-station search radius $R$ (km)")
    ax_r.set_ylabel("Pearson $r$ (log $P$ vs. log $M\\times A$)")
    ax_r.set_title("Scenario C: Sensitivity to Search Radius", pad=8)
    ax_r.grid(True, ls=":", lw=0.5, color="#cccccc")
    ax_r.legend(loc="upper left", frameon=True, facecolor="#ffffff", edgecolor="#cccccc", fontsize=5.6, ncol=1,
                bbox_to_anchor=(0.0, 1.0))
    add_panel_label(ax_r, "b", x=-0.16, y=1.04)
    save_scientific_figure(fig, str(THIS_DIR / "allometry_scenarios_radius_comparison"), dpi=300)
    plt.close(fig)


def generate_global_allometry_figure():
    set_scientific_style()
    df = pd.read_csv(RESULTS_DIR / "escala_pop_lixo_global.csv")
    df_bacias = pd.read_csv(RESULTS_DIR / "bacias_global_consolidado.csv")
    df_cal = pd.read_csv(RESULTS_DIR / "cca_calibracao_global_resumo.csv")
    with open(RESULTS_DIR / "metricas_alometria_global.json", "r", encoding="utf-8") as f:
        metricas = json.load(f)

    cores_arr = plt.get_cmap("plasma")(np.linspace(0.08, 0.92, len(REGIOES_PLOT)))
    cores = {reg["id"]: c for reg, c in zip(REGIOES_PLOT, cores_arr)}

    fig = plt.figure(figsize=(9.2, 7.6))
    gs = fig.add_gridspec(2, 2, wspace=0.46, hspace=0.38)
    sub_nurdle = df[df["unidade_slug"] == "nurdle_10min"].copy()
    _plot_scaling_panel(fig.add_subplot(gs[0, 0]), sub_nurdle, "pop_total", "mp_media",
                        "Coastal Nurdle Concentration ($R \\leq 25$ km)", "Drainage Basin Population $P$ (inhabitants)",
                        "Coastal Abundance $M$ (pellets / 10 min)", "a", cores)
    _plot_scaling_panel(fig.add_subplot(gs[0, 1]), sub_nurdle, "pop_total", "mp_x_area",
                        "Coastal Nurdle Load Index ($R \\leq 25$ km)", "Drainage Basin Population $P$ (inhabitants)",
                        "Load Index $M \\times A$ (pellets/10 min $\\cdot$ km$^2$)", "b", cores)
    df_unicas = df.drop_duplicates(subset=["global_basin_uid"], keep="first").copy()
    _plot_scaling_panel(fig.add_subplot(gs[1, 0]), df_unicas, "pop_total", "mp_x_area_norm_mediana",
                        "All Coastal Regimes, Median-Normalized ($R \\leq 25$ km)",
                        "Drainage Basin Population $P$ (inhabitants)",
                        "Normalized Load $(M \\times A) / \\widetilde{M \\times A}$", "c", cores)
    contagem = df_unicas["territorio_id"].value_counts()
    handles = [
        plt.Line2D([], [], ls="none", marker=mk, ms=6, mfc=cores[reg["id"]], mec="#222222", mew=0.45,
                   label=f"{reg['label']} ($n={int(contagem.get(reg['id'], 0))}$ in c)")
        for reg, mk in zip(REGIOES_PLOT, MARCADORES)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=True, facecolor="#ffffff",
               edgecolor="#cccccc", fontsize=6.6, bbox_to_anchor=(0.5, -0.04))

    ax_d = fig.add_subplot(gs[1, 1])
    tids = [r["id"] for r in REGIOES_PLOT]
    n_split = np.array([float(df_cal.loc[df_cal["territorio_id"] == t, "n_manchas_divididas_entre_bacias"].iloc[0]) for t in tids])
    d_star = np.array([float(df_cal.loc[df_cal["territorio_id"] == t, "D_star_hab_km2"].iloc[0]) for t in tids])
    l_star = np.array([float(df_cal.loc[df_cal["territorio_id"] == t, "l_metros"].iloc[0]) for t in tids])
    norm_d = plt.Normalize(vmin=100.0, vmax=2000.0)
    bars = ax_d.bar(np.arange(len(tids)), n_split, width=0.58, color=plt.get_cmap("plasma")(norm_d(d_star)),
                    edgecolor="#222222", linewidth=0.7, zorder=3)
    sm = plt.cm.ScalarMappable(cmap="plasma", norm=norm_d)
    sm.set_array([])
    add_vertical_colorbar(fig, sm, ax_d, label="Calibrated Zipf $D^*$ (people / km$^2$)")
    for bar, n_val, d_val, l_val in zip(bars, n_split, d_star, l_star):
        ax_d.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height() + 0.02 * n_split.max(),
                  f"{int(n_val)}\n$D^*={int(d_val)}$\n$L^*={l_val / 1000:g}$ km", ha="center", va="bottom", fontsize=5.4)
    ax_d.set_xticks(np.arange(len(tids)))
    ax_d.set_xticklabels(["N. & C.\nAmerica", "South\nAmerica", "Europe", "Africa", "Asia", "Oceania"], fontsize=6.5)
    ax_d.set_ylim(0, n_split.max() * 1.42)
    ax_d.set_ylabel("Urban Clusters Split Across $\\geq 2$ Basins")
    ax_d.set_title("Urban Clusters Divided by Watersheds", pad=8)
    ax_d.grid(True, axis="y", ls=":", lw=0.5, color="#cccccc")
    add_panel_label(ax_d, "d", x=-0.19, y=1.05)
    save_scientific_figure(fig, str(THIS_DIR / "global_allometry_population_microplastics"), dpi=300)
    plt.close(fig)

    _figura_cenarios(metricas)

    df_major = df_bacias[~df_bacias["is_endorheic"] & df_bacias["in_p90"]].copy()
    df_noaa = carregar_noaa_limpo(REPO_ROOT / "dados" / "noaa_microplastics.csv", apenas_costeiro=True)
    df_r50 = _tabela_pareada_r50(df_major, df_noaa)
    norm_a = LogNorm(vmin=4500.0, vmax=6.0e6)

    fig6, axes6 = plt.subplots(2, 3, figsize=(11.4, 6.8))
    fig6.subplots_adjust(wspace=0.46, hspace=0.40)
    for idx_t, (reg, mk, ax_r) in enumerate(zip(REGIOES_PLOT, MARCADORES, axes6.ravel())):
        tid, name_en = reg["id"], reg["label"]
        sub_r = df_r50[df_r50["territorio_id"] == tid]
        x_r = sub_r["pop_total"].to_numpy(float)
        a_r = sub_r["area_km2"].to_numpy(float)
        ax_r.scatter(x_r, sub_r["mp_x_area_norm"].to_numpy(float), c=np.clip(a_r, 4500.0, 6.0e6), cmap="plasma",
                     norm=norm_a, marker=mk, s=30, alpha=0.88, edgecolors="#222222", linewidths=0.45, zorder=4)
        sm_r = plt.cm.ScalarMappable(cmap="plasma", norm=norm_a)
        sm_r.set_array([])
        add_vertical_colorbar(fig6, sm_r, ax_r, label="Basin area $A$ (km$^2$)", shrink=0.84)
        ax_r.set_xscale("log")
        ax_r.set_yscale("log")
        if len(sub_r) >= N_MIN_REGRESSAO:
            y_r = sub_r["mp_x_area_norm"].to_numpy(float)
            fit_r = ajuste_escala_log_log(x_r, y_r, n_boot=2000, semente=42)
            xs = 10.0 ** np.array(fit_r["xs"])
            ax_r.fill_between(xs, 10.0 ** np.array(fit_r["banda"][0]), 10.0 ** np.array(fit_r["banda"][1]),
                              color="#f72585", alpha=0.18, zorder=2)
            ax_r.plot(xs, 10.0 ** np.array(fit_r["ys"]), color="#111111", lw=1.4, zorder=5)
            ax_r.set_ylim(max(1e-4, y_r.min() / 10.0), min(1e5, y_r.max() * 30.0))
            ax_r.text(0.04, 0.95, _texto_fit(fit_r, "sinks"), transform=ax_r.transAxes, fontsize=5.7, va="top",
                      ha="left", bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#bbbbbb", alpha=0.92),
                      zorder=8)
        else:
            ax_r.set_xlim(1e4, 1e8)
            ax_r.set_ylim(1e-2, 1e2)
            ax_r.text(0.5, 0.5, f"Matched coastal sinks: $n = {len(sub_r)}$\n(insufficient for regression, $n < {N_MIN_REGRESSAO}$)",
                      transform=ax_r.transAxes, fontsize=6.2, va="center", ha="center",
                      bbox=dict(boxstyle="round,pad=0.3", facecolor="#f8f9fa", edgecolor="#cccccc", alpha=0.95), zorder=8)
        ax_r.set_xlabel("Basin Population $P$ (inhabitants)")
        ax_r.set_ylabel("Normalized Load $(M\\times A)/\\widetilde{M\\times A}$")
        ax_r.set_title(f"{name_en}\n$R \\leq 50$ km, $n={len(sub_r)}$", fontsize=7.6, pad=5)
        ax_r.grid(True, which="major", ls=":", lw=0.5, color="#cccccc")
        add_panel_label(ax_r, chr(ord("a") + idx_t), x=-0.24, y=1.05)

        if len(sub_r) >= N_MIN_REGRESSAO:
            fig_ind, axes_ind = plt.subplots(1, 2, figsize=(8.8, 3.7))
            fig_ind.subplots_adjust(wspace=0.46)
            for ax_i, y_c, ylab_i, tit_i, let_i in (
                (axes_ind[0], "mp_media_norm", "Normalized Concentration $M / \\widetilde{M}$", f"{name_en}: Concentration", "a"),
                (axes_ind[1], "mp_x_area_norm", "Normalized Load $(M \\times A) / \\widetilde{M \\times A}$", f"{name_en}: Load Index", "b"),
            ):
                yi = sub_r[y_c].to_numpy(float)
                sc_i = ax_i.scatter(x_r, yi, c=np.clip(a_r, 4500.0, 6.0e6), cmap="plasma", norm=norm_a, marker=mk,
                                    s=30, alpha=0.88, edgecolors="#222222", linewidths=0.45, zorder=4)
                add_vertical_colorbar(fig_ind, sc_i, ax_i, label="Basin area $A$ (km$^2$)")
                f_i = ajuste_escala_log_log(x_r, yi, n_boot=2000, semente=42)
                xs_i = 10.0 ** np.array(f_i["xs"])
                ax_i.fill_between(xs_i, 10.0 ** np.array(f_i["banda"][0]), 10.0 ** np.array(f_i["banda"][1]),
                                  color="#f72585", alpha=0.18, zorder=2)
                ax_i.plot(xs_i, 10.0 ** np.array(f_i["ys"]), color="#111111", lw=1.4, zorder=5)
                ax_i.set_xscale("log")
                ax_i.set_yscale("log")
                ax_i.set_ylim(max(1e-4, yi.min() / 8.0), min(1e5, yi.max() * 25.0))
                ax_i.set_xlabel("Drainage Basin Population $P$ (inhabitants)")
                ax_i.set_ylabel(ylab_i)
                ax_i.set_title(f"{tit_i} ($R \\leq 50$ km)", pad=7)
                ax_i.grid(True, which="major", ls=":", lw=0.5, color="#cccccc")
                add_panel_label(ax_i, let_i, x=-0.16, y=1.04)
                ax_i.text(0.04, 0.95, _texto_fit(f_i, "sinks"), transform=ax_i.transAxes, fontsize=5.9, va="top",
                          ha="left", bbox=dict(boxstyle="round,pad=0.25", facecolor="#ffffff", edgecolor="#bbbbbb", alpha=0.92),
                          zorder=8)
            save_scientific_figure(fig_ind, str(THIS_DIR / f"allometry_correlation_{tid}"), dpi=300)
            plt.close(fig_ind)
        else:
            for ext in (".png", ".pdf"):
                (THIS_DIR / f"allometry_correlation_{tid}{ext}").unlink(missing_ok=True)

    save_scientific_figure(fig6, str(THIS_DIR / "allometry_correlation_6regions"), dpi=300)
    plt.close(fig6)


if __name__ == "__main__":
    generate_global_allometry_figure()
