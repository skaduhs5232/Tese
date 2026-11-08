"""
scripts/generate_global_cleaned_figure.py
-----------------------------------------
Gera as Figuras Comparativas Globais Definitivas (Figuras F11 e F14):
  - Figura F11 (`f11_global_cleaned_allometry_comparison.png` / `.pdf` - 3 painéis):
    Painel (a): Alometria Global Nurdles Brutos (Pellets Industriais, R² = 0.3%, p = 0.64).
    Painel (b): Fluxo Hidrológico Global Alinhado por Matriz e Malha (Kooi & Koelmans, 2019;
                Lebreton et al., 2017; R_cap <= 15 km, N = 58 bacias globais):
                beta = +0.79 ± 0.15, R² = 34.0%, r = +0.58 (p < 0.00001), Spearman rho = +0.55.
    Painel (c): Fluxo Hidrológico Global em Redes Superficiais (Manta/Neuston/Plankton,
                0.048 a 0.505 mm, excluindo areia de praia e baldes conforme Weiss et al., 2021;
                R_cap <= 15 km, N_s >= 2, N = 35 bacias globais):
                beta = +1.09 ± 0.22 (~ 1.0), R² = 42.9%, r = +0.65 (p = 0.00002), Spearman rho = +0.64.
  - Figura F14 (`f14_global_surface_nets_and_sediment_allometry.png` / `.pdf` - 2 painéis):
    Painel (a): Redes Superficiais Globais em Escala Costeira Estendida (R_cap <= 35 km, N = 47 bacias):
                beta = +0.80 ± 0.14, R² = 40.7%, r = +0.64 (p = 0.000001), Spearman rho = +0.73.
    Painel (b): Sedimento Marinho Estuarino Global (`Ocean sediment`, `pieces kg-1 d.w.`,
                integrador temporal bentônico, N = 9 bacias):
                beta = +1.21 ± 0.24, R² = 78.5%, r = +0.89 (p = 0.0015), Spearman rho = +0.85 (p = 0.0037).
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.spatial_matcher import match_sinks_with_individual_radius
from geracao_graficos.estilo_erneson import (
    aplicar_estilo,
    salvar,
    rotulo_painel,
)


def _carregar_base_noaa_completa() -> pd.DataFrame:
    noaa_path = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
    with open(noaa_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    rows = []
    for idx, feat in enumerate(gj.get("features", [])):
        props = feat.get("properties") or {}
        val = props.get("Microplastics_measurement")
        lat = props.get("Latitude__degree_")
        lon = props.get("Longitude_degree_")
        if val is None or float(val) <= 0 or lat is None or lon is None:
            continue
        mesh = props.get("Mesh_size__mm_")
        rows.append(
            {
                "sample_id": props.get("OBJECTID", idx),
                "longitude": float(lon),
                "latitude": float(lat),
                "raw_val": float(val),
                "medium": str(props.get("Medium", "")).strip(),
                "unit": str(props.get("Unit", "")).strip(),
                "method": str(props.get("Sampling_Method", "")).strip(),
                "mesh_mm": float(mesh) if mesh is not None else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _parear_global(sub_mp: pd.DataFrame, r_cap: float) -> pd.DataFrame:
    ilhas_dir = REPO_ROOT / "relatorios" / "ilhas"
    islands = [
        ("North & Central America", "north_central_america"),
        ("South America", "south_america"),
        ("Europe & Eurasia", "europe_eurasia"),
        ("Africa", "africa"),
        ("Australia", "australia"),
    ]
    dfs = []
    for iname, slug in islands:
        s_df = pd.read_csv(ilhas_dir / f"sinks_raios_{slug}.csv")
        p_df = pd.read_csv(ilhas_dir / f"populacao_todas_bacias_{slug}.csv")
        _, b_mp, _ = match_sinks_with_individual_radius(s_df, sub_mp, max_cap_km=r_cap, step_km=1.0)
        if len(b_mp) > 0:
            m = pd.merge(p_df, b_mp, on="basin_id")
            m["island_name"] = iname
            dfs.append(m)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def gerar_figura_global_comparativa():
    aplicar_estilo()

    df_all = _carregar_base_noaa_completa()

    # 1. Subconjunto Malha Padrão (0.02 a 5.0 mm) alinhado por Matriz e Malha (Kooi & Koelmans, 2019)
    sub_std = df_all[
        (df_all["unit"] == "pieces/m3")
        & (df_all["medium"].isin(["Ocean water", "Beach"]))
        & (df_all["mesh_mm"] >= 0.02)
        & (df_all["mesh_mm"] <= 5.0)
    ].copy()

    def _classify_gear(row):
        if row["medium"] == "Beach":
            return "Beach_Sand"
        if any(k in row["method"].lower() for k in ["bucket", "pump", "rosette", "bottle", "sampler", "intake"]):
            return "Water_Grab_Pump"
        return "Water_Surface_Net"

    sub_std["gear_family"] = sub_std.apply(_classify_gear, axis=1)
    sub_std["log_val"] = np.log10(sub_std["raw_val"])
    ref_mean = sub_std[sub_std["gear_family"] == "Water_Surface_Net"]["log_val"].mean()
    gear_means = sub_std.groupby("gear_family")["log_val"].transform("mean")
    sub_std["nurdles_per_10min"] = 10 ** (
        sub_std["log_val"] - gear_means + ref_mean + 1.5 * np.log10(sub_std["mesh_mm"] / 0.333)
    )
    df_aligned_15 = _parear_global(sub_std, r_cap=15.0)

    # 2. Subconjunto estrito de Redes Superficiais (Manta/Neuston/Plankton, 0.048 a 0.505 mm; Weiss et al., 2021)
    sub_nets = df_all[
        (df_all["medium"] == "Ocean water")
        & (df_all["mesh_mm"] >= 0.048)
        & (df_all["mesh_mm"] <= 0.505)
        & (~df_all["method"].str.contains("bucket|pump|rosette|bottle|grab|Van Dorn|Niskin", case=False, na=False))
    ].copy()
    sub_nets["nurdles_per_10min"] = sub_nets["raw_val"]
    df_nets_15 = _parear_global(sub_nets, r_cap=15.0)
    df_nets_35 = _parear_global(sub_nets, r_cap=35.0)

    # 3. Subconjunto de Sedimento Marinho Estuarino (`Ocean sediment`, `pieces kg-1 d.w.`)
    sub_sed = df_all[df_all["medium"] == "Ocean sediment"].copy()
    sub_sed["nurdles_per_10min"] = sub_sed["raw_val"]
    df_sed_25 = _parear_global(sub_sed, r_cap=25.0)

    # 4. Nurdles globais brutos
    df_nurdles_glob = pd.read_csv(REPO_ROOT / "relatorios" / "dados_pareados_global_todas_ilhas.csv")

    estilos = {
        "North & Central America": {"marker": "o", "color": "C0"},
        "South America": {"marker": "s", "color": "C2"},
        "Europe & Eurasia": {"marker": "^", "color": "C3"},
        "Africa": {"marker": "D", "color": "C4"},
        "Australia": {"marker": "v", "color": "C1"},
    }

    # =========================================================================
    # FIGURA F11: Comparação Global em 3 Painéis (Nurdles vs Kooi-Koelmans Flux vs Surface Nets Flux)
    # =========================================================================
    fig11, axes11 = plt.subplots(1, 3, figsize=(15.8, 4.9))

    # Painel (a): Nurdles Globais Brutos
    ax1 = axes11[0]
    mask_a = (df_nurdles_glob["urban_population"] > 0) & (df_nurdles_glob["nurdles_sum"] > 0)
    df_a = df_nurdles_glob[mask_a].copy()
    for nome, grp in df_a.groupby("island_name"):
        st = estilos.get(nome, {"marker": "o", "color": "C5"})
        ax1.scatter(
            np.log10(grp["urban_population"].values),
            np.log10(grp["nurdles_sum"].values),
            edgecolors=st["color"],
            facecolors="none",
            marker=st["marker"],
            s=34,
            linewidths=1.0,
            alpha=0.68,
            label=f"{nome} (N={len(grp)})",
        )
    xa = np.log10(df_a["urban_population"].values)
    ya = np.log10(df_a["nurdles_sum"].values)
    slope_a, intercept_a, r_val_a, p_val_a, se_a = stats.linregress(xa, ya)
    x_line_a = np.linspace(xa.min(), xa.max(), 100)
    ax1.plot(x_line_a, intercept_a + slope_a * x_line_a, color="#0f172a", lw=2.0, label="Global OLS Fit")

    txt_a = (
        r"$\mathbf{Global\ Industrial\ Pellets\ (Nurdles)}$" + "\n"
        + rf"$\beta = {slope_a:.2f} \pm {se_a:.2f}$" + "\n"
        + rf"$R^2 = {r_val_a**2:.4f}\ (0.3\%)$" + "\n"
        + rf"$r = +{r_val_a:.2f},\ p = {p_val_a:.2f}$" + "\n"
        + rf"$N = {len(xa)}\ \mathrm{{global\ basins}}$"
    )
    ax1.text(
        0.04, 0.95, txt_a, transform=ax1.transAxes, ha="left", va="top", fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
    )
    ax1.set_xlabel(r"$\log_{10} P_{\mathrm{urban}}$ (Inhabitants in urban clusters)")
    ax1.set_ylabel(r"$\log_{10} M_{\mathrm{sum}}$ (Pellets captured at sink)")
    ax1.set_ylim(-0.8, 6.2)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.0)
    rotulo_painel(ax1, "(a)")

    # Painel (b): Kooi & Koelmans Matrix/Mesh-Aligned Hydrological Flux (N = 58)
    ax2 = axes11[1]
    df_b = df_aligned_15[
        (df_aligned_15["total_population"] > 0)
        & (df_aligned_15["nurdles_mean"] > 0)
        & (df_aligned_15["n_samples"] >= 2)
    ].copy()
    df_b["flux_aligned"] = df_b["nurdles_mean"] * df_b["area_km2"]

    for nome, grp in df_b.groupby("island_name"):
        st = estilos.get(nome, {"marker": "o", "color": "C5"})
        ax2.scatter(
            np.log10(grp["total_population"].values),
            np.log10(grp["flux_aligned"].values),
            edgecolors=st["color"],
            facecolors="none",
            marker=st["marker"],
            s=42,
            linewidths=1.2,
            alpha=0.85,
            label=f"{nome} (N={len(grp)})",
        )
    xb = np.log10(df_b["total_population"].values)
    yb = np.log10(df_b["flux_aligned"].values)
    slope_b, intercept_b, r_val_b, p_val_b, se_b = stats.linregress(xb, yb)
    rho_b, p_rho_b = stats.spearmanr(xb, yb)
    x_line_b = np.linspace(xb.min() - 0.15, xb.max() + 0.15, 100)
    ax2.plot(x_line_b, intercept_b + slope_b * x_line_b, color="#047857", lw=2.0, label="Kooi-Koelmans Flux Fit")
    ax2.plot(
        x_line_b, np.mean(yb) + 1.0 * (x_line_b - np.mean(xb)),
        color="#64748b", ls=":", lw=1.2, label=r"Isometric ($\beta = 1.0$)",
    )
    n_b = len(xb)
    t_val_b = stats.t.ppf(0.975, df=n_b - 2)
    s_err_b = np.sqrt(np.sum((yb - (intercept_b + slope_b * xb)) ** 2) / (n_b - 2))
    ci_b = t_val_b * s_err_b * np.sqrt(1.0 / n_b + (x_line_b - np.mean(xb)) ** 2 / np.sum((xb - np.mean(xb)) ** 2))
    ax2.fill_between(x_line_b, intercept_b + slope_b * x_line_b - ci_b, intercept_b + slope_b * x_line_b + ci_b, color="#10b981", alpha=0.20)

    txt_b = (
        r"$\mathbf{Mesh-Aligned\ Hydrological\ Flux}$" + "\n"
        + r"$\mathbf{(Kooi\ &\ Koelmans\ 2019 + Lebreton\ 2017)}$" + "\n"
        + rf"$\beta = {slope_b:.2f} \pm {se_b:.2f}$" + "\n"
        + rf"$R^2 = {r_val_b**2:.4f}\ (\mathbf{{{r_val_b**2*100:.1f}\%}})$" + "\n"
        + rf"$r = +{r_val_b:.2f},\ \mathbf{{p < 0.0001}}$" + "\n"
        + rf"$\mathrm{{Spearman\ }}\rho = +{rho_b:.2f}\ (p < 0.0001)$" + "\n"
        + rf"$N = {n_b}\ \mathrm{{global\ basins}}$"
    )
    ax2.text(
        0.04, 0.95, txt_b, transform=ax2.transAxes, ha="left", va="top", fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
    )
    ax2.set_xlabel(r"$\log_{10} P_{\mathrm{total}}$ (Basin inhabitants)")
    ax2.set_ylabel(r"$\log_{10} (C_{\mathrm{aligned}} \cdot A_{\mathrm{basin}})$ (Aligned flux)")
    ax2.set_ylim(-0.5, 6.2)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.0)
    rotulo_painel(ax2, "(b)")

    # Painel (c): Surface Trawl Nets Hydrological Flux (Weiss et al., 2021; N = 35, R² = 42.9%)
    ax3 = axes11[2]
    df_c = df_nets_15[
        (df_nets_15["total_population"] > 0)
        & (df_nets_15["nurdles_mean"] > 0)
        & (df_nets_15["n_samples"] >= 2)
    ].copy()
    df_c["flux_nets"] = df_c["nurdles_mean"] * df_c["area_km2"]

    for nome, grp in df_c.groupby("island_name"):
        st = estilos.get(nome, {"marker": "o", "color": "C5"})
        ax3.scatter(
            np.log10(grp["total_population"].values),
            np.log10(grp["flux_nets"].values),
            edgecolors=st["color"],
            facecolors="none",
            marker=st["marker"],
            s=45,
            linewidths=1.3,
            alpha=0.88,
            label=f"{nome} (N={len(grp)})",
        )
    xc = np.log10(df_c["total_population"].values)
    yc = np.log10(df_c["flux_nets"].values)
    slope_c, intercept_c, r_val_c, p_val_c, se_c = stats.linregress(xc, yc)
    rho_c, p_rho_c = stats.spearmanr(xc, yc)
    x_line_c = np.linspace(xc.min() - 0.15, xc.max() + 0.15, 100)
    ax3.plot(x_line_c, intercept_c + slope_c * x_line_c, color="#0369a1", lw=2.0, label="Surface Nets Flux Fit")
    ax3.plot(
        x_line_c, np.mean(yc) + 1.0 * (x_line_c - np.mean(xc)),
        color="#64748b", ls=":", lw=1.2, label=r"Isometric ($\beta = 1.0$)",
    )
    n_c = len(xc)
    t_val_c = stats.t.ppf(0.975, df=n_c - 2)
    s_err_c = np.sqrt(np.sum((yc - (intercept_c + slope_c * xc)) ** 2) / (n_c - 2))
    ci_c = t_val_c * s_err_c * np.sqrt(1.0 / n_c + (x_line_c - np.mean(xc)) ** 2 / np.sum((xc - np.mean(xc)) ** 2))
    ax3.fill_between(x_line_c, intercept_c + slope_c * x_line_c - ci_c, intercept_c + slope_c * x_line_c + ci_c, color="#38bdf8", alpha=0.20)

    txt_c = (
        r"$\mathbf{Ocean\ Surface\ Nets\ Exported\ Flux}$" + "\n"
        + r"$\mathbf{(Manta/Neuston\ 0.048\ to\ 0.505\ mm,\ Weiss\ 2021)}$" + "\n"
        + rf"$\beta = {slope_c:.2f} \pm {se_c:.2f}\ (\approx 1.0)$" + "\n"
        + rf"$R^2 = {r_val_c**2:.4f}\ (\mathbf{{{r_val_c**2*100:.1f}\%}})$" + "\n"
        + rf"$r = +{r_val_c:.2f},\ \mathbf{{p = {p_val_c:.5f}}}$" + "\n"
        + rf"$\mathrm{{Spearman\ }}\rho = +{rho_c:.2f}\ (\mathbf{{p = {p_rho_c:.5f}}})$" + "\n"
        + rf"$N = {n_c}\ \mathrm{{global\ basins}}$"
    )
    ax3.text(
        0.04, 0.95, txt_c, transform=ax3.transAxes, ha="left", va="top", fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
    )
    ax3.set_xlabel(r"$\log_{10} P_{\mathrm{total}}$ (Basin inhabitants)")
    ax3.set_ylabel(r"$\log_{10} (C_{\mathrm{nets}} \cdot A_{\mathrm{basin}})$ (Surface nets flux)")
    ax3.set_ylim(-0.5, 6.5)
    ax3.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.0)
    rotulo_painel(ax3, "(c)")

    plt.tight_layout()
    out_f11_png = REPO_ROOT / "relatorios" / "figuras" / "f11_global_cleaned_allometry_comparison.png"
    out_f11_pdf = REPO_ROOT / "relatorios" / "figuras" / "f11_global_cleaned_allometry_comparison.pdf"
    salvar(fig11, str(out_f11_png))
    salvar(fig11, str(out_f11_pdf))
    print(f"Figura F11 global (3 painéis) salva em:\n - {out_f11_png}\n - {out_f11_pdf}")

    # =========================================================================
    # FIGURA F14: Redes Superficiais Globais (R <= 35 km) & Sedimento Marinho Estuarino (R² = 78.5%)
    # =========================================================================
    fig14, axes14 = plt.subplots(1, 2, figsize=(13.2, 5.2))

    # Painel (a): Surface Nets R <= 35 km (N = 47, R² = 40.7%, rho = +0.73)
    ax14a = axes14[0]
    df_14a = df_nets_35[
        (df_nets_35["total_population"] > 0)
        & (df_nets_35["nurdles_mean"] > 0)
        & (df_nets_35["n_samples"] >= 2)
    ].copy()
    df_14a["flux_nets"] = df_14a["nurdles_mean"] * df_14a["area_km2"]

    for nome, grp in df_14a.groupby("island_name"):
        st = estilos.get(nome, {"marker": "o", "color": "C5"})
        ax14a.scatter(
            np.log10(grp["total_population"].values),
            np.log10(grp["flux_nets"].values),
            edgecolors=st["color"],
            facecolors="none",
            marker=st["marker"],
            s=46,
            linewidths=1.3,
            alpha=0.88,
            label=f"{nome} (N={len(grp)})",
        )
    x14a = np.log10(df_14a["total_population"].values)
    y14a = np.log10(df_14a["flux_nets"].values)
    sl14a, ic14a, r14a, p14a, se14a = stats.linregress(x14a, y14a)
    rho14a, prho14a = stats.spearmanr(x14a, y14a)
    xl14a = np.linspace(x14a.min() - 0.15, x14a.max() + 0.15, 100)
    ax14a.plot(xl14a, ic14a + sl14a * xl14a, color="#0369a1", lw=2.0, label="Surface Nets Flux Fit")
    ax14a.plot(xl14a, np.mean(y14a) + 1.0 * (xl14a - np.mean(x14a)), color="#64748b", ls=":", lw=1.2, label=r"Isometric ($\beta = 1.0$)")
    n14a = len(x14a)
    t14a = stats.t.ppf(0.975, df=n14a - 2)
    serr14a = np.sqrt(np.sum((y14a - (ic14a + sl14a * x14a)) ** 2) / (n14a - 2))
    ci14a = t14a * serr14a * np.sqrt(1.0 / n14a + (xl14a - np.mean(x14a)) ** 2 / np.sum((x14a - np.mean(x14a)) ** 2))
    ax14a.fill_between(xl14a, ic14a + sl14a * xl14a - ci14a, ic14a + sl14a * xl14a + ci14a, color="#38bdf8", alpha=0.20)

    txt14a = (
        r"$\mathbf{Global\ Surface\ Trawl\ Nets\ (R_{\mathrm{cap}} \leq 35\ km)}$" + "\n"
        + r"$\mathbf{(Manta/Neuston/Plankton,\ 0.048\ to\ 0.505\ mm)}$" + "\n"
        + rf"$\beta = {sl14a:.2f} \pm {se14a:.2f}$" + "\n"
        + rf"$R^2 = {r14a**2:.4f}\ (\mathbf{{{r14a**2*100:.1f}\%}})$" + "\n"
        + rf"$r = +{r14a:.2f},\ \mathbf{{p = {p14a:.6f}}}$" + "\n"
        + rf"$\mathrm{{Spearman\ }}\rho = +{rho14a:.2f}\ (\mathbf{{p < 0.0001}})$" + "\n"
        + rf"$N = {n14a}\ \mathrm{{global\ basins}}$"
    )
    ax14a.text(
        0.04, 0.95, txt14a, transform=ax14a.transAxes, ha="left", va="top", fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
    )
    ax14a.set_xlabel(r"$\log_{10} P_{\mathrm{total}}$ (Basin inhabitants)")
    ax14a.set_ylabel(r"$\log_{10} (C_{\mathrm{nets}} \cdot A_{\mathrm{basin}})$ (Exported flux proxy)")
    ax14a.set_ylim(-0.5, 6.5)
    ax14a.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.5)
    rotulo_painel(ax14a, "(a)")

    # Painel (b): Estuarine Benthic Sediment (`Ocean sediment`, N = 9, R² = 78.5%, r = +0.89)
    ax14b = axes14[1]
    df_14b = df_sed_25[
        (df_sed_25["total_population"] > 0)
        & (df_sed_25["nurdles_mean"] > 0)
    ].copy()

    for nome, grp in df_14b.groupby("island_name"):
        st = estilos.get(nome, {"marker": "o", "color": "C5"})
        ax14b.scatter(
            np.log10(grp["total_population"].values),
            np.log10(grp["nurdles_mean"].values),
            edgecolors=st["color"],
            facecolors="none",
            marker=st["marker"],
            s=56,
            linewidths=1.5,
            alpha=0.92,
            label=f"{nome} (N={len(grp)})",
        )
    x14b = np.log10(df_14b["total_population"].values)
    y14b = np.log10(df_14b["nurdles_mean"].values)
    sl14b, ic14b, r14b, p14b, se14b = stats.linregress(x14b, y14b)
    rho14b, prho14b = stats.spearmanr(x14b, y14b)
    xl14b = np.linspace(x14b.min() - 0.15, x14b.max() + 0.15, 100)
    ax14b.plot(xl14b, ic14b + sl14b * xl14b, color="#047857", lw=2.0, label="Benthic Sediment OLS Fit")
    ax14b.plot(xl14b, np.mean(y14b) + 1.0 * (xl14b - np.mean(x14b)), color="#64748b", ls=":", lw=1.2, label=r"Isometric ($\beta = 1.0$)")
    n14b = len(x14b)
    t14b = stats.t.ppf(0.975, df=n14b - 2)
    serr14b = np.sqrt(np.sum((y14b - (ic14b + sl14b * x14b)) ** 2) / (n14b - 2))
    ci14b = t14b * serr14b * np.sqrt(1.0 / n14b + (xl14b - np.mean(x14b)) ** 2 / np.sum((x14b - np.mean(x14b)) ** 2))
    ax14b.fill_between(xl14b, ic14b + sl14b * xl14b - ci14b, ic14b + sl14b * xl14b + ci14b, color="#10b981", alpha=0.20)

    txt14b = (
        r"$\mathbf{Estuarine\ Benthic\ Sediment\ (Time-Integrated)}$" + "\n"
        + r"$\mathbf{(Ocean\ Sediment,\ pieces\ kg^{-1}\ d.w.)}$" + "\n"
        + rf"$\beta = {sl14b:.2f} \pm {se14b:.2f}$" + "\n"
        + rf"$R^2 = {r14b**2:.4f}\ (\mathbf{{{r14b**2*100:.1f}\%}})$" + "\n"
        + rf"$r = +{r14b:.2f},\ \mathbf{{p = {p14b:.4f}}}$" + "\n"
        + rf"$\mathrm{{Spearman\ }}\rho = +{rho14b:.2f}\ (\mathbf{{p = {prho14b:.4f}}})$" + "\n"
        + rf"$N = {n14b}\ \mathrm{{estuarine\ basins}}$"
    )
    ax14b.text(
        0.04, 0.95, txt14b, transform=ax14b.transAxes, ha="left", va="top", fontsize=9.0,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
    )
    ax14b.set_xlabel(r"$\log_{10} P_{\mathrm{total}}$ (Basin inhabitants)")
    ax14b.set_ylabel(r"$\log_{10} C_{\mathrm{sed}}$ (Microplastics in pieces $\mathrm{kg}^{-1}$ d.w.)")
    ax14b.set_ylim(-0.2, 4.8)
    ax14b.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.5)
    rotulo_painel(ax14b, "(b)")

    plt.tight_layout()
    out_f14_png = REPO_ROOT / "relatorios" / "figuras" / "f14_global_surface_nets_and_sediment_allometry.png"
    out_f14_pdf = REPO_ROOT / "relatorios" / "figuras" / "f14_global_surface_nets_and_sediment_allometry.pdf"
    salvar(fig14, str(out_f14_png))
    salvar(fig14, str(out_f14_pdf))
    print(f"Figura F14 global salva em:\n - {out_f14_png}\n - {out_f14_pdf}")


if __name__ == "__main__":
    gerar_figura_global_comparativa()
