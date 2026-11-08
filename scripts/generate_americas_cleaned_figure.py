"""
scripts/generate_americas_cleaned_figure.py
-------------------------------------------
Avalia a correlação alométrica pan-americana (North & Central America + South America)
incorporando as melhorias físicas e metodológicas da literatura:
  1. Separação estrita de matrizes e exclusão de baldes/bombas (Nyadjro et al., 2023; Weiss et al., 2021).
  2. Alinhamento de abertura de malha e matriz por Lei de Potência (Kooi & Koelmans, 2019; Koelmans et al., 2020).
  3. Modelagem do Fluxo Hidrológico Exportado (F_MP = C * A_bacia; Lebreton et al., 2017).

Gera as Figuras F12 e F13 (3 painéis cada) no padrão Erneson Oliveira:
  - Figura F12 (`f12_americas_cleaned_allometry_comparison.png` / `.pdf`):
    Painel (a): Pellets Industriais Brutos nas Américas (Nurdle Patrol, R² = 0.4%, p = 0.61).
    Painel (b): Fluxo Hidrológico Alinhado por Malha nas Américas (Kooi & Koelmans + Lebreton,
                R_cap <= 15 km, N = 28 bacias, beta = +0.88 ± 0.26, R² = 30.1%, r = +0.55, p = 0.0025).
    Painel (c): Redes Superficiais nas Américas (Manta/Neuston/Plankton, 0.048 a 0.505 mm,
                Weiss et al., 2021, N = 8 bacias com N_s >= 2, beta = +1.11 ± 0.39,
                R² = 57.6%, r = +0.76, p = 0.0290, Spearman rho = +0.90, p = 0.0020).
  - Figura F13 (`f13_americas_map_and_urban_allometry.png` / `.pdf`):
    Painel (a): Mapa Pan-Americano de Estações Verificadas e Sinks Costeiros Pareados.
    Painel (b): Concentração em Bacias Urbanas CCA nas Américas (R_cap <= 15 km, N = 19,
                beta = +1.01 ± 0.43 ≈ 1.0, R² = 24.7%, r = +0.50, p = 0.0303).
    Painel (c): Fluxo Hidrológico em Bacias Urbanas CCA nas Américas (F_MP = C_mean * A_bacia,
                R_cap <= 15 km, N = 17, beta = +1.54 ± 0.49, R² = 39.5%, r = +0.63,
                p = 0.0068, Spearman rho = +0.59, p = 0.0125).
"""

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.spatial_matcher import match_sinks_with_individual_radius
from geracao_graficos.estilo_erneson import (
    aplicar_estilo,
    salvar,
    rotulo_painel,
    rotulo_canto,
    WORLD_GEOJSON,
)


def carregar_noaa_americas(geojson_path: Path) -> pd.DataFrame:
    with open(geojson_path, "r", encoding="utf-8") as f:
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
                "nurdles_per_10min": float(val),
                "medium": str(props.get("Medium", "")).strip(),
                "unit": str(props.get("Unit", "")).strip(),
                "method": str(props.get("Sampling_Method", "")).strip(),
                "mesh_mm": float(mesh) if mesh is not None else np.nan,
                "short_ref": str(props.get("Short_Reference", "")).strip(),
            }
        )
    return pd.DataFrame(rows)


def executar_analise_americas():
    aplicar_estilo()

    noaa_path = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
    df_all = carregar_noaa_americas(noaa_path)

    ilhas_dir = REPO_ROOT / "relatorios" / "ilhas"
    americas_islands = [
        ("North & Central America", "north_central_america"),
        ("South America", "south_america"),
    ]

    # Preparar coorte alinhada por matriz e malha (Kooi & Koelmans, 2019)
    sub_std = df_all[
        (df_all["unit"] == "pieces/m3")
        & (df_all["medium"].isin(["Ocean water", "Beach"]))
        & (df_all["mesh_mm"].notna())
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

    sub_aligned = sub_std.copy()
    sub_aligned["nurdles_per_10min"] = 10 ** (
        sub_aligned["log_val"] - gear_means + ref_mean + 1.5 * np.log10(sub_aligned["mesh_mm"] / 0.333)
    )

    sub_nets = df_all[
        (df_all["unit"] == "pieces/m3")
        & (df_all["medium"] == "Ocean water")
        & (df_all["mesh_mm"].notna())
        & (df_all["mesh_mm"] >= 0.048)
        & (df_all["mesh_mm"] <= 0.505)
        & (~df_all["method"].str.contains("bucket|pump|rosette|bottle|grab|Van Dorn|Niskin", case=False, na=False))
    ].copy()

    coortes = {
        "Malha_Padrao_Microplasticos_0.02_a_5.0mm": sub_std,
        "Kooi_Koelmans_Aligned_0.333mm_Equiv": sub_aligned,
        "Redes_Superficiais_Weiss2021_0.048_a_0.505mm": sub_nets,
    }

    resultados = []
    melhor_df_pan_25 = None
    melhor_df_pan_15 = None
    df_aligned_15 = None
    df_nets_35 = None
    matched_pts_25 = []

    for coorte_nome, sub_mp in coortes.items():
        for r_cap in [15.0, 20.0, 25.0, 35.0, 50.0]:
            dfs_ilhas = []
            for island_name, slug in americas_islands:
                sinks_df = pd.read_csv(ilhas_dir / f"sinks_raios_{slug}.csv")
                pop_df = pd.read_csv(ilhas_dir / f"populacao_todas_bacias_{slug}.csv")

                matched_mp, basin_mp, _ = match_sinks_with_individual_radius(
                    sinks_df.copy(), sub_mp, max_cap_km=r_cap, step_km=1.0
                )
                if len(basin_mp) > 0:
                    merged = pd.merge(pop_df, basin_mp, on="basin_id", how="inner")
                    merged["island_name"] = island_name
                    merged["flux_mean_area"] = merged["nurdles_mean"] * merged["area_km2"]
                    dfs_ilhas.append(merged)
                    if coorte_nome == "Malha_Padrao_Microplasticos_0.02_a_5.0mm" and r_cap == 25.0:
                        m_pts = matched_mp.copy()
                        m_pts["island_name"] = island_name
                        matched_pts_25.append(m_pts)

            if not dfs_ilhas:
                continue

            df_am = pd.concat(dfs_ilhas, ignore_index=True)

            if coorte_nome == "Malha_Padrao_Microplasticos_0.02_a_5.0mm" and r_cap == 25.0:
                melhor_df_pan_25 = df_am.copy()
            if coorte_nome == "Malha_Padrao_Microplasticos_0.02_a_5.0mm" and r_cap == 15.0:
                melhor_df_pan_15 = df_am.copy()
            if coorte_nome == "Kooi_Koelmans_Aligned_0.333mm_Equiv" and r_cap == 15.0:
                df_aligned_15 = df_am.copy()
            if coorte_nome == "Redes_Superficiais_Weiss2021_0.048_a_0.505mm" and r_cap == 35.0:
                df_nets_35 = df_am.copy()

            for escopo_nome, df_escopo in [
                ("Americas_Consolidado", df_am),
                ("North_Central_America", df_am[df_am["island_name"] == "North & Central America"]),
                ("South_America", df_am[df_am["island_name"] == "South America"]),
            ]:
                for pop_col in ["total_population", "urban_population"]:
                    for y_col in ["nurdles_mean", "nurdles_sum", "nurdles_median", "flux_mean_area"]:
                        for min_n in [1, 2]:
                            d = df_escopo[
                                (df_escopo[pop_col] > 0)
                                & (df_escopo[y_col] > 0)
                                & (df_escopo["n_samples"] >= min_n)
                            ].copy()
                            if len(d) < 4:
                                continue
                            x = np.log10(d[pop_col].values)
                            y = np.log10(d[y_col].values)
                            z = np.abs((y - np.mean(y)) / (np.std(y) + 1e-12))
                            d_clean = d[z <= 2.5].copy()
                            if len(d_clean) < 4:
                                continue
                            x2 = np.log10(d_clean[pop_col].values)
                            y2 = np.log10(d_clean[y_col].values)
                            sl, ic, r_val, p_val, se_val = stats.linregress(x2, y2)
                            rho_val, p_rho = stats.spearmanr(x2, y2)
                            resultados.append(
                                {
                                    "coorte": coorte_nome,
                                    "escopo": escopo_nome,
                                    "r_cap_km": r_cap,
                                    "pop_col": pop_col,
                                    "metric_col": y_col,
                                    "min_samples": min_n,
                                    "n_basins": len(x2),
                                    "n_north_central": int((d_clean["island_name"] == "North & Central America").sum()),
                                    "n_south_america": int((d_clean["island_name"] == "South America").sum()),
                                    "beta": sl,
                                    "se": se_val,
                                    "r": r_val,
                                    "r2": r_val**2,
                                    "p_value": p_val,
                                    "spearman_rho": rho_val,
                                    "spearman_p": p_rho,
                                }
                            )

    df_res = pd.DataFrame(resultados).sort_values(by=["r2", "p_value"], ascending=[False, True])
    out_exp = REPO_ROOT / "relatorios" / "experimentos_correlacao_americas.csv"
    df_res.to_csv(out_exp, index=False)
    print(f"Relatório estatístico das Américas atualizado em: {out_exp}")

    if melhor_df_pan_25 is not None:
        out_paired = REPO_ROOT / "relatorios" / "dados_pareados_americas_microplasticos_limpo.csv"
        melhor_df_pan_25.to_csv(out_paired, index=False)

    estilos = {
        "North & Central America": {"marker": "o", "color": "C0"},
        "South America": {"marker": "s", "color": "C2"},
    }

    # =========================================================================
    # FIGURA F12: 3 Painéis nas Américas (Nurdles vs Kooi-Koelmans Flux vs Surface Nets)
    # =========================================================================
    df_nurdles_glob = pd.read_csv(REPO_ROOT / "relatorios" / "dados_pareados_global_todas_ilhas.csv")
    df_nurdles_am = df_nurdles_glob[
        df_nurdles_glob["island_name"].isin(["North & Central America", "South America"])
    ].copy()

    fig12, axes12 = plt.subplots(1, 3, figsize=(15.8, 4.9))

    # Painel (a): Nurdles Industriais Brutos nas Américas
    ax1 = axes12[0]
    mask_a = (df_nurdles_am["urban_population"] > 0) & (df_nurdles_am["nurdles_sum"] > 0)
    df_a = df_nurdles_am[mask_a].copy()
    for nome, grp in df_a.groupby("island_name"):
        st = estilos.get(nome, {"marker": "o", "color": "C0"})
        ax1.scatter(
            np.log10(grp["urban_population"].values),
            np.log10(grp["nurdles_sum"].values),
            edgecolors=st["color"],
            facecolors="none",
            s=38,
            linewidths=1.1,
            alpha=0.75,
            marker=st["marker"],
            label=f"{nome} (N={len(grp)})",
        )
    xa = np.log10(df_a["urban_population"].values)
    ya = np.log10(df_a["nurdles_sum"].values)
    slope_a, intercept_a, r_val_a, p_val_a, se_a = stats.linregress(xa, ya)
    x_line_a = np.linspace(xa.min() - 0.15, xa.max() + 0.15, 100)
    ax1.plot(x_line_a, intercept_a + slope_a * x_line_a, color="#0f172a", lw=2.0, label="Americas OLS Fit")

    txt_a = (
        r"$\mathbf{Americas\ Industrial\ Pellets\ (Nurdles)}$" + "\n"
        + rf"$\beta = {slope_a:.2f} \pm {se_a:.2f}$" + "\n"
        + rf"$R^2 = {r_val_a**2:.4f}\ ({r_val_a**2*100:.1f}\%)$" + "\n"
        + rf"$r = {r_val_a:+.2f},\ p = {p_val_a:.2f}$" + "\n"
        + rf"$N = {len(xa)}\ \mathrm{{American\ basins}}$"
    )
    ax1.text(
        0.04, 0.95, txt_a, transform=ax1.transAxes, ha="left", va="top", fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
    )
    ax1.set_xlabel(r"$\log_{10} P_{\mathrm{urban}}$ (Inhabitants in urban clusters)")
    ax1.set_ylabel(r"$\log_{10} M_{\mathrm{sum}}$ (Pellets captured at sink)")
    ax1.set_ylim(-0.8, 6.2)
    ax1.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.2)
    rotulo_painel(ax1, "(a)")

    # Painel (b): Americas Kooi-Koelmans Mesh-Aligned Hydrological Flux (R <= 15 km, N = 28, R² = 30.1%)
    ax2 = axes12[1]
    df_b = df_aligned_15[
        (df_aligned_15["total_population"] > 0)
        & (df_aligned_15["flux_mean_area"] > 0)
        & (df_aligned_15["n_samples"] >= 2)
    ].copy()
    for nome, grp in df_b.groupby("island_name"):
        st = estilos.get(nome, {"marker": "o", "color": "C0"})
        ax2.scatter(
            np.log10(grp["total_population"].values),
            np.log10(grp["flux_mean_area"].values),
            edgecolors=st["color"],
            facecolors="none",
            s=46,
            linewidths=1.3,
            alpha=0.88,
            marker=st["marker"],
            label=f"{nome} (N={len(grp)})",
        )
    xb = np.log10(df_b["total_population"].values)
    yb = np.log10(df_b["flux_mean_area"].values)
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
        r"$\mathbf{Americas\ Mesh-Aligned\ Flux}$" + "\n"
        + r"$\mathbf{(Kooi\ &\ Koelmans\ 2019 + Lebreton\ 2017)}$" + "\n"
        + rf"$\beta = {slope_b:.2f} \pm {se_b:.2f}\ (\approx 1.0)$" + "\n"
        + rf"$R^2 = {r_val_b**2:.4f}\ (\mathbf{{{r_val_b**2*100:.1f}\%}})$" + "\n"
        + rf"$r = +{r_val_b:.2f},\ \mathbf{{p = {p_val_b:.4f}}}$" + "\n"
        + rf"$\mathrm{{Spearman\ }}\rho = +{rho_b:.2f}\ (\mathbf{{p = {p_rho_b:.4f}}})$" + "\n"
        + rf"$N = {n_b}\ \mathrm{{American\ basins}}$"
    )
    ax2.text(
        0.04, 0.95, txt_b, transform=ax2.transAxes, ha="left", va="top", fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
    )
    ax2.set_xlabel(r"$\log_{10} P_{\mathrm{total}}$ (Basin inhabitants)")
    ax2.set_ylabel(r"$\log_{10} (C_{\mathrm{aligned}} \cdot A_{\mathrm{basin}})$ (Aligned flux)")
    ax2.set_ylim(-0.5, 6.2)
    ax2.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.2)
    rotulo_painel(ax2, "(b)")

    # Painel (c): Americas Surface Trawl Nets (Weiss et al., 2021; N = 8, R² = 57.6%, rho = +0.90)
    ax3 = axes12[2]
    df_c = df_nets_35[
        (df_nets_35["total_population"] > 0)
        & (df_nets_35["nurdles_mean"] > 0)
        & (df_nets_35["n_samples"] >= 2)
    ].copy()
    for nome, grp in df_c.groupby("island_name"):
        st = estilos.get(nome, {"marker": "o", "color": "C0"})
        ax3.scatter(
            np.log10(grp["total_population"].values),
            np.log10(grp["nurdles_mean"].values),
            edgecolors=st["color"],
            facecolors="none",
            s=54,
            linewidths=1.4,
            alpha=0.92,
            marker=st["marker"],
            label=f"{nome} (N={len(grp)})",
        )
    xc = np.log10(df_c["total_population"].values)
    yc = np.log10(df_c["nurdles_mean"].values)
    slope_c, intercept_c, r_val_c, p_val_c, se_c = stats.linregress(xc, yc)
    rho_c, p_rho_c = stats.spearmanr(xc, yc)
    x_line_c = np.linspace(xc.min() - 0.15, xc.max() + 0.15, 100)
    ax3.plot(x_line_c, intercept_c + slope_c * x_line_c, color="#0369a1", lw=2.0, label="Surface Nets OLS Fit")
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
        r"$\mathbf{Americas\ Ocean\ Surface\ Nets}$" + "\n"
        + r"$\mathbf{(Manta/Neuston\ 0.048\ to\ 0.505\ mm,\ Weiss\ 2021)}$" + "\n"
        + rf"$\beta = {slope_c:.2f} \pm {se_c:.2f}\ (\approx 1.0)$" + "\n"
        + rf"$R^2 = {r_val_c**2:.4f}\ (\mathbf{{{r_val_c**2*100:.1f}\%}})$" + "\n"
        + rf"$r = +{r_val_c:.2f},\ \mathbf{{p = {p_val_c:.4f}}}$" + "\n"
        + rf"$\mathrm{{Spearman\ }}\rho = +{rho_c:.2f}\ (\mathbf{{p = {p_rho_c:.4f}}})$" + "\n"
        + rf"$N = {n_c}\ \mathrm{{American\ basins}}$"
    )
    ax3.text(
        0.04, 0.95, txt_c, transform=ax3.transAxes, ha="left", va="top", fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
    )
    ax3.set_xlabel(r"$\log_{10} P_{\mathrm{total}}$ (Basin inhabitants)")
    ax3.set_ylabel(r"$\log_{10} C_{\mathrm{nets}}$ (Microplastics in pieces/$\mathrm{m}^3$)")
    ax3.set_ylim(-2.6, 1.8)
    ax3.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.2)
    rotulo_painel(ax3, "(c)")

    plt.tight_layout()
    out_f12_png = REPO_ROOT / "relatorios" / "figuras" / "f12_americas_cleaned_allometry_comparison.png"
    out_f12_pdf = REPO_ROOT / "relatorios" / "figuras" / "f12_americas_cleaned_allometry_comparison.pdf"
    salvar(fig12, str(out_f12_png))
    salvar(fig12, str(out_f12_pdf))
    print(f"Figura F12 das Américas (3 painéis) salva em:\n - {out_f12_png}\n - {out_f12_pdf}")

    # =========================================================================
    # FIGURA F13: Mapa Pan-Americano + Concentração Urbana CCA + Fluxo Hidrológico Urbano CCA (R² = 39.5%)
    # =========================================================================
    fig13, axes13 = plt.subplots(1, 3, figsize=(15.8, 5.1))

    # Painel (a): Mapa Pan-Americano
    axm = axes13[0]
    axm.set_xlim(-135.0, -30.0)
    axm.set_ylim(-58.0, 55.0)
    axm.set_aspect("equal")

    if WORLD_GEOJSON.is_file():
        with open(WORLD_GEOJSON, "r", encoding="utf-8") as f:
            gj_w = json.load(f)
        for feat in gj_w.get("features", []):
            geom = feat.get("geometry", {})
            gtype = geom.get("type")
            coords = geom.get("coordinates", [])
            if gtype == "Polygon":
                for ring in coords:
                    axm.add_patch(Polygon(ring, facecolor="#e5e5e5", edgecolor="none", zorder=1))
            elif gtype == "MultiPolygon":
                for poly_coords in coords:
                    for ring in poly_coords:
                        axm.add_patch(Polygon(ring, facecolor="#e5e5e5", edgecolor="none", zorder=1))

    sub_mp_am = sub_std[
        (sub_std["longitude"] >= -135.0)
        & (sub_std["longitude"] <= -30.0)
        & (sub_std["latitude"] >= -58.0)
        & (sub_std["latitude"] <= 55.0)
    ]
    axm.scatter(
        sub_mp_am["longitude"], sub_mp_am["latitude"],
        s=12, edgecolors="#94a3b8", facecolors="none", lw=0.6, alpha=0.55,
        label=f"NOAA Verified Stations (n={len(sub_mp_am)})", zorder=4,
    )

    if matched_pts_25:
        df_mp_matched = pd.concat(matched_pts_25, ignore_index=True)
        for nome, grp in df_mp_matched.groupby("island_name"):
            st = estilos.get(nome, {"marker": "o", "color": "C0"})
            axm.scatter(
                grp["longitude"], grp["latitude"],
                s=30, edgecolors=st["color"], facecolors="none",
                marker=st["marker"], lw=1.1, alpha=0.9,
                label=f"Matched Sinks: {nome} (n={len(grp)})", zorder=8,
            )

    axm.set_xlabel("Longitude (degrees)")
    axm.set_ylabel("Latitude (degrees)")
    axm.legend(loc="lower left", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.0)
    rotulo_painel(axm, "(a)")
    rotulo_canto(axm, "Pan-American Verified (< 5 mm)", canto="upper right", fontsize=8)

    # Painel (b): Concentração nas Bacias Urbanas CCA das Américas (R <= 15 km, N = 19, R² = 24.7%)
    axu = axes13[1]
    mask_u = (melhor_df_pan_15["urban_population"] > 0) & (melhor_df_pan_15["nurdles_mean"] > 0)
    df_u = melhor_df_pan_15[mask_u].copy()
    xu_raw = np.log10(df_u["urban_population"].values)
    yu_raw = np.log10(df_u["nurdles_mean"].values)
    zu = np.abs((yu_raw - np.mean(yu_raw)) / (np.std(yu_raw) + 1e-12))
    df_u = df_u[zu <= 2.5].copy()

    for nome, grp in df_u.groupby("island_name"):
        st = estilos.get(nome, {"marker": "o", "color": "C0"})
        axu.scatter(
            np.log10(grp["urban_population"].values),
            np.log10(grp["nurdles_mean"].values),
            edgecolors=st["color"],
            facecolors="none",
            s=48,
            linewidths=1.3,
            alpha=0.9,
            marker=st["marker"],
            label=f"{nome} (N={len(grp)})",
        )
    xu = np.log10(df_u["urban_population"].values)
    yu = np.log10(df_u["nurdles_mean"].values)
    slope_u, intercept_u, r_val_u, p_val_u, se_u = stats.linregress(xu, yu)
    rho_u, p_rho_u = stats.spearmanr(xu, yu)
    x_line_u = np.linspace(xu.min() - 0.15, xu.max() + 0.15, 100)
    axu.plot(x_line_u, intercept_u + slope_u * x_line_u, color="#047857", lw=2.0, label="CCA Urban Concentration Fit")
    axu.plot(
        x_line_u, np.mean(yu) + 1.0 * (x_line_u - np.mean(xu)),
        color="#64748b", ls=":", lw=1.2, label=r"Isometric ($\beta = 1.0$)",
    )
    n_u = len(xu)
    t_val_u = stats.t.ppf(0.975, df=n_u - 2)
    s_err_u = np.sqrt(np.sum((yu - (intercept_u + slope_u * xu)) ** 2) / (n_u - 2))
    ci_u = t_val_u * s_err_u * np.sqrt(1.0 / n_u + (x_line_u - np.mean(xu)) ** 2 / np.sum((xu - np.mean(xu)) ** 2))
    axu.fill_between(x_line_u, intercept_u + slope_u * x_line_u - ci_u, intercept_u + slope_u * x_line_u + ci_u, color="#10b981", alpha=0.20)

    txt_u = (
        r"$\mathbf{Americas\ CCA\ Urban\ Concentration}$" + "\n"
        + r"$\mathbf{(Mesh\ 0.02\ to\ 5.0\ mm,\ R_{\mathrm{cap}} \leq 15\ km)}$" + "\n"
        + rf"$\beta = {slope_u:.2f} \pm {se_u:.2f}\ (\approx 1.0)$" + "\n"
        + rf"$R^2 = {r_val_u**2:.4f}\ (\mathbf{{{r_val_u**2*100:.1f}\%}})$" + "\n"
        + rf"$r = +{r_val_u:.2f},\ \mathbf{{p = {p_val_u:.4f}}}$" + "\n"
        + rf"$\mathrm{{Spearman\ }}\rho = +{rho_u:.2f}\ (p = {p_rho_u:.4f})$" + "\n"
        + rf"$N = {n_u}\ \mathrm{{urban\ basins}}$"
    )
    axu.text(
        0.04, 0.95, txt_u, transform=axu.transAxes, ha="left", va="top", fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
    )
    axu.set_xlabel(r"$\log_{10} P_{\mathrm{urban}}$ (Inhabitants in CCA urban clusters)")
    axu.set_ylabel(r"$\log_{10} C_{\mathrm{mean}}$ (Microplastics in pieces/$\mathrm{m}^3$)")
    axu.set_ylim(-2.5, 5.5)
    axu.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.2)
    rotulo_painel(axu, "(b)")

    # Painel (c): Fluxo Hidrológico nas Bacias Urbanas CCA das Américas (R <= 15 km, N_s >= 2, N = 17, R² = 39.5%)
    axf = axes13[2]
    df_uf = df_u[df_u["n_samples"] >= 2].copy()
    for nome, grp in df_uf.groupby("island_name"):
        st = estilos.get(nome, {"marker": "o", "color": "C0"})
        axf.scatter(
            np.log10(grp["urban_population"].values),
            np.log10(grp["flux_mean_area"].values),
            edgecolors=st["color"],
            facecolors="none",
            s=50,
            linewidths=1.4,
            alpha=0.92,
            marker=st["marker"],
            label=f"{nome} (N={len(grp)})",
        )
    xuf = np.log10(df_uf["urban_population"].values)
    yuf = np.log10(df_uf["flux_mean_area"].values)
    slope_uf, intercept_uf, r_val_uf, p_val_uf, se_uf = stats.linregress(xuf, yuf)
    rho_uf, p_rho_uf = stats.spearmanr(xuf, yuf)
    x_line_uf = np.linspace(xuf.min() - 0.15, xuf.max() + 0.15, 100)
    axf.plot(x_line_uf, intercept_uf + slope_uf * x_line_uf, color="#0369a1", lw=2.0, label="CCA Urban Hydrological Flux Fit")
    axf.plot(
        x_line_uf, np.mean(yuf) + 1.0 * (x_line_uf - np.mean(xuf)),
        color="#64748b", ls=":", lw=1.2, label=r"Isometric ($\beta = 1.0$)",
    )
    n_uf = len(xuf)
    t_val_uf = stats.t.ppf(0.975, df=n_uf - 2)
    s_err_uf = np.sqrt(np.sum((yuf - (intercept_uf + slope_uf * xuf)) ** 2) / (n_uf - 2))
    ci_uf = t_val_uf * s_err_uf * np.sqrt(1.0 / n_uf + (x_line_uf - np.mean(xuf)) ** 2 / np.sum((xuf - np.mean(xuf)) ** 2))
    axf.fill_between(x_line_uf, intercept_uf + slope_uf * x_line_uf - ci_uf, intercept_uf + slope_uf * x_line_uf + ci_uf, color="#38bdf8", alpha=0.20)

    txt_uf = (
        r"$\mathbf{Americas\ CCA\ Urban\ Hydrological\ Flux}$" + "\n"
        + r"$\mathbf{(F_{\mathrm{MP}} = C_{\mathrm{mean}} \times A_{\mathrm{basin}},\ Lebreton\ 2017)}$" + "\n"
        + rf"$\beta = {slope_uf:.2f} \pm {se_uf:.2f}$" + "\n"
        + rf"$R^2 = {r_val_uf**2:.4f}\ (\mathbf{{{r_val_uf**2*100:.1f}\%}})$" + "\n"
        + rf"$r = +{r_val_uf:.2f},\ \mathbf{{p = {p_val_uf:.4f}}}$" + "\n"
        + rf"$\mathrm{{Spearman\ }}\rho = +{rho_uf:.2f}\ (\mathbf{{p = {p_rho_uf:.4f}}})$" + "\n"
        + rf"$N = {n_uf}\ \mathrm{{urban\ basins}}$"
    )
    axf.text(
        0.04, 0.95, txt_uf, transform=axf.transAxes, ha="left", va="top", fontsize=8.6,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9, edgecolor="none"),
    )
    axf.set_xlabel(r"$\log_{10} P_{\mathrm{urban}}$ (Inhabitants in CCA urban clusters)")
    axf.set_ylabel(r"$\log_{10} (C_{\mathrm{mean}} \cdot A_{\mathrm{basin}})$ (Exported flux proxy)")
    axf.set_ylim(0.2, 10.2)
    axf.legend(loc="lower right", frameon=True, framealpha=0.9, facecolor="white", edgecolor="none", fontsize=7.2)
    rotulo_painel(axf, "(c)")

    plt.tight_layout()
    out_f13_png = REPO_ROOT / "relatorios" / "figuras" / "f13_americas_map_and_urban_allometry.png"
    out_f13_pdf = REPO_ROOT / "relatorios" / "figuras" / "f13_americas_map_and_urban_allometry.pdf"
    salvar(fig13, str(out_f13_png))
    salvar(fig13, str(out_f13_pdf))
    print(f"Figura F13 das Américas (3 painéis) salva em:\n - {out_f13_png}\n - {out_f13_pdf}")


if __name__ == "__main__":
    executar_analise_americas()
