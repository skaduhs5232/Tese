"""
scripts/experiment_global_correlation_data_science.py
-----------------------------------------------------
Avaliação da Correlação Alométrica em Escala GLOBAL (Mundo Inteiro):
Compara o resultado GLOBAL entre:
1. Nurdle Patrol Bruto (Pellets industriais pré-produção em `pieces/10 mins`).
2. Microplásticos Secundários Verificados por Malha Padrão (`0.02 <= Mesh_size__mm_ <= 5.0 mm`,
   exclusivamente em concentração volumétrica `Unit == 'pieces/m3'`, sem `Hand picking` e sem
   garrafas/bombas de 1 litro com filtros sub-micrométricos < 20 um).
3. Redes Superficiais de Microplásticos em Água Oceânica (`Neuston net`, `Manta net`,
   `Plankton net`, `Bongo net`, `AVANI net`, malha 0.048 a 0.505 mm, `pieces/m3`).
Analisa tanto globalmente quanto por continente (América do Norte, América do Sul, Europa, África, Austrália).
"""

import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.spatial_matcher import match_sinks_with_individual_radius


def carregar_noaa_nurdles_e_limpos():
    noaa_path = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
    with open(noaa_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    nurdles_records = []
    clean_mesh_records = []
    nets_records = []

    for feat in gj.get("features", []):
        p = feat.get("properties") or {}
        lat = p.get("Latitude__degree_")
        lon = p.get("Longitude_degree_")
        if lat is None or lon is None:
            continue
        lat, lon = float(lat), float(lon)

        med = str(p.get("Medium", "")).strip()
        unit = str(p.get("Unit", "")).strip()
        method = str(p.get("Sampling_Method", "")).strip()
        val_mp = p.get("Microplastics_measurement")
        val_nurdle = p.get("Standardized_Nurdle__Amount")
        mesh_raw = p.get("Mesh_size__mm_")
        ref = str(p.get("Short_Reference", "")).strip()

        # 1. Nurdle Patrol (Pellets industriais pré-produção)
        if med == "Nurdle Patrol" and val_nurdle is not None:
            try:
                v = float(val_nurdle)
                if v >= 0:
                    nurdles_records.append({
                        "latitude": lat,
                        "longitude": lon,
                        "nurdles_per_10min": v,
                        "medium": med,
                        "short_ref": ref,
                    })
            except (ValueError, TypeError):
                pass

        # 2. Microplásticos Verificados por Malha Padrão (0.02 a 5.0 mm, Unit == 'pieces/m3')
        if med in ["Ocean water", "Beach"] and unit == "pieces/m3" and val_mp is not None and mesh_raw is not None:
            try:
                v = float(val_mp)
                mesh_mm = float(mesh_raw)
                if v > 0 and (0.02 <= mesh_mm <= 5.0):
                    rec = {
                        "latitude": lat,
                        "longitude": lon,
                        "nurdles_per_10min": v,
                        "measurement": v,
                        "medium": med,
                        "unit": unit,
                        "method": method,
                        "mesh_mm": mesh_mm,
                        "short_ref": ref,
                    }
                    clean_mesh_records.append(rec)
                    if med == "Ocean water" and method in [
                        "Neuston net",
                        "Manta net",
                        "Plankton net",
                        "Bongo net",
                        "AVANI net",
                    ]:
                        nets_records.append(rec)
            except (ValueError, TypeError):
                pass

    return pd.DataFrame(nurdles_records), pd.DataFrame(clean_mesh_records), pd.DataFrame(nets_records)


def avaliar(xs, ys):
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    x = xs[mask]
    y = ys[mask]
    n = len(x)
    if n < 4:
        return {
            "N": n,
            "beta": np.nan,
            "se": np.nan,
            "r2": np.nan,
            "r": np.nan,
            "p_val": np.nan,
            "rho": np.nan,
            "rho_p": np.nan,
        }

    lx, ly = np.log10(x), np.log10(y)
    slope, intercept, r_val, p_val, se = stats.linregress(lx, ly)
    rho, rho_p = stats.spearmanr(x, y)
    return {
        "N": n,
        "beta": slope,
        "se": se,
        "r2": r_val ** 2,
        "r": r_val,
        "p_val": p_val,
        "rho": rho,
        "rho_p": rho_p,
    }


def executar_estudo_global():
    df_nurdles, df_clean, df_nets = carregar_noaa_nurdles_e_limpos()
    print(f"Total Nurdles carregados: {len(df_nurdles)}")
    print(f"Total Microplásticos Malha Padrão (0.02-5.0 mm, pieces/m³) carregados: {len(df_clean)}")
    print(f"Total Redes Superficiais de Microplásticos carregados: {len(df_nets)}")

    ilhas = [
        ("North & Central America", "north_central_america"),
        ("South America", "south_america"),
        ("Europe & Eurasia", "europe_eurasia"),
        ("Africa", "africa"),
        ("Australia", "australia"),
    ]

    global_nurdles_list = []
    global_clean_list = []
    global_nets_list = []

    for nome, slug in ilhas:
        sinks_p = REPO_ROOT / "relatorios" / "ilhas" / f"sinks_raios_{slug}.csv"
        pop_p = REPO_ROOT / "relatorios" / "ilhas" / f"populacao_todas_bacias_{slug}.csv"

        df_sinks = pd.read_csv(sinks_p)
        df_pop = pd.read_csv(pop_p)

        # 1. Parear Nurdles
        _, sum_nur, _ = match_sinks_with_individual_radius(df_sinks, df_nurdles, max_cap_km=50.0)
        if len(sum_nur) > 0:
            m_nur = sum_nur.merge(df_pop, on="basin_id", how="inner")
            m_nur["island_name"] = nome
            global_nurdles_list.append(m_nur)

        # 2. Parear Microplásticos Malha Padrão (0.02-5.0 mm, R_cap <= 25 km)
        _, sum_cln, _ = match_sinks_with_individual_radius(df_sinks, df_clean, max_cap_km=25.0)
        if len(sum_cln) > 0:
            m_cln = sum_cln.merge(df_pop, on="basin_id", how="inner")
            m_cln["island_name"] = nome
            global_clean_list.append(m_cln)

        # 3. Parear Redes Superficiais de Microplásticos (R_cap <= 25 km)
        _, sum_net, _ = match_sinks_with_individual_radius(df_sinks, df_nets, max_cap_km=25.0)
        if len(sum_net) > 0:
            m_net = sum_net.merge(df_pop, on="basin_id", how="inner")
            m_net["island_name"] = nome
            global_nets_list.append(m_net)

    df_glob_nurdles = pd.concat(global_nurdles_list, ignore_index=True)
    df_glob_clean = pd.concat(global_clean_list, ignore_index=True)
    df_glob_nets = pd.concat(global_nets_list, ignore_index=True)

    print(f"\nBacias globais pareadas com Nurdles: {len(df_glob_nurdles)}")
    print(f"Bacias globais pareadas com Microplásticos (0.02-5.0 mm): {len(df_glob_clean)}")
    print(f"Bacias globais pareadas com Redes Superficiais de Microplásticos: {len(df_glob_nets)}")
    print("Por continente (Microplásticos 0.02-5.0 mm):")
    print(df_glob_clean["island_name"].value_counts())

    # Salvar tabela de dados pareados globais de microplásticos verificados
    out_clean_csv = REPO_ROOT / "relatorios" / "dados_pareados_global_nao_industrial_limpo.csv"
    df_glob_clean.to_csv(out_clean_csv, index=False)

    resultados = []

    cenarios_nurdles = [
        ("Nurdles Bruto Global", df_glob_nurdles, "urban_population", "nurdles_sum"),
        ("Nurdles Bruto Global", df_glob_nurdles, "total_population", "nurdles_sum"),
        ("Nurdles Bruto Global", df_glob_nurdles, "total_population", "nurdles_mean"),
    ]

    for desc, df_c, pop_col, met_col in cenarios_nurdles:
        res = avaliar(df_c[pop_col].values, df_c[met_col].values)
        res["Cenário"] = desc
        res["Tipo_Plástico"] = "Industrial (Nurdles)"
        res["População"] = pop_col
        res["Métrica_MP"] = met_col
        res["Região"] = "Global (Todas as Ilhas)"
        resultados.append(res)

    # Cenários Microplásticos Secundários Verificados (0.02 a 5.0 mm)
    cenarios_clean = [
        ("Microplásticos (0.02-5.0 mm, Todas Bacias)", df_glob_clean.copy(), "Microplásticos (<5mm, Malha 0.02-5mm)"),
        ("Microplásticos (0.02-5.0 mm, N >= 2)", df_glob_clean[df_glob_clean["n_samples"] >= 2].copy(), "Microplásticos (<5mm, Malha 0.02-5mm)"),
        ("Microplásticos (0.02-5.0 mm, N >= 3)", df_glob_clean[df_glob_clean["n_samples"] >= 3].copy(), "Microplásticos (<5mm, Malha 0.02-5mm)"),
        ("Redes Superficiais (0.048-0.505 mm, N >= 2)", df_glob_nets[df_glob_nets["n_samples"] >= 2].copy(), "Redes Superficiais (<5mm)"),
    ]

    for desc, df_c, tipo_mp in cenarios_clean:
        if len(df_c) < 5:
            continue

        log_y = np.log10(df_c["nurdles_mean"].values)
        z = np.abs((log_y - np.nanmean(log_y)) / (np.nanstd(log_y) + 1e-9))
        df_no_out = df_c[z <= 2.5].copy()

        for df_curr, out_desc in [(df_c, ""), (df_no_out, " + Sem Outliers (z<=2.5)")]:
            for pop_col in ["urban_population", "total_population"]:
                for met_col in ["nurdles_mean", "nurdles_sum", "nurdles_median"]:
                    res = avaliar(df_curr[pop_col].values, df_curr[met_col].values)
                    res["Cenário"] = desc + out_desc
                    res["Tipo_Plástico"] = tipo_mp
                    res["População"] = pop_col
                    res["Métrica_MP"] = met_col
                    res["Região"] = "Global (Todas as Ilhas)"
                    resultados.append(res)

    # Avaliar por continente individual para Microplásticos (0.02-5.0 mm, N >= 2)
    df_clean_n2 = df_glob_clean[df_glob_clean["n_samples"] >= 2].copy()
    for nome, _ in ilhas:
        sub_c = df_clean_n2[df_clean_n2["island_name"] == nome]
        if len(sub_c) >= 4:
            for pop_col in ["urban_population", "total_population"]:
                for met_col in ["nurdles_mean", "nurdles_sum"]:
                    res = avaliar(sub_c[pop_col].values, sub_c[met_col].values)
                    if not np.isnan(res["r2"]):
                        res["Cenário"] = "Microplásticos (0.02-5.0 mm, N >= 2)"
                        res["Tipo_Plástico"] = "Microplásticos (<5mm, Malha 0.02-5mm)"
                        res["População"] = pop_col
                        res["Métrica_MP"] = met_col
                        res["Região"] = nome
                        resultados.append(res)

    df_res = pd.DataFrame(resultados)
    out_res_csv = REPO_ROOT / "relatorios" / "experimentos_correlacao_global.csv"
    df_res.to_csv(out_res_csv, index=False)
    print(f"\nResultados salvos em {out_res_csv}")

    print("\n" + "=" * 95)
    print("RESULTADOS GLOBAIS: NURDLES (INDUSTRIAL) VS MICROPLÁSTICOS VERIFICADOS (< 5 mm)")
    print("=" * 95)
    glob_res = df_res[df_res["Região"] == "Global (Todas as Ilhas)"].sort_values(by="r2", ascending=False)
    for idx, r in glob_res.head(20).iterrows():
        sig = "***" if r["p_val"] < 0.01 else ("**" if r["p_val"] < 0.05 else ("*" if r["p_val"] < 0.1 else ""))
        print(
            f"R²={r['r2']*100:5.1f}% | r={r['r']:+5.2f} (p={r['p_val']:.4f} {sig:3s}) "
            f"| beta={r['beta']:+5.2f}±{r['se']:.2f} | N={r['N']:3d} | rho={r['rho']:+5.2f} "
            f"| [{r['Tipo_Plástico'][:22]}] {r['Cenário'][:34]} | {r['População']} vs {r['Métrica_MP']}"
        )

    return df_res


if __name__ == "__main__":
    executar_estudo_global()
