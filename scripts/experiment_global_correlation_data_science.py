"""
scripts/experiment_global_correlation_data_science.py
-----------------------------------------------------
Avaliação da Correlação Alométrica em Escala GLOBAL (Mundo Inteiro):
Compara o resultado GLOBAL com:
1. Nurdle Patrol Bruto (Pellets industriais não-limpos).
2. Microplásticos Não-Industriais Limpos (Lixo Urbano / Pós-Consumo: Ocean water e Beach,
   sem garrafas Van Dorn, R_cap <= 25 km, controle de esforço amostral e remoção de outliers).
Analisa tanto globalmente quanto por continente (América do Norte, América do Sul, Europa, África, Austrália).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.spatial_matcher import match_sinks_with_individual_radius
from scripts.find_optimal_cleaning_pipeline import carregar_dados_limpos

def carregar_noaa_nurdles_e_limpos():
    noaa_path = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
    import json
    with open(noaa_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    nurdles_records = []
    clean_records = []

    for feat in gj.get("features", []):
        p = feat.get("properties") or {}
        lat = p.get("Latitude__degree_")
        lon = p.get("Longitude_degree_")
        if lat is None or lon is None:
            continue
        lat, lon = float(lat), float(lon)

        med = str(p.get("Medium", "")).strip()
        method = str(p.get("Sampling_Method", "")).strip()
        val_mp = p.get("Microplastics_measurement")
        val_nurdle = p.get("Standardized_Nurdle__Amount")
        ref = str(p.get("Short_Reference", "")).strip()

        # Nurdle Patrol
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

        # Não-Industriais Limpos (Ocean water e Beach, sem Van Dorn sampler)
        if med in ["Ocean water", "Beach"] and method != "Van Dorn sampler" and val_mp is not None:
            try:
                v = float(val_mp)
                if v > 0:
                    clean_records.append({
                        "latitude": lat,
                        "longitude": lon,
                        "nurdles_per_10min": v,
                        "measurement": v,
                        "medium": med,
                        "method": method,
                        "short_ref": ref,
                    })
            except (ValueError, TypeError):
                pass

    return pd.DataFrame(nurdles_records), pd.DataFrame(clean_records)


def avaliar(xs, ys):
    mask = np.isfinite(xs) & np.isfinite(ys) & (xs > 0) & (ys > 0)
    x = xs[mask]
    y = ys[mask]
    n = len(x)
    if n < 4:
        return {"N": n, "beta": np.nan, "se": np.nan, "r2": np.nan, "r": np.nan, "p_val": np.nan, "rho": np.nan, "rho_p": np.nan}

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
    df_nurdles, df_clean = carregar_noaa_nurdles_e_limpos()
    print(f"Total Nurdles carregados: {len(df_nurdles)}")
    print(f"Total Não-Industriais Limpos carregados: {len(df_clean)}")

    ilhas = [
        ("North & Central America", "north_central_america"),
        ("South America", "south_america"),
        ("Europe & Eurasia", "europe_eurasia"),
        ("Africa", "africa"),
        ("Australia", "australia"),
    ]

    # Tabelas consolidadas
    global_nurdles_list = []
    global_clean_list = []

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

        # 2. Parear Não-Industriais Limpos (R_cap <= 25 km)
        _, sum_cln, _ = match_sinks_with_individual_radius(df_sinks, df_clean, max_cap_km=25.0)
        if len(sum_cln) > 0:
            m_cln = sum_cln.merge(df_pop, on="basin_id", how="inner")
            m_cln["island_name"] = nome
            global_clean_list.append(m_cln)

    df_glob_nurdles = pd.concat(global_nurdles_list, ignore_index=True)
    df_glob_clean = pd.concat(global_clean_list, ignore_index=True)

    print(f"\nBacias globais pareadas com Nurdles: {len(df_glob_nurdles)}")
    print(f"Bacias globais pareadas com Não-Industriais Limpos: {len(df_glob_clean)}")
    print("Por continente (Não-Industriais Limpos):")
    print(df_glob_clean["island_name"].value_counts())

    # Salvar tabela de dados pareados globais limpos
    out_clean_csv = REPO_ROOT / "relatorios" / "dados_pareados_global_nao_industrial_limpo.csv"
    df_glob_clean.to_csv(out_clean_csv, index=False)

    # Executar comparações estatísticas
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

    # Cenários Limpos Não-Industriais
    # Filtro 1: Bruto Não-Industrial
    # Filtro 2: N_samples >= 2
    # Filtro 3: N_samples >= 2 + sem outliers (z <= 2.5)
    # Filtro 4: N_samples >= 3
    cenarios_clean = [
        ("Não-Industrial Bruto", df_glob_clean.copy()),
        ("Não-Industrial (N >= 2)", df_glob_clean[df_glob_clean["n_samples"] >= 2].copy()),
        ("Não-Industrial (N >= 3)", df_glob_clean[df_glob_clean["n_samples"] >= 3].copy()),
    ]

    for desc, df_c in cenarios_clean:
        if len(df_c) < 5:
            continue

        # Com remoção de outliers z <= 2.5
        log_y = np.log10(df_c["nurdles_mean"].values)
        z = np.abs((log_y - np.nanmean(log_y)) / (np.nanstd(log_y) + 1e-9))
        df_no_out = df_c[z <= 2.5].copy()

        for df_curr, out_desc in [(df_c, ""), (df_no_out, " + Sem Outliers (z<=2.5)")]:
            for pop_col in ["urban_population", "total_population"]:
                for met_col in ["nurdles_mean", "nurdles_sum", "nurdles_median"]:
                    res = avaliar(df_curr[pop_col].values, df_curr[met_col].values)
                    res["Cenário"] = desc + out_desc
                    res["Tipo_Plástico"] = "Não-Industrial (Lixo Urbano)"
                    res["População"] = pop_col
                    res["Métrica_MP"] = met_col
                    res["Região"] = "Global (Todas as Ilhas)"
                    resultados.append(res)

    # Também avaliar por continente individual para Não-Industriais Limpos
    df_clean_n2 = df_glob_clean[df_glob_clean["n_samples"] >= 2].copy()
    for nome, _ in ilhas:
        sub_c = df_clean_n2[df_clean_n2["island_name"] == nome]
        if len(sub_c) >= 4:
            for pop_col in ["urban_population", "total_population"]:
                for met_col in ["nurdles_mean", "nurdles_sum"]:
                    res = avaliar(sub_c[pop_col].values, sub_c[met_col].values)
                    if not np.isnan(res["r2"]):
                        res["Cenário"] = "Não-Industrial Limpo (N >= 2)"
                        res["Tipo_Plástico"] = "Não-Industrial (Lixo Urbano)"
                        res["População"] = pop_col
                        res["Métrica_MP"] = met_col
                        res["Região"] = nome
                        resultados.append(res)

    df_res = pd.DataFrame(resultados)
    out_res_csv = REPO_ROOT / "relatorios" / "experimentos_correlacao_global.csv"
    df_res.to_csv(out_res_csv, index=False)
    print(f"\nResultados salvos em {out_res_csv}")

    print("\n" + "=" * 90)
    print("RESULTADOS GLOBAIS: NURDLES (INDUSTRIAL) VS NÃO-INDUSTRIAL LIMPO")
    print("=" * 90)
    glob_res = df_res[df_res["Região"] == "Global (Todas as Ilhas)"].sort_values(by="r2", ascending=False)
    for idx, r in glob_res.head(20).iterrows():
        sig = "***" if r["p_val"] < 0.01 else ("**" if r["p_val"] < 0.05 else ("*" if r["p_val"] < 0.1 else ""))
        print(f"R²={r['r2']*100:5.1f}% | r={r['r']:+5.2f} (p={r['p_val']:.4f} {sig:3s}) | beta={r['beta']:+5.2f}±{r['se']:.2f} | N={r['N']:3d} | rho={r['rho']:+5.2f} | [{r['Tipo_Plástico'][:18]}] {r['Cenário'][:30]} | {r['População']} vs {r['Métrica_MP']}")

    print("\n" + "=" * 90)
    print("RESULTADOS POR CONTINENTE (NÃO-INDUSTRIAL LIMPO, N >= 2)")
    print("=" * 90)
    cont_res = df_res[df_res["Região"] != "Global (Todas as Ilhas)"].sort_values(by="r2", ascending=False)
    for idx, r in cont_res.iterrows():
        sig = "***" if r["p_val"] < 0.01 else ("**" if r["p_val"] < 0.05 else ("*" if r["p_val"] < 0.1 else ""))
        print(f"R²={r['r2']*100:5.1f}% | r={r['r']:+5.2f} (p={r['p_val']:.4f} {sig:3s}) | beta={r['beta']:+5.2f}±{r['se']:.2f} | N={r['N']:3d} | rho={r['rho']:+5.2f} | {r['Região']:25s} | {r['População']} vs {r['Métrica_MP']}")

    return df_res

if __name__ == "__main__":
    executar_estudo_global()
