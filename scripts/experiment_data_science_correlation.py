"""
scripts/experiment_data_science_correlation.py
----------------------------------------------
Bateria de Experimentos de Ciência de Dados para Investigação e Otimização da Correlação:
1. Avaliação de múltiplos tipos de dados NOAA: Nurdle Patrol (pieces/10 min), Ocean Water (pieces/m³), Beach (pieces/m³).
2. Normalização do esforço amostral (Soma bruta vs. Média vs. Mediana CPUE).
3. Tratamento e remoção de outliers extremos (derramamentos acidentais agudos com z-score > 3).
4. Definições demográficas: População Urbana do CCA (com D* dinâmico alto) vs. População Total da Bacia (P_total).
5. Estratificação geográfica: Complexo Petroquímico (Golfo do México) vs. Bacias Urbanas Gerais.
6. Modelagem comparativa: OLS linear, Theil-Sen robusto, Huber M-estimator e Spearman rank correlation.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

# Configurar caminhos
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.spatial_matcher import match_sinks_with_individual_radius

def carregar_dados_noaa(geojson_path: Path):
    """Carrega todas as categorias de microplásticos da base NOAA NCEI."""
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    nurdles_records = []
    ocean_records = []
    beach_records = []

    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        lat = props.get("Latitude__degree_")
        lon = props.get("Longitude_degree_")
        if lat is None or lon is None:
            continue

        lat, lon = float(lat), float(lon)
        med = str(props.get("Medium", "")).strip()

        # Nurdle Patrol
        if med == "Nurdle Patrol":
            val = props.get("Standardized_Nurdle__Amount")
            if val is not None:
                nurdles_records.append({
                    "latitude": lat,
                    "longitude": lon,
                    "nurdles_per_10min": float(val),
                    "medium": med,
                    "date": props.get("Date__MM_DD_YYYY_"),
                })
        # Ocean Water (redes Neuston/Manta em água superficial)
        elif med == "Ocean water":
            val = props.get("Microplastics_measurement")
            if val is not None:
                ocean_records.append({
                    "latitude": lat,
                    "longitude": lon,
                    "nurdles_per_10min": float(val), # compatibilidade de coluna
                    "medium": med,
                    "date": props.get("Date__MM_DD_YYYY_"),
                })
        # Beach (sedimentos/resíduos gerais de praia)
        elif med == "Beach":
            val = props.get("Microplastics_measurement")
            if val is not None:
                beach_records.append({
                    "latitude": lat,
                    "longitude": lon,
                    "nurdles_per_10min": float(val),
                    "medium": med,
                    "date": props.get("Date__MM_DD_YYYY_"),
                })

    return pd.DataFrame(nurdles_records), pd.DataFrame(ocean_records), pd.DataFrame(beach_records)


def avaliar_regressoes(x: np.ndarray, y: np.ndarray) -> dict:
    """Calcula métricas estatísticas: OLS, Pearson, Theil-Sen, Siegel e Spearman."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xs, ys = x[mask], y[mask]
    n = len(xs)
    if n < 5:
        return {"n": n, "ols_slope": np.nan, "r2": np.nan, "pearson_r": np.nan, "p_val": np.nan,
                "theil_slope": np.nan, "siegel_slope": np.nan, "spearman_rho": np.nan, "spearman_p": np.nan}

    lx = np.log10(xs)
    ly = np.log10(ys)

    # OLS
    slope, intercept, r_val, p_val, stderr = stats.linregress(lx, ly)
    r2 = r_val ** 2

    # Spearman
    rho, p_rho = stats.spearmanr(xs, ys)

    # Theil-Sen robust (SciPy)
    try:
        theil_res = stats.theilslopes(ly, lx)
        theil_slope = theil_res[0]
    except Exception:
        theil_slope = np.nan

    # Siegel repeated medians (SciPy)
    try:
        siegel_res = stats.siegelslopes(ly, lx)
        siegel_slope = siegel_res[0]
    except Exception:
        siegel_slope = np.nan

    return {
        "n": n,
        "ols_slope": slope,
        "ols_stderr": stderr,
        "r2": r2,
        "pearson_r": r_val,
        "p_val": p_val,
        "theil_slope": theil_slope,
        "siegel_slope": siegel_slope,
        "spearman_rho": rho,
        "spearman_p": p_rho,
    }


def executar_bateria_experimentos():
    print("=" * 70)
    print("INICIANDO BATERIA DE EXPERIMENTOS DE CIÊNCIA DE DADOS (NOAA x POPULAÇÃO)")
    print("=" * 70)

    noaa_path = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
    print("Carregando datasets da NOAA...")
    df_nurdles, df_ocean, df_beach = carregar_dados_noaa(noaa_path)
    print(f"Observações NOAA: Nurdle Patrol = {len(df_nurdles):,}, Ocean water = {len(df_ocean):,}, Beach = {len(df_beach):,}")

    # Carregar população e sinks de North America
    sinks_na = pd.read_csv(REPO_ROOT / "relatorios" / "ilhas" / "sinks_raios_north_central_america.csv")
    pop_na = pd.read_csv(REPO_ROOT / "relatorios" / "ilhas" / "populacao_todas_bacias_north_central_america.csv")

    # Adicionar coordenadas das desembocaduras para identificar Golfo do México
    sinks_coords = sinks_na.groupby("basin_id")[["latitude", "longitude"]].mean().reset_index()
    pop_na = pop_na.merge(sinks_coords, on="basin_id", how="left")
    pop_na["is_gulf"] = (pop_na["latitude"] >= 24) & (pop_na["latitude"] <= 31) & (pop_na["longitude"] >= -98) & (pop_na["longitude"] <= -80)

    print(f"Total de bacias com topografia/população: {len(pop_na)} (Bacias no Golfo do México: {pop_na['is_gulf'].sum()})")

    # Parear Nurdles, Ocean Water e Beach com as bacias
    print("\nPareando NOAA Nurdles aos sinks...")
    _, summary_nurdles, _ = match_sinks_with_individual_radius(sinks_na, df_nurdles, max_cap_km=50.0)

    print("Pareando NOAA Ocean water aos sinks...")
    _, summary_ocean, _ = match_sinks_with_individual_radius(sinks_na, df_ocean, max_cap_km=50.0)

    print("Pareando NOAA Beach aos sinks...")
    _, summary_beach, _ = match_sinks_with_individual_radius(sinks_na, df_beach, max_cap_km=50.0)

    # Dicionário de datasets pareados
    datasets = {
        "Nurdle Patrol (Pellets)": summary_nurdles.merge(pop_na, on="basin_id", how="inner"),
        "Ocean Water (Redes Neuston)": summary_ocean.merge(pop_na, on="basin_id", how="inner"),
        "Beach (Lixo de Praia)": summary_beach.merge(pop_na, on="basin_id", how="inner"),
    }

    resultados = []

    for nome_ds, df_data in datasets.items():
        print(f"\n--- Analisando {nome_ds} (Bacias Pareadas = {len(df_data)}) ---")

        # Configurações de Métricas
        metricas = ["nurdles_sum", "nurdles_mean", "nurdles_median"]
        pops = ["urban_population", "total_population"]

        # 1. Dados Brutos
        for pop_col in pops:
            for met_col in metricas:
                x = df_data[pop_col].values
                y = df_data[met_col].values
                res = avaliar_regressoes(x, y)
                resultados.append({
                    "Dataset": nome_ds,
                    "Cenário": "1. Bruto (Todas as bacias)",
                    "População": pop_col,
                    "Métrica MP": met_col,
                    "N": res["n"],
                    "Beta_OLS": f"{res['ols_slope']:.3f}" if not np.isnan(res['ols_slope']) else "NaN",
                    "R2": f"{res['r2']:.4f}" if not np.isnan(res['r2']) else "NaN",
                    "Pearson_r": f"{res['pearson_r']:+.3f}" if not np.isnan(res['pearson_r']) else "NaN",
                    "p_val": f"{res['p_val']:.4f}" if not np.isnan(res['p_val']) else "NaN",
                    "Theil_Beta": f"{res['theil_slope']:.3f}" if not np.isnan(res['theil_slope']) else "NaN",
                    "Siegel_Beta": f"{res['siegel_slope']:.3f}" if not np.isnan(res['siegel_slope']) else "NaN",
                    "Spearman_rho": f"{res['spearman_rho']:+.3f}" if not np.isnan(res['spearman_rho']) else "NaN",
                    "Spearman_p": f"{res['spearman_p']:.4f}" if not np.isnan(res['spearman_p']) else "NaN",
                })

        # 2. Remoção de Outliers Extremos de Derramamento (z-score > 3 na concentração)
        # Especificamente bacias como a 9873 (365 mil nurdles)
        if len(df_data) > 10:
            log_y = np.log10(np.where(df_data["nurdles_mean"] > 0, df_data["nurdles_mean"], np.nan))
            z_scores = np.abs((log_y - np.nanmean(log_y)) / np.nanstd(log_y))
            df_no_outliers = df_data[z_scores <= 3.0].copy()

            for pop_col in pops:
                for met_col in ["nurdles_mean", "nurdles_median"]:
                    x = df_no_outliers[pop_col].values
                    y = df_no_outliers[met_col].values
                    res = avaliar_regressoes(x, y)
                    resultados.append({
                        "Dataset": nome_ds,
                        "Cenário": "2. Sem Outliers Agudos (z <= 3)",
                        "População": pop_col,
                        "Métrica MP": met_col,
                        "N": res["n"],
                        "Beta_OLS": f"{res['ols_slope']:.3f}" if not np.isnan(res['ols_slope']) else "NaN",
                        "R2": f"{res['r2']:.4f}" if not np.isnan(res['r2']) else "NaN",
                        "Pearson_r": f"{res['pearson_r']:+.3f}" if not np.isnan(res['pearson_r']) else "NaN",
                        "p_val": f"{res['p_val']:.4f}" if not np.isnan(res['p_val']) else "NaN",
                        "Theil_Beta": f"{res['theil_slope']:.3f}" if not np.isnan(res['theil_slope']) else "NaN",
                        "Siegel_Beta": f"{res['siegel_slope']:.3f}" if not np.isnan(res['siegel_slope']) else "NaN",
                        "Spearman_rho": f"{res['spearman_rho']:+.3f}" if not np.isnan(res['spearman_rho']) else "NaN",
                        "Spearman_p": f"{res['spearman_p']:.4f}" if not np.isnan(res['spearman_p']) else "NaN",
                    })

        # 3. Filtro de Consistência Amostral (N_samples >= 3 visitas)
        df_min_samples = df_data[df_data["n_samples"] >= 3].copy()
        if len(df_min_samples) >= 5:
            for pop_col in pops:
                for met_col in ["nurdles_mean", "nurdles_median"]:
                    x = df_min_samples[pop_col].values
                    y = df_min_samples[met_col].values
                    res = avaliar_regressoes(x, y)
                    resultados.append({
                        "Dataset": nome_ds,
                        "Cenário": "3. Amostragem Mínima (N_samples >= 3)",
                        "População": pop_col,
                        "Métrica MP": met_col,
                        "N": res["n"],
                        "Beta_OLS": f"{res['ols_slope']:.3f}" if not np.isnan(res['ols_slope']) else "NaN",
                        "R2": f"{res['r2']:.4f}" if not np.isnan(res['r2']) else "NaN",
                        "Pearson_r": f"{res['pearson_r']:+.3f}" if not np.isnan(res['pearson_r']) else "NaN",
                        "p_val": f"{res['p_val']:.4f}" if not np.isnan(res['p_val']) else "NaN",
                        "Theil_Beta": f"{res['theil_slope']:.3f}" if not np.isnan(res['theil_slope']) else "NaN",
                        "Siegel_Beta": f"{res['siegel_slope']:.3f}" if not np.isnan(res['siegel_slope']) else "NaN",
                        "Spearman_rho": f"{res['spearman_rho']:+.3f}" if not np.isnan(res['spearman_rho']) else "NaN",
                        "Spearman_p": f"{res['spearman_p']:.4f}" if not np.isnan(res['spearman_p']) else "NaN",
                    })

        # 4. Estratificação Espacial (Golfo do México vs. Demais Regiões)
        # Relevante principalmente para Nurdle Patrol
        if nome_ds.startswith("Nurdle"):
            # Golfo do México
            df_gulf = df_data[df_data["is_gulf"] == True]
            df_nongulf = df_data[df_data["is_gulf"] == False]

            for nome_reg, df_reg in [("4. Golfo (Polo Petroquímico)", df_gulf), ("5. Fora do Golfo (Urbano Geral)", df_nongulf)]:
                for pop_col in pops:
                    for met_col in ["nurdles_sum", "nurdles_mean"]:
                        x = df_reg[pop_col].values
                        y = df_reg[met_col].values
                        res = avaliar_regressoes(x, y)
                        resultados.append({
                            "Dataset": nome_ds,
                            "Cenário": nome_reg,
                            "População": pop_col,
                            "Métrica MP": met_col,
                            "N": res["n"],
                            "Beta_OLS": f"{res['ols_slope']:.3f}" if not np.isnan(res['ols_slope']) else "NaN",
                            "R2": f"{res['r2']:.4f}" if not np.isnan(res['r2']) else "NaN",
                            "Pearson_r": f"{res['pearson_r']:+.3f}" if not np.isnan(res['pearson_r']) else "NaN",
                            "p_val": f"{res['p_val']:.4f}" if not np.isnan(res['p_val']) else "NaN",
                            "Theil_Beta": f"{res['theil_slope']:.3f}" if not np.isnan(res['theil_slope']) else "NaN",
                            "Siegel_Beta": f"{res['siegel_slope']:.3f}" if not np.isnan(res['siegel_slope']) else "NaN",
                            "Spearman_rho": f"{res['spearman_rho']:+.3f}" if not np.isnan(res['spearman_rho']) else "NaN",
                            "Spearman_p": f"{res['spearman_p']:.4f}" if not np.isnan(res['spearman_p']) else "NaN",
                        })

    df_res = pd.DataFrame(resultados)
    csv_out = REPO_ROOT / "relatorios" / "experimentos_correlacao_data_science.csv"
    df_res.to_csv(csv_out, index=False)
    print(f"\nBateria concluída! {len(df_res)} configurações avaliadas. Resultados salvos em {csv_out}")
    return df_res

if __name__ == "__main__":
    df_res = executar_bateria_experimentos()
