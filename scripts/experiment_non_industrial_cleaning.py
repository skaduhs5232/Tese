"""
scripts/experiment_non_industrial_cleaning.py
---------------------------------------------
Investigação e limpeza de dados focada em MICROPLÁSTICOS NÃO-INDUSTRIAIS
(Resíduos urbanos / domésticos / pós-consumo da base NOAA NCEI).

Objetivos:
1. Filtrar estritamente dados NÃO-INDUSTRIAIS:
   - Excluir Nurdle Patrol (pellets industriais pré-produção).
   - Utilizar medições de 'Ocean water' (redes Neuston/Manta em águas costeiras/superficiais)
     e 'Beach' (sedimentos/resíduos gerais de praia).
2. Avaliar separadamente ou harmonizar as matrizes (água em pieces/m³ vs sedimento).
3. Testar diferentes filtros de limpeza:
   a) Distância à costa / foz (eliminar pontos de mar aberto/giros oceânicos pelágicos > 50 km).
   b) Remoção de outliers extremos de concentração (z-score > 2.5 em log10).
   c) Controle de esforço amostral (N_samples >= 1, 2, 3, 5).
   d) Métrica de agregação por foz (Mediana vs Média vs Soma vs Percentil 75).
   e) Métrica demográfica da bacia: População Total (P_total) vs População Urbana CCA (P_urban).
   f) Limiar de densidade urbana D* (4000 vs 1000 vs 500 hab/km²).
4. Medir o ganho de correlação (Pearson r, R², Spearman rho, p-valor e expoente beta).
"""

import os
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.spatial_matcher import match_sinks_with_individual_radius

def carregar_dados_noaa_detalhado(geojson_path: Path):
    with open(geojson_path, "r", encoding="utf-8") as f:
        gj = json.load(f)

    records = []
    for feat in gj.get("features", []):
        props = feat.get("properties") or {}
        lat = props.get("Latitude__degree_")
        lon = props.get("Longitude_degree_")
        if lat is None or lon is None:
            continue

        lat, lon = float(lat), float(lon)
        med = str(props.get("Medium", "")).strip()
        method = str(props.get("Sampling_Method", "")).strip()
        val_mp = props.get("Microplastics_measurement")
        val_nurdle = props.get("Standardized_Nurdle__Amount")
        unit = str(props.get("Unit", "")).strip()
        ref = str(props.get("Short_Reference", "")).strip()

        # Classificação de Origem:
        # Nurdle Patrol é 100% industrial pré-produção (pellets de resina)
        if med == "Nurdle Patrol" or val_nurdle is not None:
            source_type = "Industrial (Pellets/Nurdles)"
            val = float(val_nurdle) if val_nurdle is not None else np.nan
        elif med in ["Ocean water", "Beach", "Ocean sediment"]:
            source_type = "Non-Industrial (Urban/Post-Consumer)"
            val = float(val_mp) if val_mp is not None else np.nan
        else:
            source_type = "Unknown"
            val = np.nan

        if np.isfinite(val) and val > 0:
            records.append({
                "latitude": lat,
                "longitude": lon,
                "measurement": val,
                "nurdles_per_10min": val, # compatibilidade com pipeline
                "medium": med,
                "sampling_method": method,
                "unit": unit,
                "source_type": source_type,
                "short_ref": ref,
                "ocean": props.get("Location_Oceans"),
                "date": props.get("Date__MM_DD_YYYY_"),
            })

    return pd.DataFrame(records)


def avaliar_regressao(x: np.ndarray, y: np.ndarray):
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    xs, ys = x[mask], y[mask]
    n = len(xs)
    if n < 4:
        return {"n": n, "beta": np.nan, "se": np.nan, "r2": np.nan, "r": np.nan, "p": np.nan, "rho": np.nan, "p_rho": np.nan}

    lx, ly = np.log10(xs), np.log10(ys)
    slope, intercept, r_val, p_val, stderr = stats.linregress(lx, ly)
    rho, p_rho = stats.spearmanr(xs, ys)
    return {
        "n": n,
        "beta": slope,
        "se": stderr,
        "r2": r_val ** 2,
        "r": r_val,
        "p": p_val,
        "rho": rho,
        "p_rho": p_rho,
    }


def rodar_experimentos():
    print("Carregando bases...")
    noaa_path = REPO_ROOT / "dados" / "noaa_microplastic" / "Hub_Microplastics_Replace_2485283804436522212.geojson"
    df_all = carregar_dados_noaa_detalhado(noaa_path)
    print(f"Total registros NOAA válidos (>0): {len(df_all)}")
    print(df_all["source_type"].value_counts())

    # Filtrar apenas Não-Industriais
    df_non_ind = df_all[df_all["source_type"] == "Non-Industrial (Urban/Post-Consumer)"].copy()
    print(f"\nTotal registros NÃO-INDUSTRIAIS: {len(df_non_ind)}")
    print("Por Medium:\n", df_non_ind["medium"].value_counts())
    print("Por Método de Amostragem (Top 5):\n", df_non_ind["sampling_method"].value_counts().head(5))

    # Carregar topografia e sinks de North & Central America
    sinks_na = pd.read_csv(REPO_ROOT / "relatorios" / "ilhas" / "sinks_raios_north_central_america.csv")
    pop_na = pd.read_csv(REPO_ROOT / "relatorios" / "ilhas" / "populacao_todas_bacias_north_central_america.csv")
    print(f"\nBacias carregadas: {len(pop_na)} (Sinks: {len(sinks_na)})")

    # Experimentos
    cenarios = []

    # 1. Separar por Medium: Ocean water vs Beach vs Ambos
    subconjuntos = {
        "Ocean Water (Redes Neuston/Manta em Água)": df_non_ind[df_non_ind["medium"] == "Ocean water"].copy(),
        "Beach (Resíduos em Sedimento Costeiro)": df_non_ind[df_non_ind["medium"] == "Beach"].copy(),
        "Todos Não-Industriais (Água + Praia)": df_non_ind.copy(),
    }

    # Raio de pareamento com os sinks
    for nome_sub, df_sub in subconjuntos.items():
        if len(df_sub) == 0:
            continue

        # Pareamento espacial com sinks (raio de até 50 km)
        _, summary, _ = match_sinks_with_individual_radius(sinks_na, df_sub, max_cap_km=50.0)
        df_merged = summary.merge(pop_na, on="basin_id", how="inner")

        # Filtros de limpeza
        filtros = {
            "1. Sem filtro (bruto)": df_merged.copy(),
            "2. Limpeza de Outliers (z <= 2.5 log)": None, # calculado dinamicamente
            "3. Amostragem Confiável (N_samples >= 2)": df_merged[df_merged["n_samples"] >= 2].copy(),
            "4. Amostragem Confiável (N_samples >= 3)": df_merged[df_merged["n_samples"] >= 3].copy(),
            "5. Limpeza Outliers + N_samples >= 2": None, # calculado dinamicamente
            "6. Limpeza Outliers + N_samples >= 3": None, # calculado dinamicamente
        }

        # Calcular filtros com outliers
        if len(df_merged) >= 5:
            # Outlier em median
            log_med = np.log10(np.where(df_merged["nurdles_median"] > 0, df_merged["nurdles_median"], np.nan))
            z = np.abs((log_med - np.nanmean(log_med)) / np.nanstd(log_med))
            valid_z = z <= 2.5
            filtros["2. Limpeza de Outliers (z <= 2.5 log)"] = df_merged[valid_z].copy()
            filtros["5. Limpeza Outliers + N_samples >= 2"] = df_merged[valid_z & (df_merged["n_samples"] >= 2)].copy()
            filtros["6. Limpeza Outliers + N_samples >= 3"] = df_merged[valid_z & (df_merged["n_samples"] >= 3)].copy()

        # Testar combinações
        for nome_filtro, df_f in filtros.items():
            if df_f is None or len(df_f) < 4:
                continue

            for pop_col in ["total_population", "urban_population"]:
                for met_col in ["nurdles_median", "nurdles_mean", "nurdles_sum"]:
                    res = avaliar_regressao(df_f[pop_col].values, df_f[met_col].values)
                    if not np.isnan(res["r2"]):
                        cenarios.append({
                            "Dataset": nome_sub,
                            "Filtro_Limpeza": nome_filtro,
                            "População": pop_col,
                            "Métrica_MP": met_col,
                            "N": res["n"],
                            "Beta": res["beta"],
                            "SE": res["se"],
                            "R2": res["r2"],
                            "Pearson_r": res["r"],
                            "p_val": res["p"],
                            "Spearman_rho": res["rho"],
                            "Spearman_p": res["p_rho"],
                        })

    df_res = pd.DataFrame(cenarios)
    out_csv = REPO_ROOT / "relatorios" / "experimentos_limpeza_nao_industrial.csv"
    df_res.to_csv(out_csv, index=False)
    print(f"\nExperimentos concluídos! Total de configurações avaliadas: {len(df_res)}")

    print("\n" + "=" * 80)
    print("TOP 15 CONFIGURAÇÕES DE DADOS NÃO-INDUSTRIAIS POR R²:")
    print("=" * 80)
    top = df_res.sort_values(by="R2", ascending=False).head(15)
    for idx, r in top.iterrows():
        sig = " ***" if r["p_val"] < 0.01 else (" **" if r["p_val"] < 0.05 else (" *" if r["p_val"] < 0.1 else ""))
        print(f"R²={r['R2']*100:5.1f}% | r={r['Pearson_r']:+5.2f} (p={r['p_val']:.4f}{sig}) | beta={r['Beta']:+5.2f}±{r['SE']:.2f} | N={r['N']:2d} | rho={r['Spearman_rho']:+5.2f} | [{r['Dataset'][:25]}] {r['Filtro_Limpeza'][:25]} | {r['População']} vs {r['Métrica_MP']}")

    return df_res

if __name__ == "__main__":
    rodar_experimentos()
