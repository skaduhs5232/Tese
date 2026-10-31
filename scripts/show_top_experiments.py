import pandas as pd

df = pd.read_csv("relatorios/experimentos_correlacao_data_science.csv", encoding="utf-8")
print(f"Total de cenários testados: {len(df)}")

print("\n--- TOP 15 CENÁRIOS POR R² ---")
top = df.sort_values(by="R2", ascending=False)
for idx, r in top.head(15).iterrows():
    print(f"[{r['Dataset']}] {r['Cenário']} | {r['População']} vs {r['Métrica MP']} | N={r['N']} | beta={r['Beta_OLS']:.3f} | R2={r['R2']:.4f} ({r['R2']*100:.1f}%) | r={r['Pearson_r']:.3f} | p={r['p_val']:.4f} | Spearman rho={r['Spearman_rho']:.3f} (p={r['Spearman_p']:.4f})")

print("\n--- RESUMO POR DATASET ---")
for dataset, grp in df.groupby("Dataset"):
    print(f"\nDataset: {dataset} (n_testes={len(grp)})")
    best = grp.sort_values(by="R2", ascending=False).iloc[0]
    print(f"  Melhor R²: {best['R2']*100:.1f}% (beta={best['Beta_OLS']:.3f}, p={best['p_val']:.4f}) no cenário: {best['Cenário']} com {best['População']} vs {best['Métrica MP']}")
