import os
import pandas as pd

# Nome da pasta de saída
output_dir = "deteccoes_filtradas"
os.makedirs(output_dir, exist_ok=True)

# Intervalos atualizados (POSTE aparece o tempo todo)
intervalos = {
    "ECO_LETHICIA": (10, 35),
    "BOSS": (40, 675),
    "VORTEX": (40, 675),
    "ECOBOSS": (40, 675),
    "SUMMON": (40, 675),
    "MAOBOSS": (40, 675),
    "TECLADO": (679, float("inf")),
    "POSTE": (0, float("inf"))
}

# Caminhos dos arquivos
arquivos = {
    "BOSS": "deteccao_BOSS.csv",
    "ECOBOSS": "deteccao_ECOBOSS.csv",
    "ECO_LETHICIA": "deteccao_ECO_LETHICIA.csv",
    "MAOBOSS": "deteccao_MAOBOSS.csv",
    "POSTE": "deteccao_POSTE.csv",
    "SUMMON": "deteccao_SUMMON.csv",
    "TECLADO": "deteccao_TECLADO.csv",
    "VORTEX": "deteccao_VORTEX.csv",
}

# Definição das faixas de score
faixas_score = {
    "0.50–0.60": (0.5, 0.6),
    "0.61–0.70": (0.61, 0.7),
    "0.71–0.80": (0.71, 0.8),
    "0.81–0.90": (0.81, 0.9),
    "0.91–0.99": (0.91, 0.99)
}

# Filtrar, salvar e mostrar resumo
resumo = {}
for nome, caminho in arquivos.items():
    df = pd.read_csv(caminho)
    ini, fim = intervalos[nome]
    df_filtrado = df[(df["tempo_s"] >= ini) & (df["tempo_s"] <= fim)]

    # Salvar arquivo filtrado
    out_path = os.path.join(output_dir, f"deteccao_{nome}_filtrado.csv")
    df_filtrado.to_csv(out_path, index=False)

    # Contagem de faixas de score
    contagem_scores = {}
    for faixa_nome, (s_ini, s_fim) in faixas_score.items():
        contagem_scores[faixa_nome] = len(df_filtrado[(df_filtrado["score"] >= s_ini) & (df_filtrado["score"] <= s_fim)])

    resumo[nome] = contagem_scores

# Exibir o resumo
print("\nResumo dos dados filtrados por faixa de score:")
for nome, contagem in resumo.items():
    print(f"\n{nome}:")
    for faixa, qtd in contagem.items():
        print(f"  {faixa}: {qtd} detecções")

