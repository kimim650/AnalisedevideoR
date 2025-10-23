import cv2
import numpy as np
import pandas as pd
import os

# ==============================
# CONFIGURAÇÕES INICIAIS
# ==============================
# Lista de imagens que você quer processar
image_paths = ["BOSS.jpeg",
               "Eco_lethicia.jpeg",
               "ECOCORROMPIDO.jpeg",
               "maoBoss.jpg",
               "MIA.jpg",
               "VORTEX.jpg",
]

# Pasta onde os CSVs serão salvos
output_folder = "dados_ecos"
os.makedirs(output_folder, exist_ok=True)

# ==============================
# PROCESSAMENTO DE CADA IMAGEM
# ==============================
for image_path in image_paths:
    # Verifica se o arquivo existe
    if not os.path.exists(image_path):
        print(f"⚠️ Arquivo não encontrado: {image_path}")
        continue

    # Carrega imagem
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erro ao carregar a imagem: {image_path}")
        continue

    print(f"\n🖼️ Processando imagem: {image_path}")

    # ==============================
    # SELEÇÃO DE ROI
    # ==============================
    roi = cv2.selectROI(f"Selecione a ROI em {os.path.basename(image_path)} (ENTER para confirmar)",
                        img, showCrosshair=True, fromCenter=False)
    x, y, w, h = roi

    # Se o usuário não selecionar nada
    if w == 0 or h == 0:
        print("ROI vazia — imagem ignorada.")
        continue

    roi_img = img[y:y+h, x:x+w]

    # ==============================
    # EXTRAÇÃO DE PIXELS
    # ==============================
    pixels_data = []

    for i in range(h):
        for j in range(w):
            b, g, r = roi_img[i, j]
            pixels_data.append({
                "x": x + j,
                "y": y + i,
                "B": int(b),
                "G": int(g),
                "R": int(r)
            })

    # ==============================
    # SALVA EM CSV
    # ==============================
    csv_name = os.path.splitext(os.path.basename(image_path))[0] + ".csv"
    csv_path = os.path.join(output_folder, csv_name)

    df = pd.DataFrame(pixels_data)
    df.to_csv(csv_path, index=False)
    print(f"✅ Arquivo salvo com {len(df)} pixels em '{csv_path}'")

    # Mostra ROI para conferência
    cv2.imshow("ROI Selecionada", roi_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

print("\n🎯 Finalizado! Todos os CSVs foram salvos na pasta:", output_folder)
