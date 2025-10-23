import cv2
import numpy as np
import pandas as pd
import os

# ==============================
# CONFIGURAÇÕES INICIAIS
# ==============================
image_path = ("ECOCORROMPIDO.jpeg"
              )
csv_path = "dadosECO.csv"

# Remove CSV anterior se existir
if os.path.exists(csv_path):
    os.remove(csv_path)

# Carrega imagem
img = cv2.imread(image_path)
if img is None:
    print("Erro ao carregar a imagem.")
    exit()

# ==============================
# SELEÇÃO DE ROI
# ==============================
roi = cv2.selectROI("Selecione a ROI (pressione ENTER para confirmar)", img, showCrosshair=True, fromCenter=False)
x, y, w, h = roi
roi_img = img[y:y+h, x:x+w]

# ==============================
# EXTRAÇÃO DE PIXELS
# ==============================
# Cria listas com coordenadas e valores de cor
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
df = pd.DataFrame(pixels_data)
df.to_csv(csv_path, index=False)
print(f"Arquivo salvo com {len(df)} pixels em '{csv_path}'")

# Mostra ROI para conferência
cv2.imshow("ROI Selecionada", roi_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
