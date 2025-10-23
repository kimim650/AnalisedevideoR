import cv2
import pandas as pd
import numpy as np
import os

# ==============================
# CONFIGURAÇÕES
# ==============================
csv_ref = "dadosECO.csv"        # CSV com os pixels de referência
csv_out = "deteccaoECO.csv"     # CSV para salvar as detecções
video_path = "Lethicia.mp4"     # vídeo de entrada

# Remove CSV anterior se existir
if os.path.exists(csv_out):
    os.remove(csv_out)

# ==============================
# CARREGAR CSV DE REFERÊNCIA
# ==============================
df_ref = pd.read_csv(csv_ref)

# Detectar formato do CSV
if all(col in df_ref.columns for col in ["B", "G", "R"]):
    pixels_ref = df_ref[["B", "G", "R"]].to_numpy()
elif all(col in df_ref.columns for col in ["media_B", "media_G", "media_R"]):
    pixels_ref = df_ref[["media_B", "media_G", "media_R"]].to_numpy()
else:
    raise ValueError("O CSV deve conter colunas B,G,R ou media_B,media_G,media_R.")

# Calcula média da cor de referência (BGR)
media_ref_bgr = np.mean(pixels_ref, axis=0)

# Converte para inteiro antes de converter para HSV (evita overflow)
media_ref_bgr_int = np.uint8([[media_ref_bgr]])
media_ref_hsv = cv2.cvtColor(media_ref_bgr_int, cv2.COLOR_BGR2HSV)[0][0].astype(int)

print("Média de cor (BGR):", media_ref_bgr)
print("Média de cor (HSV):", media_ref_hsv)

# ==============================
# INTERVALO DE COR (ajuste sensibilidade)
# ==============================
tol_h, tol_s, tol_v = 12, 80, 80

# Evita overflow/underflow convertendo para int e limitando faixa
lower_hsv = np.array([
    max(0, int(media_ref_hsv[0]) - tol_h),
    max(0, int(media_ref_hsv[1]) - tol_s),
    max(0, int(media_ref_hsv[2]) - tol_v)
], dtype=np.uint8)

upper_hsv = np.array([
    min(180, int(media_ref_hsv[0]) + tol_h),
    min(255, int(media_ref_hsv[1]) + tol_s),
    min(255, int(media_ref_hsv[2]) + tol_v)
], dtype=np.uint8)

# ==============================
# ABRIR VÍDEO
# ==============================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Erro ao abrir o vídeo.")
    exit()

# ==============================
# CRIAR DATAFRAME DE SAÍDA
# ==============================
df_out = pd.DataFrame(columns=["frame", "x", "y", "w", "h", "area"])

# ==============================
# LOOP PRINCIPAL DE DETECÇÃO
# ==============================
frame_num = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    # Converte frame para HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Cria máscara com base na cor de referência
    mask = cv2.inRange(hsv, lower_hsv, upper_hsv)

    # Limpeza de ruído
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # Detectar contornos
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if 10000 < area < 100000:  # ignora ruído pequeno
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            df_out.loc[len(df_out)] = [frame_num, x, y, w, h, area]

    # Exibe o resultado
    cv2.imshow("Detecção Alheio", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

# ==============================
# FINALIZAÇÃO
# ==============================
cap.release()
cv2.destroyAllWindows()

# Salvar CSV
df_out.to_csv(csv_out, index=False)
print(f"✅ Detecções salvas em: {csv_out}")
