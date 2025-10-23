import cv2
import pandas as pd
import numpy as np
import os
import random

# ==============================
# CONFIGURAÇÕES
# ==============================
ref_folder = "dados_ecos"        # Pasta com os CSVs de referência
video_path = "Lethicia.mp4"      # Vídeo de entrada
output_folder = "deteccoes_ecos" # Pasta para salvar os CSVs de saída
os.makedirs(output_folder, exist_ok=True)

# ==============================
# FUNÇÃO PARA OBTER MÉDIA DE COR DE CADA CSV
# ==============================
def get_mean_color_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    if all(col in df.columns for col in ["B", "G", "R"]):
        pixels = df[["B", "G", "R"]].to_numpy()
    elif all(col in df.columns for col in ["media_B", "media_G", "media_R"]):
        pixels = df[["media_B", "media_G", "media_R"]].to_numpy()
    else:
        raise ValueError(f"CSV inválido: {csv_file}")
    return np.mean(pixels, axis=0)

# ==============================
# CARREGAR TODAS AS REFERÊNCIAS
# ==============================
ref_colors = {}
for file in os.listdir(ref_folder):
    if file.endswith(".csv"):
        path = os.path.join(ref_folder, file)
        name = os.path.splitext(file)[0]
        mean_bgr = get_mean_color_from_csv(path)
        mean_bgr_int = np.uint8([[mean_bgr]])
        mean_hsv = cv2.cvtColor(mean_bgr_int, cv2.COLOR_BGR2HSV)[0][0].astype(int)
        ref_colors[name] = mean_hsv

print(f"🔍 {len(ref_colors)} referências carregadas:")
for k, v in ref_colors.items():
    print(f"  - {k}: HSV={v}")

# ==============================
# PARÂMETROS DE DETECÇÃO
# ==============================
tol_h, tol_s, tol_v = 12, 80, 80
color_map = {
    name: (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))
    for name in ref_colors.keys()
}

# ==============================
# ABRIR VÍDEO
# ==============================
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Erro ao abrir o vídeo.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)

# ==============================
# LOOP PRINCIPAL
# ==============================
frame_num = 0

# Dicionário de DataFrames para cada referência
deteccoes = {name: pd.DataFrame(columns=["frame", "tempo_s", "x", "y", "w", "h", "area"])
             for name in ref_colors.keys()}

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1
    tempo_s = frame_num / fps

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    for ref_name, ref_hsv in ref_colors.items():
        lower_hsv = np.array([
            max(0, ref_hsv[0] - tol_h),
            max(0, ref_hsv[1] - tol_s),
            max(0, ref_hsv[2] - tol_v)
        ], dtype=np.uint8)
        upper_hsv = np.array([
            min(180, ref_hsv[0] + tol_h),
            min(255, ref_hsv[1] + tol_s),
            min(255, ref_hsv[2] + tol_v)
        ], dtype=np.uint8)

        mask = cv2.inRange(hsv, lower_hsv, upper_hsv)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if 2000 < area < 50000:  # ajuste conforme necessário
                x, y, w, h = cv2.boundingRect(cnt)
                color = color_map[ref_name]
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, ref_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                deteccoes[ref_name].loc[len(deteccoes[ref_name])] = [frame_num, tempo_s, x, y, w, h, area]

    # Exibe o resultado
    cv2.imshow("Detecção ECO", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC para sair
        break

# ==============================
# FINALIZAÇÃO
# ==============================
cap.release()
cv2.destroyAllWindows()

# Salva um CSV por referência
for ref_name, df in deteccoes.items():
    csv_path = os.path.join(output_folder, f"deteccao_{ref_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ CSV salvo: {csv_path} ({len(df)} detecções)")

print("\n🎯 Finalizado! Todos os CSVs foram salvos na pasta:", output_folder)
