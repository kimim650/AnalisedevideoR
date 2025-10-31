from ultralytics import YOLO
import cv2
import os

# ======================
# CONFIGURAÇÕES
# ======================
video_path = r"C:\Users\Jacó\Downloads\Energia\Lethicia.mp4"
output_dir = r"C:\Users\Jacó\Downloads\Energia\deteccoes"  # pasta para salvar imagens
os.makedirs(output_dir, exist_ok=True)

# Verifica se o vídeo existe
if not os.path.exists(video_path):
    raise FileNotFoundError(f"❌ Arquivo de vídeo não encontrado: {video_path}")

# Carrega o modelo YOLO pré-treinado
model = YOLO("yolov8n.pt")

# Abre o vídeo
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"❌ Erro ao abrir o vídeo: {video_path}")

print("✅ Vídeo aberto com sucesso! Pressione ESC para sair.")

skip_frames = 10  # quantos frames pular (1 = todos)
frame_num = 0
save_count = 0

# ======================
# LOOP DE DETECÇÃO
# ======================
while True:
    ret, frame = cap.read()
    if not ret:
        print("🎬 Fim do vídeo ou erro ao ler frame.")
        break

    frame_num += 1
    if frame_num % skip_frames != 0:
        continue

    # Realiza a inferência
    results = model(frame, verbose=False)[0]

    detected = False  # flag para saber se houve detecção
    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < 0.3:
            continue

        detected = True
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0])
        label = model.names[cls]

        # Desenha caixa e texto
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Se houve detecção, salva o frame como .jpg
    if detected:
        save_path = os.path.join(output_dir, f"frame_{frame_num:06d}.jpg")
        cv2.imwrite(save_path, frame)
        save_count += 1

    # Mostra o frame
    cv2.imshow("YOLOv8 - Detecção em Vídeo", frame)

    # Sai com ESC
    if cv2.waitKey(1) & 0xFF == 27:
        print("🛑 Execução interrompida pelo usuário.")
        break

# Libera recursos
cap.release()
cv2.destroyAllWindows()
print(f"✅ Finalizado com sucesso! {save_count} imagens salvas em: {output_dir}")

