from ultralytics import YOLO
import serial
import time
import cv2

SERIAL_PORT = '/dev/ttyUSB0' # sesauikan
BAUD_RATE = 115200
MODEL_PATH = "model/yolo8x320.tflite"

print("loading model...")

# tes serial esp32
try:
    esp32 = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    time.sleep(2) # tunggu esp 32 resett
    print(f"ESP32 terhubung di {SERIAL_PORT}")
    
except Exception as e:
    print(f"Gagal terhubung ke ESP32: {e}")
    print("Mode: vision only, tanpa kirim data")
    esp32 = None
    
# load model
print(f"Loading model from {MODEL_PATH}...")
try:
    model = YOLO(MODEL_PATH)
    print("Model loaded.")
except Exception as e:
    print(f"Gagal load model: {e}")
    exit()
    
# buka kamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

if not cap.isOpened():
    print("Gagal membuka kamera")
    exit()
    
print("Starting video stream...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Gagal membaca frame dari kamera")
        break
    
    # monitoring fps
    start_time = time.time()    
    results = model.predict(source=frame, imgsz=320, verbose=False, task="detect", conf=0.25)
    
    ball_found = False
    x_center = -1
    
    result = results[0] # ambil hasil frame pertama
    
    for box in result.boxes:
        # ambil koordinat
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy() # convert ke numpy array
        
        # Hitung titik tengah X
        x_center = int((x1 + x2) / 2)
        y_center = int((y1 + y2) / 2)
        
        # gambar visualisasi (debug monitor)
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.circle(frame, (x_center, y_center), 5, (0, 0, 255), -1)
        
        # kita anggap bola ketemu
        ball_found = True        
        # jika cuma mau 1 bola, langsung break
        break 

    # kirim data ke esp32
    if ball_found and esp32 is not None:
        # format: "X:160\n"
        data_to_send = f"X:{x_center}\n"
        esp32.write(data_to_send.encode('utf-8'))
        print(f"Sent: {data_to_send.strip()}") # Uncomment buat debug

    # hitung FPS
    fps = 1.0 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    # tampilkan di layar
    cv2.imshow("Robot Vision", frame)
    
    # tombol Q untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
if esp32:
    esp32.close()
print("Yeay, udahh!")