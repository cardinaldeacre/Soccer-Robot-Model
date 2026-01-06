from ultralytics import YOLO

model = YOLO("model/best_float32.tflite")
model.predict(source=0, imgsz=640, conf=0.25, show=True) # webcam

print("done")