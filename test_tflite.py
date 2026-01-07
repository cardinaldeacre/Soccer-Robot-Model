from ultralytics import YOLO

model = YOLO("model/yolo8x320.tflite")
model.predict(source=0, imgsz=320, conf=0.25, show=True) # webcam

print("done")