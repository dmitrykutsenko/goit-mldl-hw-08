from pathlib import Path
import matplotlib.pyplot as plt
import cv2

img_dir = Path("data/indoor/train/images")
sample_img = next(img_dir.glob("*.jpg"))

img = cv2.imread(str(sample_img))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis("off")

# 6. Навчання моделі (YOLOv8/YOLOv9)
# 6.1. Запуск тренування

from ultralytics import YOLO

model = YOLO("yolov8s.pt")  # або "yolov9s.pt", якщо хочеш v9

model.train(
    data="configs/indoor.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,          # GPU
    workers=4,
    project="experiments",
    name="yolo8s_indoor",
    patience=10,       # early stopping
    lr0=0.01,
    optimizer="SGD",   # або "AdamW"
)

# 7. Оцінка та аналіз результатів
#
# Після тренування Ultralytics автоматично:
#    - Зберігає ваги (best.pt, last.pt).
#    - Генерує графіки (results.png, metrics.csv).
# 7.1. Базова оцінка (mAP, precision, recall)

from ultralytics import YOLO

model = YOLO("experiments/yolo8s_indoor/weights/best.pt")
metrics = model.val(data="configs/indoor.yaml")
print(metrics)  # тут будуть mAP@0.5, mAP@0.5:0.95, precision, recall


# 7.3. Матриця помилок

#Для матриці помилок по класах (на рівні класифікації детектованих боксів):
#    Збираємо предикти на валідації:

preds = model.predict(
    data="configs/indoor.yaml",
    split="val",
    imgsz=640,
    conf=0.25
)
