from ultralytics import YOLO
import torch


def main():
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    model = YOLO("yolov9m.pt")

    model.train(
        data="configs/indoor.yaml",
        epochs=50,
        imgsz=512,          # зменшено для MX550
        batch=2,            # MX550 максимум 2
        device=0,
        workers=0,          # критично для Windows
        cache=False,        # не вистачає RAM
        amp=False,          # AMP падає на MX550
        half=True,          # FP16 економить VRAM
        optimizer="AdamW",
        lr0=0.002,
        patience=10,
        project="experiments",
        name="yolo9m_indoor_mx550",
        pretrained=True
    )


if __name__ == "__main__":
    main()
