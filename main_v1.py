from pathlib import Path
import matplotlib.pyplot as plt
import cv2

img_dir = Path("data/indoor/train/images")
sample_img = next(img_dir.glob("*.jpg"))

img = cv2.imread(str(sample_img))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.axis("off")

