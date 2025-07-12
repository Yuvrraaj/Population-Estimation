import os
import math
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Load YOLOv8 model
model = YOLO("runs/detect/train/weights/best.pt")  # update with your model path

class PopulationEstimator:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Building Detection + DSM/DTM Floor Estimation")

        self.image_label = tk.Label(root)
        self.image_label.pack()

        self.stats_text = tk.StringVar()
        tk.Label(root, textvariable=self.stats_text, font=("Arial", 12)).pack(pady=5)

        self.population_label = tk.Label(root, text="Estimated Population: --", font=("Arial", 14, "bold"))
        self.population_label.pack(pady=5)

        tk.Button(root, text="Select Image", command=self.analyze_image).pack(pady=10)

        self.people_per_floor = 4.23

    def analyze_image(self):
        file_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image Files", "*.png;*.jpg")])
        if not file_path:
            return

        image = cv2.imread(file_path)
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_pil = Image.fromarray(original_rgb)
        original_pil.thumbnail((500, 500))
        self.display_image(original_pil)

        try:
            dsm = rasterio.open("DSM.tif")
            dtm = rasterio.open("DTM.tif")
        except:
            dsm = dtm = None

        results = model.predict(source=file_path, save=False, save_txt=False, imgsz=640)
        result = results[0]

        residential_count = 0
        non_residential_count = 0
        estimated_population = 0

        for box in result.boxes:
            cls = int(box.cls[0])
            if cls != 0 and cls != 1:
                continue

            x_center, y_center, w, h = box.xywh[0]
            x1 = int(x_center - w / 2)
            y1 = int(y_center - h / 2)
            x2 = int(x_center + w / 2)
            y2 = int(y_center + h / 2)

            if cls == 0:  # Residential
                residential_count += 1

                area = w * h
                multiplier = 1 if area < 800 else 2 if area < 1500 else 3

                if dsm and dtm:
                    height = self.get_building_height(dsm, dtm, (x1, y1, x2, y2))
                    floors = max(1, round(height / 3))
                else:
                    floors = 1

                population = self.people_per_floor * multiplier * floors
                estimated_population += population
            else:
                non_residential_count += 1

        self.population_label.config(text=f"Estimated Population: {math.ceil(estimated_population)}")
        self.stats_text.set(f"Residential: {residential_count} | Non-Residential: {non_residential_count}")

    def get_building_height(self, dsm, dtm, bbox):
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        try:
            dsm_window = dsm.read(1, window=Window(x1, y1, w, h))
            dtm_window = dtm.read(1, window=Window(x1, y1, w, h))
            height_map = dsm_window - dtm_window
            height_map = height_map[height_map > 0]  # remove invalid
            return float(np.mean(height_map)) if height_map.size > 0 else 3.0
        except:
            return 3.0

    def display_image(self, pil_image):
        tk_image = ImageTk.PhotoImage(pil_image)
        self.image_label.configure(image=tk_image)
        self.image_label.image = tk_image

if __name__ == '__main__':
    root = tk.Tk()
    app = PopulationEstimator(root)
    root.mainloop()
