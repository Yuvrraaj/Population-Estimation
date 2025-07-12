

# Satellite-Based Population Estimation using Deep Learning & Elevation Modeling

This project leverages **remote sensing**, **deep learning**, and **digital elevation data** to estimate population density from high-resolution satellite imagery. The system is designed to be interpretable, modular, and capable of working independently of traditional census data.

## Overview

The project involves:
- Preprocessing of satellite tiles to mask vegetation and water bodies using **VARI** and **NDWI** indices.
- Detection of residential and non-residential buildings using a **YOLOv8 segmentation model**.
- Estimation of building heights and number of floors using **DSM (Digital Surface Model)** and **DTM (Digital Terrain Model)** elevation data.
- Calculation of population estimates based on building area and vertical extent.
- A user-friendly **Tkinter-based GUI** for uploading inputs and visualizing results.

---

## Features

-  **Automated Masking**: Filters out vegetation and water areas before detection.
-  **Object Detection**: Segments buildings from cleaned satellite tiles using YOLOv8.
-  **Elevation-Aware Estimation**: Uses DSM–DTM elevation difference to estimate number of floors.
-  **Population Calculation**: Applies rule-based logic on building size and height.
-  **Interactive GUI**: Desktop-based application for easy execution and visualization.

---

## 🛠 Tech Stack

- Python 3.x
- OpenCV, NumPy, Pillow
- Ultralytics YOLOv8
- Rasterio, GeoPandas, Shapely
- Tkinter (for GUI)
- Roboflow (for annotation)

---



## 🧪 How to Run

1. **Clone the repo**
   bash
   git clone https://github.com/YuvrajJha-01/Population-Estimation-AI
   cd Population-Estimation-AI


2. **Install dependencies**

   bash
   pip install -r requirements.txt
   

3. **Run the GUI**

   bash
   python gui/main_gui.py
   

4. **Upload satellite tile**, optional DSM/DTM files, and view population output.

---

## 📸 Sample Output

> Add screenshots here showing:
>
> * Input image
> * Building detection result
> * Final population estimate overlay

---

## 📝 Acknowledgements

* **NESAC – North Eastern Space Applications Centre**, Department of Space, Govt. of India
* Mentored by **Mr. Santanu Das**, Urban & Regional Planning Division


---

## 🛡 License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.


