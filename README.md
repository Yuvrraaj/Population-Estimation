
# **🛰️ Interpretable AI for Census-Independent Population Estimation**

This project implements a full **GeoAI pipeline** to estimate population using **only satellite imagery, DSM/DTM elevation models, and deep learning**, without relying on census datasets.

It combines **YOLOv8 segmentation**, **DSM–DTM height modeling**, **coordinate-aware tiling**, and a **Tkinter-based GUI** to deliver interpretable, high-resolution demographic analytics.

---


# **📖 About the Project**

This project delivers a complete **population estimation system** driven by:

* High-resolution **satellite imagery**
* Building segmentation via **YOLOv8**
* Height extraction from **DSM (Digital Surface Model)** and **DTM (Digital Terrain Model)**
* **Coordinate-aware tile generation** for geospatial accuracy
* Rule-based **multi-floor population modeling**
* A **desktop GUI** for non-technical users

The system supports **large-scale raster data (45GB+)**, includes **automatic reprojection**, and produces fully **interpretable geospatial outputs**.

> **Note:** This project is part of ongoing development at NESAC (Department of Space, India).

---

# **✨ Key Features**

### ✔ **Coordinate-Aware Tiling (Area-Based, Not Pixel-Based)**

Tiles are generated using **real-world area (m²)** ensuring perfect alignment between imagery, DSM, and DTM.

### ✔ **YOLOv8 Building Detection**

Detects & segments:

* Residential buildings
* Non-residential buildings

Outputs bounding boxes and masks.

### ✔ **DSM–DTM Height Extraction**

* Computes **true building height**
* Converts height → **floor estimate**
* Removes vegetation/water false positives using VARI & NDWI masks

### ✔ **Population Estimation**

Population per building is calculated from:

* Bounding box size
* Floor count
* Area-based multipliers
* Household density constants

### ✔ **Tkinter-Based GUI**

Includes:

* File upload system
* Tile generation panel
* YOLO detection runner
* DSM-DTM alignment checker
* Visualization tools
* CSV export module

---

# **🏗 System Architecture**


The pipeline includes:

1. **Satellite Raster Preprocessing**
2. **Vegetation/Water Masking (VARI, NDWI)**
3. **Coordinate-Aware Area-Based Tiling**
4. **Building Detection (YOLOv8)**
5. **DSM–DTM Height Extraction (Reprojected & Aligned)**
6. **Floor Estimation**
7. **Population Modeling**
8. **Visualization & Export**

---

# **🔬 Methodology**

## **1️⃣ Vegetation & Water Masking**

Computed using spectral indices:

```
VARI = (G - R) / (G + R - B)
NDWI = (G - NIR) / (G + NIR)
```

Used to remove vegetation & water pixels before detecting buildings.

---

## **2️⃣ Building Detection with YOLOv8**

YOLO returns for each structure:

* **Bounding box**
* **Segmentation mask**
* **Confidence score**
* **Class (residential / non-residential)**

---

## **3️⃣ Elevation-Based Floor Estimation**

Height = DSM − DTM

```
floor_count = round(height / 3.0)
```

Where 3 meters = standard floor height.

Patch-based sampling rejects outliers and invalid elevation values.

---

## **4️⃣ Population Modeling**

Based on building footprint categories:

| Size Category | Area Threshold | Weight |
| ------------- | -------------- | ------ |
| Small         | <150 m²        | 1×     |
| Medium        | 150–250 m²     | 2×     |
| Large         | >250 m²        | 3×     |

Final estimation:

```
population = base_population × size_multiplier × floors
```

---

# **🛠 Tech Stack**

### **Core Languages**

* Python 3.x

### **Deep Learning**

* YOLOv8 (Ultralytics)

### **Remote Sensing & GIS**

* Rasterio
* GDAL
* Shapely
* GeoPandas

### **Image Processing**

* OpenCV
* Pillow
* NumPy
* SciPy

### **Visualization**

* Matplotlib
* Seaborn

### **Frontend**

* Tkinter GUI

---

# **🔁 Pipeline Steps**

## **1. Load Inputs**

* Satellite TIF
* DSM
* DTM

## **2. Generate Area-Based Tiles**

* Uses CRS transform
* Ensures DSM/DTM alignment
* Maintains pixel grid consistency

## **3. Run YOLOv8 Detection**

* Outputs bounding boxes + masks

## **4. Align DSM/DTM via Reprojection**

* Ensures identical spatial grid
* No pixel drift
* Perfect CRS matching

## **5. Extract Height**

* Windowed raster sampling
* Median height
* Floor estimation

## **6. Compute Population**

* Size-based coefficients
* Floors × multiplier

## **7. Visualize & Export**

* GUI preview
* Charts
* CSV metadata

---

# **🖥 GUI Features**

> *(Insert GUI screenshots here)*
> Example: `![GUI](images/gui.png)`

The GUI provides:

* **File selection modules**
* **Coordinate-aware tile generation**
* **Detection preview**
* **Height visualization**
* **Population summary charts**
* **CSV export**

---

# **📤 Output Format**

Generated CSV includes:

```
tile_path
bbox_x, bbox_y
world_x, world_y
width, height
class_id
confidence
estimated_floors
building_area
population_estimate
tile_crs
transform
```

Compatible with **QGIS**, **ArcGIS**, **PostGIS**, and research workflows.

---

# **📈 Results**

The system achieves:

* Processing of **45GB+ raster datasets**
* Accurate segmentation of **residential buildings**
* Elevation-based floor estimation (DSM–DTM)
* High-quality population estimates
* **35% reduction in manual analysis time** for research teams

---

# **📁 Project Structure**

```
├── main.py                # Tkinter GUI application
├── best.pt                # YOLOv8 model
├── utils/                 # Preprocessing + helper functions
├── data/                  # Sample rasters/tiles
├── exports/               # Output CSVs/plots
├── README.md
└── LICENSE
```

---

# **🚀 Usage**

### **1. Clone the repository**

```bash
git clone https://github.com/Yuvrraaj/Population-Estimation
cd Population-Estimation
```

### **2. Install dependencies**

```bash
pip install -r requirements.txt
```

### **3. Run the GUI**

```bash
python main.py
```

### **4. Upload inputs**

* Satellite TIF
* DSM
* DTM

### **5. Run analysis**

* Generate tiles
* Detect buildings
* Estimate population
* Export CSV

---

# **🙏 Acknowledgements**

This project was developed as part of ongoing work at:

**NESAC – North Eastern Space Applications Centre (Department of Space, Govt. of India)**

Mentored by: **Mr. Santanu Das**, Urban & Regional Planning Division.

---

# **🛡 License**

This project is released under the **MIT License**.
See the `LICENSE` file for details.

---


Just say **"Add badges"**, **"Add diagrams"**, or **"Add demo section"**.
