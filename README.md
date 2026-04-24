# Interpretable AI for Census-Independent Population Estimation
### Built at NESAC · North Eastern Space Applications Centre, Department of Space, Govt. of India

> Estimate population from satellite imagery alone — no census, no ground surveys, no assumptions about existing data.

---

## The Problem

Census data is expensive, infrequent, and often unavailable for regions that need demographic insight the most — conflict zones, rapidly developing urban fringes, remote northeast Indian terrain. Existing population models either require census ground truth to calibrate against, or rely on coarse global datasets with poor local resolution.

This project asks: **can you derive population estimates purely from what's visible from space?**

---

## What This System Does

A complete GeoAI pipeline that takes three raster inputs — a high-resolution satellite image, a Digital Surface Model (DSM), and a Digital Terrain Model (DTM) — and produces building-level population estimates with full coordinate metadata. No census dependency at any stage.

```
Satellite TIF  ──┐
DSM (surface)  ──┼──▶  Pipeline  ──▶  Per-building population CSV
DTM (terrain)  ──┘                    + Visualizations + GUI
```

The output is a coordinate-aware CSV compatible with QGIS, ArcGIS, and PostGIS — each row is a detected building with its world coordinates, floor count, footprint area, and population estimate.

---

## How It Works

### Stage 1 — Spectral Masking
Before any detection runs, non-building pixels are removed using two spectral indices computed directly from satellite band values:

```
VARI  =  (Green − Red) / (Green + Red − Blue)      → masks vegetation
NDWI  =  (Green − NIR) / (Green + NIR)             → masks water bodies
```

This is critical. Running YOLO directly on raw imagery produces false positives on dense tree canopy and riverbanks — vegetation rooftops are visually similar to building rooftops at many resolutions. Masking first means the detector only ever sees plausible building regions.

---

### Stage 2 — Coordinate-Aware Area-Based Tiling

Large satellite rasters (this system handles **45GB+**) cannot be fed directly into a detector. They need to be tiled. The key engineering decision here: tiles are defined in **real-world area (m²)**, not pixels.

Most tiling systems slice by pixel count. This breaks when:
- The satellite image, DSM, and DTM have different native resolutions or CRS
- You need consistent physical coverage per tile regardless of sensor GSD

Area-based tiling uses the raster's affine transform and CRS to compute tile boundaries in projected coordinates first, then derives pixel windows from those. The result: every tile covers exactly the same ground area, and DSM/DTM tiles are guaranteed to be spatially aligned to the satellite tile without pixel drift — even after reprojection.

---

### Stage 3 — Building Detection via YOLOv8

YOLOv8 segmentation runs on each masked tile and returns per-structure:
- Bounding box (pixel coordinates)
- Segmentation mask
- Confidence score
- Class: **residential** vs **non-residential**

The residential/non-residential classification matters for population modeling — offices, warehouses, and industrial buildings are detected and excluded from population counts, not lumped in.

---

### Stage 4 — DSM−DTM Height Extraction

True building height is computed by differencing the two elevation models:

```
building_height = DSM − DTM
```

DSM captures the top of everything (buildings, trees, terrain). DTM captures bare ground. The difference isolates vertical structure height.

For each detected building bounding box, height is sampled from the corresponding DSM−DTM patch using **windowed raster reads with median aggregation** — not a single pixel lookup. Median is used instead of mean to reject outliers from:
- Vegetation canopy bleed at building edges
- NoData / invalid elevation pixels at tile boundaries
- Sensor artifacts in low-quality DSM regions

Floor count is then estimated as:

```
floor_count = round(building_height / 3.0)
```

3.0 metres is the standard floor-to-floor height used in Indian residential construction. Minimum floor count is clamped to 1.

---

### Stage 5 — Population Modeling

Population per building is estimated from three factors: footprint area (from bounding box), floor count (from elevation), and a household density constant.

Building footprint is categorised by area:

| Category | Threshold  | Size Multiplier |
|----------|------------|-----------------|
| Small    | < 150 m²   | 1×              |
| Medium   | 150–250 m² | 2×              |
| Large    | > 250 m²   | 3×              |

Final estimate:

```
population = base_occupancy × size_multiplier × floor_count
```

This is a rule-based model, not a learned one — intentionally. Rule-based models are interpretable, auditable, and don't require labelled population data to train. Every estimate can be traced back to a measurable satellite observation.

---

### Stage 6 — Output & Export

Each detected building produces one row in the output CSV:

```
tile_path, bbox_x, bbox_y, world_x, world_y,
width_m, height_m, class_id, confidence,
estimated_floors, building_area_m2,
population_estimate, tile_crs, affine_transform
```

World coordinates (not pixel coordinates) are stored — so outputs load directly into GIS tools without any additional georeferencing step.

---

## The GUI

The entire pipeline is wrapped in a Tkinter desktop application so that domain scientists — remote sensing specialists, urban planners — can run analyses without writing code.

Panels:
- **File loader** — satellite TIF, DSM, DTM selection with CRS validation
- **Tile generator** — area-based tiling with preview
- **Detection runner** — YOLO inference with confidence threshold control
- **Height visualiser** — DSM−DTM overlay per tile
- **Population summary** — per-class charts and aggregate estimates
- **CSV export** — coordinate-aware output with full metadata

The GUI was the difference between a research prototype and a system actually used by scientists. 12+ researchers across 8 projects at NESAC now use it as part of their standard workflow.

---

## Results

| Metric | Value |
|--------|-------|
| Raster data processed | 45 GB+ |
| Manual analysis time reduction | 35% |
| Active researchers using the system | 12+ |
| Research projects deployed across | 8 |
| Population modeling | Census-independent, fully interpretable |

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Deep Learning | YOLOv8 (Ultralytics) |
| Geospatial / GIS | Rasterio, GDAL, GeoPandas, Shapely |
| Image Processing | OpenCV, Pillow, NumPy, SciPy |
| Visualisation | Matplotlib, Seaborn |
| GUI | Tkinter |
| Language | Python 3.x |

---

## Project Structure

```
├── app.py               # Main GUI application entry point
├── main.py              # Pipeline orchestrator
├── best.pt              # YOLOv8 segmentation model weights
├── dhm2.py / dhm3.py    # DSM-DTM height extraction modules
├── image_to_tiles.py    # Coordinate-aware area-based tiling
├── utils/               # Preprocessing helpers, masking, reprojection
├── data/                # Sample raster inputs
├── exports/             # Output CSVs and visualisation plots
└── requirements.txt
```

---

## Quickstart

```bash
git clone https://github.com/Yuvrraaj/Population-Estimation
cd Population-Estimation
pip install -r requirements.txt
python app.py
```

Load your satellite TIF, DSM, and DTM via the GUI. Generate tiles, run detection, export CSV.

---

## Acknowledgements

Developed at **NESAC — North Eastern Space Applications Centre**, Department of Space, Govt. of India.  
Mentored by **Mr. Santanu Das**, Urban & Regional Planning Division.

---

## License

MIT License. See `LICENSE` for details.
