import tkinter as tk
from tkinter import filedialog, messagebox, ttk, Canvas
from PIL import Image, ImageTk
import cv2
import os
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from ultralytics import YOLO
import threading
#from rasterio.sample import sample
import rasterio
import time
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio import features
from rasterio.features import shapes, rasterize, geometry_mask

# ==== Load YOLO Model ====
model = YOLO('runs/detect/train/weights/best.pt')  # Update this path

# ==== Constants ====
colors = {0: (255, 0, 255), 1: (0, 165, 255)}
class_names = {0: 'Residential', 1: 'Non-Residential'}

class ModernYOLOApp:
    def __init__(self, root):
        self.root = root
        self.people_per_building = tk.DoubleVar(value=4.23)
        self.file_path = None
        self.dsm_path = None
        self.dtm_path = None
        self.setup_window()
        self.create_styles()
        self.create_widgets()

    def setup_window(self):
        self.root.title("YOLO Building & Population Estimator")
        self.root.geometry("1200x900")
        self.root.configure(bg="#002B5B")
        self.root.resizable(True, True)
        self.center_window()

    def center_window(self):
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.root.winfo_screenheight() // 2) - (900 // 2)
        self.root.geometry(f"1200x900+{x}+{y}")

    def create_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Modern.TButton', 
                           background='#00d4aa', 
                           foreground='white', 
                           borderwidth=0, 
                           padding=(20, 15), 
                           font=('Segoe UI', 12, 'bold'))
        self.style.map('Modern.TButton', 
                      background=[('active', '#00b899'), ('pressed', '#009688')])
        
        # Special style for the analyze button to make it more prominent
        self.style.configure('Analyze.TButton', 
                           background='#ff4444', 
                           foreground='white', 
                           borderwidth=0, 
                           padding=(25, 20), 
                           font=('Segoe UI', 14, 'bold'))
        self.style.map('Analyze.TButton', 
                      background=[('active', '#ff6666'), ('pressed', '#cc3333')])
        
        self.style.configure('Small.TButton', 
                           background='#00d4aa', 
                           foreground='white', 
                           borderwidth=0, 
                           padding=(10, 8), 
                           font=('Segoe UI', 10, 'bold'))
        self.style.map('Small.TButton', 
                      background=[('active', '#00b899'), ('pressed', '#009688')])
        
        self.style.configure('Modern.Horizontal.TProgressbar', 
                           background='#00d4aa', 
                           troughcolor='#1e2328', 
                           borderwidth=0)

    def create_widgets(self):
        container = tk.Frame(self.root, bg="#3E8EDE")
        container.pack(fill=tk.BOTH, expand=True)

        self.canvas = Canvas(container, bg="#3E8EDE", highlightthickness=0)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = ttk.Scrollbar(container, orient=tk.VERTICAL, command=self.canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.canvas.configure(yscrollcommand=scrollbar.set)
        self.canvas.bind('<Configure>', lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")))

        self.scrollable_frame = tk.Frame(self.canvas, bg="#3E8EDE")
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

        self.scrollable_frame.bind_all("<MouseWheel>", self._on_mousewheel)

        self.create_header(self.scrollable_frame)
        content_frame = tk.Frame(self.scrollable_frame, bg="#3E8EDE")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0), padx=10)

        left_panel = tk.Frame(content_frame, bg="#1a1f26", width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10), pady=10)
        left_panel.pack_propagate(False)
        self.create_control_panel(left_panel)

        right_container = tk.Frame(content_frame, bg="#3E8EDE")
        right_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))

        right_panel = tk.Frame(right_container, bg="#1a1f26", height=500)
        right_panel.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        self.create_image_panel(right_panel)

        chart_panel = tk.Frame(right_container, bg="#1a1f26", height=300)
        chart_panel.pack(fill=tk.X, pady=(10, 0))
        chart_panel.pack_propagate(False)
        self.create_chart_panel(chart_panel)

        self.create_status_bar(self.scrollable_frame)

    def _on_mousewheel(self, event):
        self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def create_header(self, parent):
        header = tk.Label(parent, text="Population Estimator", font=('Segoe UI', 28, 'bold'), fg="white", bg="#3E8EDE")
        header.pack(pady=10)
        sub = tk.Label(parent, text="Advanced Population Estimation with YOLO", font=('Segoe UI', 14), fg="#ffffff", bg="#3E8EDE")
        sub.pack(pady=0)

    def create_control_panel(self, parent):
        # Image file selection
        self.upload_btn = ttk.Button(parent, text="Choose Image File", style='Modern.TButton', command=self.choose_image)
        self.upload_btn.pack(fill=tk.X, pady=10, padx=10)

        self.file_info = tk.Label(parent, text="No file selected", fg="#8892b0", bg="#1a1f26", wraplength=250)
        self.file_info.pack(anchor=tk.W, padx=10)

        # Height Analysis File Upload Section
        height_frame = tk.Frame(parent, bg="#1a1f26")
        height_frame.pack(pady=15, padx=10, fill=tk.X)
        
        height_label = tk.Label(height_frame, text="Height Analysis Options:", fg="white", bg="#1a1f26", font=('Segoe UI', 11, 'bold'))
        height_label.pack(anchor=tk.W)
        
        # DSM Upload
        dsm_frame = tk.Frame(height_frame, bg="#1a1f26")
        dsm_frame.pack(fill=tk.X, pady=(5, 2))
        
        self.dsm_btn = ttk.Button(dsm_frame, text="Upload DSM File", style='Small.TButton', command=self.upload_dsm)
        self.dsm_btn.pack(side=tk.LEFT)
        
        self.dsm_info = tk.Label(dsm_frame, text="No DSM file", fg="#8892b0", bg="#1a1f26", font=('Segoe UI', 9))
        self.dsm_info.pack(side=tk.LEFT, padx=(10, 0))
        
        # DTM Upload
        dtm_frame = tk.Frame(height_frame, bg="#1a1f26")
        dtm_frame.pack(fill=tk.X, pady=(2, 5))
        
        self.dtm_btn = ttk.Button(dtm_frame, text="Upload DTM File", style='Small.TButton', command=self.upload_dtm)
        self.dtm_btn.pack(side=tk.LEFT)
        
        self.dtm_info = tk.Label(dtm_frame, text="No DTM file", fg="#8892b0", bg="#1a1f26", font=('Segoe UI', 9))
        self.dtm_info.pack(side=tk.LEFT, padx=(10, 0))
        
        # Info text
        info_text = tk.Label(
            height_frame, 
            text="Upload both DSM & DTM files for floor-based estimation\n(adds floor count to area-based count)", 
            fg="#8892b0", 
            bg="#1a1f26", 
            font=('Segoe UI', 9),
            wraplength=300,
            justify=tk.LEFT
        )
        info_text.pack(anchor=tk.W, pady=(5, 0))

        # People per building input
        input_frame = tk.Frame(parent, bg="#1a1f26")
        input_frame.pack(pady=10, padx=10, fill=tk.X)
        
        label = tk.Label(input_frame, text="People/Residential Building:", fg="white", bg="#1a1f26", font=('Segoe UI', 11))
        label.pack(anchor=tk.W)
        
        self.people_entry = tk.Entry(input_frame, textvariable=self.people_per_building, font=('Segoe UI', 11))
        self.people_entry.pack(fill=tk.X, pady=5)
        
        # MAIN ANALYZE BUTTON - Made more prominent
        analyze_frame = tk.Frame(parent, bg="#1a1f26")
        analyze_frame.pack(pady=20, padx=10, fill=tk.X)
        
        self.analyze_button = ttk.Button(
            analyze_frame, 
            text="🔍 START ANALYSIS", 
            style='Analyze.TButton', 
            command=self.start_analysis
        )
        self.analyze_button.pack(fill=tk.X)
        
        # Add a separator line
        separator = tk.Frame(parent, bg="#444444", height=2)
        separator.pack(fill=tk.X, pady=15, padx=10)

        # Statistics display
        stats_frame = tk.Frame(parent, bg="#1a1f26")
        stats_frame.pack(pady=10, padx=10, fill=tk.X)

        stats_title = tk.Label(stats_frame, text="Analysis Results:", fg="white", bg="#1a1f26", font=('Segoe UI', 12, 'bold'))
        stats_title.pack(anchor=tk.W, pady=(0, 10))

        self._add_stat(stats_frame, "Estimated Population:", 'pop_value', "#00d4aa")
        self._add_stat(stats_frame, "Area-based Population:", 'area_pop_value', "#ff6600")
        self._add_stat(stats_frame, "Floor-based Population:", 'floor_pop_value', "#9900ff")
        self._add_stat(stats_frame, "Residential Buildings:", 'residential_count', "#ff0088")
        self._add_stat(stats_frame, "Non-Residential Buildings:", 'non_residential_count', "#00a5ff")
        self._add_stat(stats_frame, "Total Detections:", 'total_count', "#ffffff")
        self._add_stat(stats_frame, "Estimation Method:", 'method_info', "#ffaa00")

    def _add_stat(self, parent, label_text, attr_name, color):
        frame = tk.Frame(parent, bg="#1a1f26")
        frame.pack(fill=tk.X, pady=2)
        label = tk.Label(frame, text=label_text, font=('Segoe UI', 10, 'bold'), fg="#bbbbbb", bg="#1a1f26")
        label.pack(side=tk.LEFT)
        value = tk.Label(frame, text="--", font=('Segoe UI', 12, 'bold'), fg=color, bg="#1a1f26")
        value.pack(side=tk.RIGHT)
        setattr(self, attr_name, value)

    def create_image_panel(self, parent):
        # Title for image panel
        title_frame = tk.Frame(parent, bg="#1a1f26")
        title_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        title_label = tk.Label(title_frame, text="Image Analysis Results", fg="white", bg="#1a1f26", font=('Segoe UI', 14, 'bold'))
        title_label.pack()

        # Image display area
        image_frame = tk.Frame(parent, bg="#1a1f26")
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        self.original_label = tk.Label(image_frame, bg="#0f1419", fg="#8892b0", text="Original Image\n(Select an image file)", font=('Segoe UI', 12))
        self.original_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))

        self.result_label = tk.Label(image_frame, bg="#0f1419", fg="#8892b0", text="Detection Results\n(Click START ANALYSIS)", font=('Segoe UI', 12))
        self.result_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(5, 0))

    def create_chart_panel(self, parent):
        # Title for chart panel
        title_frame = tk.Frame(parent, bg="#1a1f26")
        title_frame.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        title_label = tk.Label(title_frame, text="Statistical Analysis", fg="white", bg="#1a1f26", font=('Segoe UI', 14, 'bold'))
        title_label.pack()

        self.chart_container = tk.Frame(parent, bg="#2d3748")
        self.chart_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))
        
        self.chart_placeholder = tk.Label(self.chart_container, text="Charts will appear here after analysis", fg="#8892b0", bg="#0f1419", font=('Segoe UI', 12))
        self.chart_placeholder.pack(fill=tk.BOTH, expand=True)

    def create_status_bar(self, parent):
        status_frame = tk.Frame(parent, bg="#1a1f26")
        status_frame.pack(fill=tk.X, pady=10, padx=15)
        
        status_title = tk.Label(status_frame, text="Status:", fg="white", bg="#1a1f26", font=('Segoe UI', 10, 'bold'))
        status_title.pack(side=tk.LEFT)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select an image file and click START ANALYSIS")
        status_label = tk.Label(status_frame, textvariable=self.status_var, bg="#1a1f26", fg="#00d4aa", anchor=tk.W, font=('Segoe UI', 10))
        status_label.pack(side=tk.LEFT, padx=(10, 0))

    def upload_dsm(self):
        """Upload DSM (Digital Surface Model) file"""
        file_path = filedialog.askopenfilename(
            title="Select DSM File",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.dsm_path = file_path
            filename = os.path.basename(file_path)
            self.dsm_info.config(text=f"DSM: {filename[:20]}..." if len(filename) > 20 else f"DSM: {filename}")
            self.update_estimation_method()
            self.status_var.set(f"DSM file loaded: {filename}")

    def upload_dtm(self):
        """Upload DTM (Digital Terrain Model) file"""
        file_path = filedialog.askopenfilename(
            title="Select DTM File",
            filetypes=[("TIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        if file_path:
            self.dtm_path = file_path
            filename = os.path.basename(file_path)
            self.dtm_info.config(text=f"DTM: {filename[:20]}..." if len(filename) > 20 else f"DTM: {filename}")
            self.update_estimation_method()
            self.status_var.set(f"DTM file loaded: {filename}")

    def start_analysis(self):
        """Main analysis function triggered by the ANALYZE button"""
        try:
            # Validate people per building input
            new_value = float(self.people_entry.get())
            if new_value <= 0:
                raise ValueError("Value must be greater than 0.")
            self.people_per_building.set(new_value)
            
            # Check if image file is selected
            if not self.file_path:
                messagebox.showerror("No Image Selected", "Please select an image file first using the 'Choose Image File' button.")
                return
            
            # Disable analyze button during processing
            self.analyze_button.config(state='disabled', text="🔄 ANALYZING...")
            self.status_var.set("Starting analysis...")
            self.root.update_idletasks()
            
            # Run analysis in separate thread to prevent GUI freezing
            threading.Thread(target=self.analyze_image, args=(self.file_path,), daemon=True).start()
            
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid positive number for people per residential building.")

    def choose_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File", 
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
        )
        if file_path:
            self.file_path = file_path
            filename = os.path.basename(file_path)
            self.file_info.config(text=f"Selected: {filename}")
            self.status_var.set(f"Image selected: {filename}. Ready for analysis.")
            
            # Load and display the selected image immediately
            try:
                original_image = cv2.imread(file_path)
                original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
                original_pil = Image.fromarray(original_rgb).resize((600, 400))
                original_tk = ImageTk.PhotoImage(original_pil)
                self.original_label.config(image=original_tk, text="")
                self.original_label.image = original_tk
            except Exception as e:
                print(f"Error loading image preview: {e}")

    def update_estimation_method(self):
        """Update the estimation method display based on available files"""
        if self.dsm_path and self.dtm_path:
            self.method_info.config(text="Area + Floor")
            self.floor_pop_value.config(text="--")
            self.status_var.set("Height analysis enabled - will use area + floor estimation")
        else:
            self.method_info.config(text="Area-only")
            self.floor_pop_value.config(text="0")
            if not self.dsm_path and not self.dtm_path:
                self.status_var.set("Area-only estimation (no height files)")
            else:
                self.status_var.set("Need both DSM and DTM files for height analysis")

    def analyze_image(self, file_path):
        try:
            self.status_var.set("Loading image and running YOLO detection...")
            
            # Load and display original image
            original_image = cv2.imread(file_path)
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_pil = Image.fromarray(original_rgb).resize((600, 400))
            original_tk = ImageTk.PhotoImage(original_pil)
            self.original_label.config(image=original_tk, text="")
            self.original_label.image = original_tk

            self.status_var.set("Running YOLO prediction...")
            
            # Run YOLO prediction
            results = model.predict(source=file_path, save=False, save_txt=False, imgsz=640)[0]
            processed_image = original_image.copy()
            residential_count = 0
            non_residential_count = 0
            
            # Separate population counts
            area_based_population = 0
            floor_based_population = 0
            people_per_floor = self.people_per_building.get()

            self.status_var.set("Processing detections...")

            # Load height data if both files are available
            dsm = None
            dtm = None
            height_analysis_available = False
            
            if self.dsm_path and self.dtm_path:
                try:
                    self.status_var.set("Loading height analysis files...")
                    dsm = rasterio.open(self.dsm_path)
                    dtm = rasterio.open(self.dtm_path)
                    height_analysis_available = True
                except Exception as e:
                    self.status_var.set(f"Error loading height files: {str(e)}")
                    messagebox.showwarning("Height Analysis Warning", f"Could not load height analysis files.\nUsing area-only estimation.\nError: {str(e)}")

            # Get image dimensions for coordinate transformation
            img_height, img_width = original_image.shape[:2]

            # Process detections
            for i, box in enumerate(results.boxes):
                self.status_var.set(f"Processing detection {i+1}/{len(results.boxes)}...")
                
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                x_center, y_center, w, h = box.xywh[0]
                x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
                x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
                
                # Draw bounding box
                color = colors.get(cls, (255, 255, 255))
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_names.get(cls, str(cls))} ({confidence:.2f})"
                cv2.putText(processed_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

                if cls == 0:  # Residential building
                    residential_count += 1
                    
                    # ALWAYS calculate area-based population
                    area = w * h
                    area_multiplier = 1 if area < 800 else 2 if area < 1500 else 3
                    building_area_population = people_per_floor * area_multiplier
                    area_based_population += building_area_population
                    
                    # ADDITIONALLY calculate floor-based population if height analysis is available
                    if height_analysis_available and dsm and dtm:
                        try:
                            height = self.get_building_height(dsm, dtm, (x1, y1, x2, y2), img_width, img_height)
                            floors = max(1, round(height / 3))  # Assume 3m per floor
                            building_floor_population = floors * people_per_floor
                            floor_based_population += building_floor_population
                        except Exception as e:
                            print(f"Failed to get height for building: {e}")

                elif cls == 1:  # Non-residential building
                    non_residential_count += 1

            # Close height analysis files if they were opened
            if dsm:
                dsm.close()
            if dtm:
                dtm.close()

            total = residential_count + non_residential_count

            if total == 0:
                self.pop_value.config(text="--")
                self.area_pop_value.config(text="--")
                self.floor_pop_value.config(text="--")
                self.residential_count.config(text="0")
                self.non_residential_count.config(text="0")
                self.total_count.config(text="0")
                self.method_info.config(text="--")
                messagebox.showinfo("No Buildings Detected", "No buildings were detected in the selected image.")
                self.status_var.set("Analysis complete - No buildings detected.")
                self.analyze_button.config(state='normal', text="🔍 START ANALYSIS")
                return

            # Calculate total population
            if height_analysis_available and self.dsm_path and self.dtm_path:
                # Height analysis mode: Add area-based + floor-based
                total_estimated_population = area_based_population + floor_based_population
                self.method_info.config(text="Area + Floor")
                self.floor_pop_value.config(text=str(math.ceil(floor_based_population)))
            else:
                # Normal mode: Only area-based
                total_estimated_population = area_based_population
                self.method_info.config(text="Area-only")
                self.floor_pop_value.config(text="0")

            # Update UI with results
            self.pop_value.config(text=str(math.ceil(total_estimated_population)))
            self.area_pop_value.config(text=str(math.ceil(area_based_population)))
            self.residential_count.config(text=str(residential_count))
            self.non_residential_count.config(text=str(non_residential_count))
            self.total_count.config(text=str(total))

            # Display processed image
            processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            processed_pil = Image.fromarray(processed_rgb).resize((600, 400))
            processed_tk = ImageTk.PhotoImage(processed_pil)
            self.result_label.config(image=processed_tk, text="")
            self.result_label.image = processed_tk

            # Create charts
            self.create_charts(residential_count, non_residential_count)
            
            # Update status with breakdown
            if height_analysis_available and self.dsm_path and self.dtm_path:
                self.status_var.set(f"Analysis complete! {total} buildings found. Population: {math.ceil(area_based_population)} (area) + {math.ceil(floor_based_population)} (floors) = {math.ceil(total_estimated_population)} total")
            else:
                self.status_var.set(f"Analysis complete! {total} buildings found. Population: {math.ceil(total_estimated_population)} (area-based)")

        except Exception as e:
            self.status_var.set("Error during analysis")
            messagebox.showerror("Analysis Error", f"An error occurred during analysis:\n{str(e)}")
        finally:
            # Re-enable analyze button
            self.analyze_button.config(state='normal', text="🔍 START ANALYSIS")

    def get_building_height(self, dsm, dtm, bbox, img_width, img_height):
        """Extract building height from DSM and DTM data"""
        try:
            # Get raster bounds and transform
            dsm_bounds = dsm.bounds
            dtm_bounds = dtm.bounds
            dsm_transform = dsm.transform
            dtm_transform = dtm.transform
            
            # Convert image coordinates to geographic coordinates
            # This is a simplified approach - you might need to adjust based on your specific coordinate system
            x1, y1, x2, y2 = bbox
            
            # Calculate center of bounding box in image coordinates
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            
            # Convert to normalized coordinates (0-1)
            norm_x = center_x / img_width
            norm_y = center_y / img_height
            
            # Map to raster coordinates
            geo_x = dsm_bounds.left + norm_x * (dsm_bounds.right - dsm_bounds.left)
            geo_y = dsm_bounds.top - norm_y * (dsm_bounds.top - dsm_bounds.bottom)
            
            # Convert geographic coordinates to pixel coordinates
            dsm_col, dsm_row = ~dsm_transform * (geo_x, geo_y)
            dtm_col, dtm_row = ~dtm_transform * (geo_x, geo_y)
            
            # Ensure coordinates are within bounds
            dsm_col = max(0, min(int(dsm_col), dsm.width - 1))
            dsm_row = max(0, min(int(dsm_row), dsm.height - 1))
            dtm_col = max(0, min(int(dtm_col), dtm.width - 1))
            dtm_row = max(0, min(int(dtm_row), dtm.height - 1))
            
            # Read height values
            dsm_window = rasterio.windows.Window(dsm_col, dsm_row, 3, 3)
            dtm_window = rasterio.windows.Window(dtm_col, dtm_row, 3, 3)
            
            dsm_data = dsm.read(1, window=dsm_window)
            dtm_data = dtm.read(1, window=dtm_window)
            
            # Calculate height difference
            height_diff = dsm_data - dtm_data
            
            # Filter out invalid values and get mean height
            valid_heights = height_diff[height_diff > 0]
            if valid_heights.size > 0:
                return np.mean(valid_heights)
            else:
                return 3  # Default height if no valid data
                
        except Exception as e:
            print(f"Height extraction failed: {e}")
            return 3  # Default height of 3 meters (1 floor)

    def create_charts(self, residential_count, non_residential_count):
        # Clear existing charts
        for widget in self.chart_container.winfo_children():
            widget.destroy()

        if residential_count == 0 and non_residential_count == 0:
            no_data_label = tk.Label(self.chart_container, text="No buildings detected for chart display", fg="#8892b0", bg="#0f1419", font=('Segoe UI', 12))
            no_data_label.pack(fill=tk.BOTH, expand=True)
            return

        # Create new charts
        fig = Figure(figsize=(6, 4), dpi=100)
        fig.patch.set_facecolor('#2d3748')
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        labels = ['Residential', 'Non-Residential']
        values = [residential_count, non_residential_count]
        colors_list = ['#ff0088', '#00a5ff']

        # Pie chart
        if sum(values) > 0:
            ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_list)
            ax1.set_title("Building Distribution", color='white', fontsize=12)
            ax1.axis('equal')

        # Bar chart
        ax2.bar(labels, values, color=colors_list)
        ax2.set_title("Building Count", color='white', fontsize=12)
        ax2.set_ylabel("Count", color='white')
        ax2.tick_params(colors='white')
        ax2.set_ylim(0, max(values) + 1 if max(values) > 0 else 1)

        # Set background colors
        ax1.set_facecolor('#2d3748')
        ax2.set_facecolor('#2d3748')

        # Add chart to GUI
        canvas = FigureCanvasTkAgg(fig, master=self.chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernYOLOApp(root)
    root.mainloop()