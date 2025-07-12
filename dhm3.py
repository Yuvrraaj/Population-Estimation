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
import rasterio
import time

# ==== Load YOLO Model ====
model = YOLO('runs/detect/train/weights/best.pt')  # Update this path

# ==== Constants ====
colors = {0: (255, 0, 255), 1: (0, 165, 255)}
class_names = {0: 'Residential', 1: 'Non-Residential'}

class ModernYOLOApp:
    def __init__(self, root):
        self.root = root
        self.people_per_building = tk.DoubleVar(value=4.23)
        self.dhm_enabled = tk.BooleanVar(value=False)
        self.file_path = None
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
        self.style.configure('Modern.TButton', background='#00d4aa', foreground='white', borderwidth=0, padding=(20, 15), font=('Segoe UI', 12, 'bold'))
        self.style.map('Modern.TButton', background=[('active', '#00b899'), ('pressed', '#009688')])
        self.style.configure('Modern.Horizontal.TProgressbar', background='#00d4aa', troughcolor='#1e2328', borderwidth=0)
        self.style.configure('DHM.TCheckbutton', background='#1a1f26', foreground='white', font=('Segoe UI', 11))

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
        self.upload_btn = ttk.Button(parent, text="Choose Image File", style='Modern.TButton', command=self.choose_image)
        self.upload_btn.pack(fill=tk.X, pady=10, padx=10)

        self.file_info = tk.Label(parent, text="No file selected", fg="#8892b0", bg="#1a1f26", wraplength=250)
        self.file_info.pack(anchor=tk.W, padx=10)

        # DHM Toggle Section
        dhm_frame = tk.Frame(parent, bg="#1a1f26")
        dhm_frame.pack(pady=10, padx=10, fill=tk.X)
        
        dhm_label = tk.Label(dhm_frame, text="Height Analysis Options:", fg="white", bg="#1a1f26", font=('Segoe UI', 11, 'bold'))
        dhm_label.pack(anchor=tk.W)
        
        self.dhm_checkbox = tk.Checkbutton(
            dhm_frame, 
            text="Enable DHM (Digital Height Model)", 
            variable=self.dhm_enabled,
            bg="#1a1f26", 
            fg="white", 
            selectcolor="#00d4aa",
            font=('Segoe UI', 10),
            command=self.toggle_dhm_mode
        )
        self.dhm_checkbox.pack(anchor=tk.W, pady=5)
        
        self.dhm_info = tk.Label(
            dhm_frame, 
            text="Adds floor-based count to area-based count (higher population)", 
            fg="#8892b0", 
            bg="#1a1f26", 
            font=('Segoe UI', 9),
            wraplength=300
        )
        self.dhm_info.pack(anchor=tk.W, padx=20)

        input_frame = tk.Frame(parent, bg="#1a1f26")
        input_frame.pack(pady=10, padx=10, fill=tk.X)
        label = tk.Label(input_frame, text="People/Residential Building:", fg="white", bg="#1a1f26")
        label.pack(anchor=tk.W)
        self.people_entry = tk.Entry(input_frame, textvariable=self.people_per_building)
        self.people_entry.pack(fill=tk.X, pady=5)
        self.enter_button = ttk.Button(input_frame, text="Enter", command=self.update_people_per_building)
        self.enter_button.pack(fill=tk.X)

        stats_frame = tk.Frame(parent, bg="#1a1f26")
        stats_frame.pack(pady=10, padx=10, fill=tk.X)

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
        self.original_label = tk.Label(parent, bg="#0f1419", fg="#8892b0", text="Original Image Here")
        self.original_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.result_label = tk.Label(parent, bg="#0f1419", fg="#8892b0", text="Detection Results Here")
        self.result_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=10, pady=10)

    def create_chart_panel(self, parent):
        self.chart_container = tk.Frame(parent, bg="#2d3748")
        self.chart_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.chart_placeholder = tk.Label(self.chart_container, text="Charts will appear here", fg="#8892b0", bg="#0f1419")
        self.chart_placeholder.pack(fill=tk.BOTH, expand=True)

    def create_status_bar(self, parent):
        self.status_var = tk.StringVar()
        self.status_var.set("Ready to analyze images")
        status_label = tk.Label(parent, textvariable=self.status_var, bg="#1a1f26", fg="#ffffff", anchor=tk.W)
        status_label.pack(fill=tk.X, pady=10, padx=15)

    def toggle_dhm_mode(self):
        if self.dhm_enabled.get():
            self.method_info.config(text="Area + Floor")
            self.status_var.set("DHM mode enabled - will add area + floor population counts")
            self.floor_pop_value.config(text="--")
        else:
            self.method_info.config(text="Area-only")
            self.status_var.set("Area-only estimation mode")
            self.floor_pop_value.config(text="0")

    def choose_image(self):
        self.file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if self.file_path:
            self.file_info.config(text=f"Selected: {os.path.basename(self.file_path)}")
            self.status_var.set("Set value, click Enter to update, then analysis will proceed")

    def update_people_per_building(self):
        try:
            new_value = float(self.people_entry.get())
            if new_value <= 0:
                raise ValueError("Value must be greater than 0.")
            self.people_per_building.set(new_value)
            if self.file_path:
                self.status_var.set("Analyzing image...")
                self.root.update_idletasks()
                threading.Thread(target=self.analyze_image, args=(self.file_path,), daemon=True).start()
            else:
                messagebox.showinfo("Input Accepted", f"Population multiplier set to {new_value}. Please upload an image to run the estimation.")
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid positive number for people per residential building.")

    def analyze_image(self, file_path):
        try:
            # Load and display original image
            original_image = cv2.imread(file_path)
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_pil = Image.fromarray(original_rgb).resize((600, 400))
            original_tk = ImageTk.PhotoImage(original_pil)
            self.original_label.config(image=original_tk, text="")
            self.original_label.image = original_tk

            # Run YOLO prediction
            results = model.predict(source=file_path, save=False, save_txt=False, imgsz=640)[0]
            processed_image = original_image.copy()
            residential_count = 0
            non_residential_count = 0
            
            # Separate population counts
            area_based_population = 0
            floor_based_population = 0
            people_per_floor = self.people_per_building.get()

            # Load DHM data if enabled
            dsm = None
            dtm = None
            dhm_available = False
            
            if self.dhm_enabled.get():
                try:
                    dsm_path = "DSM.tif"
                    dtm_path = "DTM.tif"
                    dsm = rasterio.open(dsm_path)
                    dtm = rasterio.open(dtm_path)
                    dhm_available = True
                    self.status_var.set("DHM files loaded successfully")
                except Exception as e:
                    self.status_var.set(f"DHM files not found: {str(e)}")
                    messagebox.showwarning("DHM Warning", f"DHM files (DSM.tif/DTM.tif) not found. Using area-only estimation.\nError: {str(e)}")

            # Process detections
            for box in results.boxes:
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
                    
                    # ADDITIONALLY calculate floor-based population if DHM is enabled
                    if self.dhm_enabled.get() and dhm_available and dsm and dtm:
                        try:
                            height = self.get_building_height(dsm, dtm, (x1, y1, x2, y2))
                            floors = max(1, round(height / 3))  # Assume 3m per floor
                            building_floor_population = floors * people_per_floor
                            floor_based_population += building_floor_population
                        except Exception as e:
                            print(f"Failed to get height for building: {e}")
                            # Don't add anything to floor_based_population if height extraction fails

                elif cls == 1:  # Non-residential building
                    non_residential_count += 1

            # Close DHM files if they were opened
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
                self.status_var.set("No buildings detected.")
                return

            # Calculate total population
            if self.dhm_enabled.get() and dhm_available:
                # DHM mode: Add area-based + floor-based
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
            if self.dhm_enabled.get() and dhm_available:
                self.status_var.set(f"Analysis complete. {total} buildings found. Population: {math.ceil(area_based_population)} (area) + {math.ceil(floor_based_population)} (floors) = {math.ceil(total_estimated_population)} total")
            else:
                self.status_var.set(f"Analysis complete. {total} buildings found. Population: {math.ceil(total_estimated_population)} (area-based)")

        except Exception as e:
            self.status_var.set("Error during processing")
            messagebox.showerror("Error", f"An error occurred during analysis:\n{str(e)}")

    def get_building_height(self, dsm, dtm, bbox):
        """Extract building height from DSM and DTM data"""
        try:
            # Create window from bounding box
            window = rasterio.windows.Window.from_slices(
                (bbox[1], bbox[3]),  # y1, y2
                (bbox[0], bbox[2])   # x1, x2
            )
            
            # Read data from both rasters
            dsm_data = dsm.read(1, window=window)
            dtm_data = dtm.read(1, window=window)
            
            # Calculate height difference
            height = dsm_data - dtm_data
            
            # Filter out negative or zero heights
            height = height[height > 0]
            
            # Return mean height or 0 if no valid data
            return np.mean(height) if height.size > 0 else 0
            
        except Exception as e:
            print(f"Height extraction failed: {e}")
            return 0

    def create_charts(self, residential_count, non_residential_count):
        # Clear existing charts
        for widget in self.chart_container.winfo_children():
            widget.destroy()

        # Create new charts
        fig = Figure(figsize=(6, 4), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        labels = ['Residential', 'Non-Residential']
        values = [residential_count, non_residential_count]
        colors_list = ['#ff0088', '#00a5ff']

        # Pie chart
        ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_list)
        ax1.set_title("Building Distribution")
        ax1.axis('equal')

        # Bar chart
        ax2.bar(labels, values, color=colors_list)
        ax2.set_title("Building Count")
        ax2.set_ylabel("Count")

        # Add chart to GUI
        canvas = FigureCanvasTkAgg(fig, master=self.chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernYOLOApp(root)
    root.mainloop()