import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageFilter, ImageDraw
import cv2
import os
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from ultralytics import YOLO

# ==== Load YOLO Model ====
model = YOLO('runs/detect/train/weights/best.pt')  # Update this path

# ==== Constants ====
PEOPLE_PER_BUILDING = 4.23
colors = {0: (255, 0, 255), 1: (0, 165, 255)}
class_names = {0: 'Residential', 1: 'Non-Residential'}

class ModernYOLOApp:
    def __init__(self, root):
        self.root = root
        self.setup_window()
        self.create_styles()
        self.create_widgets()
        
    def setup_window(self):
        self.root.title("YOLO Building & Population Estimator")
        self.root.geometry("1200x900")  # Increased height for pie chart
        self.root.configure(bg="#0f1419")
        self.root.resizable(True, True)
        
        # Center window on screen
        self.center_window()
        
    def center_window(self):
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (1200 // 2)
        y = (self.root.winfo_screenheight() // 2) - (900 // 2)
        self.root.geometry(f"1200x900+{x}+{y}")
        
    def create_styles(self):
        # Configure ttk styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Modern button style
        self.style.configure('Modern.TButton',
                           background='#00d4aa',
                           foreground='white',
                           borderwidth=0,
                           focuscolor='none',
                           padding=(20, 15),
                           font=('Segoe UI', 12, 'bold'))
        
        self.style.map('Modern.TButton',
                      background=[('active', '#00b899'),
                                ('pressed', '#009688')])
        
        # Progress bar style
        self.style.configure('Modern.Horizontal.TProgressbar',
                           background='#00d4aa',
                           troughcolor='#1e2328',
                           borderwidth=0,
                           lightcolor='#00d4aa',
                           darkcolor='#00d4aa')
        
    def create_widgets(self):
        # Main container with gradient effect
        main_frame = tk.Frame(self.root, bg="#0f1419")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header section with modern design
        self.create_header(main_frame)
        
        # Content area with cards
        content_frame = tk.Frame(main_frame, bg="#0f1419")
        content_frame.pack(fill=tk.BOTH, expand=True, pady=(20, 0))
        
        # Left panel for controls
        left_panel = tk.Frame(content_frame, bg="#1a1f26", relief=tk.FLAT, bd=0, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)  # Maintain fixed width
        
        # Add control panel
        self.create_control_panel(left_panel)
        
        # Right panel container
        right_container = tk.Frame(content_frame, bg="#0f1419")
        right_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        # Right panel for image display
        right_panel = tk.Frame(right_container, bg="#1a1f26", relief=tk.FLAT, bd=0, height=500)
        right_panel.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        self.create_image_panel(right_panel)
        
        # Chart panel below images
        chart_panel = tk.Frame(right_container, bg="#1a1f26", relief=tk.FLAT, bd=0, height=250)
        chart_panel.pack(fill=tk.X, pady=(10, 0))
        chart_panel.pack_propagate(False)
        
        self.create_chart_panel(chart_panel)
        
        # Status bar at bottom
        self.create_status_bar(main_frame)
        
    def create_header(self, parent):
        header_frame = tk.Frame(parent, bg="#0f1419", height=100)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        header_frame.pack_propagate(False)
        
        # Title with gradient effect
        title_frame = tk.Frame(header_frame, bg="#0f1419")
        title_frame.pack(expand=True)
        
        title = tk.Label(title_frame, 
                        text="Population Estimator",
                        font=('Segoe UI', 28, 'bold'),
                        fg="#ffffff",
                        bg="#0f1419")
        title.pack(pady=(10, 5))
        
        subtitle = tk.Label(title_frame,
                          text="Advanced Population Estimation with YOLO",
                          font=('Segoe UI', 14),
                          fg="#8892b0",
                          bg="#0f1419")
        subtitle.pack()
        
        # Decorative line
        line_frame = tk.Frame(header_frame, bg="#00d4aa", height=3)
        line_frame.pack(fill=tk.X, pady=(10, 0))
        
    def create_control_panel(self, parent):
        # Control panel with modern card design
        control_frame = tk.Frame(parent, bg="#1a1f26", padx=20, pady=20)
        control_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Upload section
        upload_section = tk.Frame(control_frame, bg="#1a1f26")
        upload_section.pack(fill=tk.X, pady=(0, 30))
        
        upload_title = tk.Label(upload_section,
                               text="📁 Upload Image",
                               font=('Segoe UI', 16, 'bold'),
                               fg="#ffffff",
                               bg="#1a1f26")
        upload_title.pack(anchor=tk.W, pady=(0, 15))
        
        # Modern upload button
        self.upload_btn = ttk.Button(upload_section,
                                   text="Choose Image File",
                                   style='Modern.TButton',
                                   command=self.process_image)
        self.upload_btn.pack(fill=tk.X, pady=(0, 10))
        
        # File info
        self.file_info = tk.Label(upload_section,
                                 text="No file selected",
                                 font=('Segoe UI', 10),
                                 fg="#8892b0",
                                 bg="#1a1f26",
                                 wraplength=250)
        self.file_info.pack(anchor=tk.W)
        
        # Separator
        separator = tk.Frame(control_frame, bg="#2d3748", height=2)
        separator.pack(fill=tk.X, pady=20)
        
        # Results section
        results_section = tk.Frame(control_frame, bg="#1a1f26")
        results_section.pack(fill=tk.X, pady=(0, 30))
        
        results_title = tk.Label(results_section,
                               text="📊 Analysis Results",
                               font=('Segoe UI', 16, 'bold'),
                               fg="#ffffff",
                               bg="#1a1f26")
        results_title.pack(anchor=tk.W, pady=(0, 15))
        
        # Population card
        pop_card = tk.Frame(results_section, bg="#2d3748", padx=15, pady=15)
        pop_card.pack(fill=tk.X, pady=(0, 10))
        
        self.pop_icon = tk.Label(pop_card,
                               text="👥",
                               font=('Segoe UI', 24),
                               bg="#2d3748")
        self.pop_icon.pack(side=tk.LEFT, padx=(0, 10))
        
        pop_text_frame = tk.Frame(pop_card, bg="#2d3748")
        pop_text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        pop_label = tk.Label(pop_text_frame,
                           text="Estimated Population",
                           font=('Segoe UI', 10),
                           fg="#8892b0",
                           bg="#2d3748")
        pop_label.pack(anchor=tk.W)
        
        self.pop_value = tk.Label(pop_text_frame,
                                text="---",
                                font=('Segoe UI', 20, 'bold'),
                                fg="#00d4aa",
                                bg="#2d3748")
        self.pop_value.pack(anchor=tk.W)
        
        # Detection stats - Residential buildings
        residential_card = tk.Frame(results_section, bg="#2d3748", padx=15, pady=10)
        residential_card.pack(fill=tk.X, pady=(0, 5))
        
        res_icon = tk.Label(residential_card,
                           text="🏠",
                           font=('Segoe UI', 16),
                           bg="#2d3748")
        res_icon.pack(side=tk.LEFT, padx=(0, 10))
        
        res_text_frame = tk.Frame(residential_card, bg="#2d3748")
        res_text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        res_label = tk.Label(res_text_frame,
                           text="Residential Buildings",
                           font=('Segoe UI', 10),
                           fg="#8892b0",
                           bg="#2d3748")
        res_label.pack(anchor=tk.W)
        
        self.residential_count = tk.Label(res_text_frame,
                                        text="0",
                                        font=('Segoe UI', 16, 'bold'),
                                        fg="#ff0088",
                                        bg="#2d3748")
        self.residential_count.pack(anchor=tk.W)
        
        # Detection stats - Non-residential buildings
        non_residential_card = tk.Frame(results_section, bg="#2d3748", padx=15, pady=10)
        non_residential_card.pack(fill=tk.X, pady=(0, 10))
        
        non_res_icon = tk.Label(non_residential_card,
                               text="🏢",
                               font=('Segoe UI', 16),
                               bg="#2d3748")
        non_res_icon.pack(side=tk.LEFT, padx=(0, 10))
        
        non_res_text_frame = tk.Frame(non_residential_card, bg="#2d3748")
        non_res_text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        non_res_label = tk.Label(non_res_text_frame,
                               text="Non-Residential Buildings",
                               font=('Segoe UI', 10),
                               fg="#8892b0",
                               bg="#2d3748")
        non_res_label.pack(anchor=tk.W)
        
        self.non_residential_count = tk.Label(non_res_text_frame,
                                            text="0",
                                            font=('Segoe UI', 16, 'bold'),
                                            fg="#00a5ff",
                                            bg="#2d3748")
        self.non_residential_count.pack(anchor=tk.W)
        
        # Total detections
        total_card = tk.Frame(results_section, bg="#2d3748", padx=15, pady=10)
        total_card.pack(fill=tk.X)
        
        total_icon = tk.Label(total_card,
                            text="📍",
                            font=('Segoe UI', 16),
                            bg="#2d3748")
        total_icon.pack(side=tk.LEFT, padx=(0, 10))
        
        total_text_frame = tk.Frame(total_card, bg="#2d3748")
        total_text_frame.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        total_label = tk.Label(total_text_frame,
                             text="Total Detections",
                             font=('Segoe UI', 10),
                             fg="#8892b0",
                             bg="#2d3748")
        total_label.pack(anchor=tk.W)
        
        self.total_count = tk.Label(total_text_frame,
                                  text="0",
                                  font=('Segoe UI', 16, 'bold'),
                                  fg="#ffffff",
                                  bg="#2d3748")
        self.total_count.pack(anchor=tk.W)
        
    def create_image_panel(self, parent):
        # Image display panel
        image_frame = tk.Frame(parent, bg="#1a1f26", padx=25, pady=25)
        image_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        image_title = tk.Label(image_frame,
                             text="🖼️ Image Analysis - Before & After",
                             font=('Segoe UI', 16, 'bold'),
                             fg="#ffffff",
                             bg="#1a1f26")
        image_title.pack(anchor=tk.W, pady=(0, 15))
        
        # Container for both images
        images_container = tk.Frame(image_frame, bg="#1a1f26")
        images_container.pack(fill=tk.BOTH, expand=True)
        
        # Original image section
        original_section = tk.Frame(images_container, bg="#1a1f26")
        original_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        original_title = tk.Label(original_section,
                                text="📷 Original Image",
                                font=('Segoe UI', 12, 'bold'),
                                fg="#8892b0",
                                bg="#1a1f26")
        original_title.pack(pady=(0, 10))
        
        # Original image container with border
        self.original_container = tk.Frame(original_section, bg="#2d3748", padx=2, pady=2)
        self.original_container.pack(fill=tk.BOTH, expand=True)
        
        # Original image placeholder
        self.original_label = tk.Label(self.original_container, 
                                     bg="#0f1419",
                                     fg="#8892b0",
                                     font=('Segoe UI', 12),
                                     text="\n\nOriginal Image\nwill appear here")
        self.original_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Processed image section
        processed_section = tk.Frame(images_container, bg="#1a1f26")
        processed_section.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0))
        
        processed_title = tk.Label(processed_section,
                                 text="🎯 Detection Results",
                                 font=('Segoe UI', 12, 'bold'),
                                 fg="#8892b0",
                                 bg="#1a1f26")
        processed_title.pack(pady=(0, 10))
        
        # Processed image container with border
        self.processed_container = tk.Frame(processed_section, bg="#2d3748", padx=2, pady=2)
        self.processed_container.pack(fill=tk.BOTH, expand=True)
        
        # Processed image placeholder
        self.result_label = tk.Label(self.processed_container, 
                                   bg="#0f1419",
                                   fg="#8892b0",
                                   font=('Segoe UI', 12),
                                   text="🎯\n\nDetection Results\nwill appear here")
        self.result_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Progress bar (hidden initially)
        self.progress_frame = tk.Frame(image_frame, bg="#1a1f26")
        
        self.progress_label = tk.Label(self.progress_frame,
                                     text="Processing...",
                                     font=('Segoe UI', 10),
                                     fg="#8892b0",
                                     bg="#1a1f26")
        self.progress_label.pack(pady=(10, 5))
        
        self.progress_bar = ttk.Progressbar(self.progress_frame,
                                          style='Modern.Horizontal.TProgressbar',
                                          mode='indeterminate',
                                          length=300)
        self.progress_bar.pack(pady=(0, 10))
        
    def create_chart_panel(self, parent):
        # Chart panel
        chart_frame = tk.Frame(parent, bg="#1a1f26", padx=0, pady=0)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)
        
        chart_title = tk.Label(chart_frame,
                             text="📊 Building Distribution",
                             font=('Segoe UI', 16, 'bold'),
                             fg="#ffffff",
                             bg="#1a1f26")
        chart_title.pack(anchor=tk.W, pady=(0, 5))
        
        # Chart container
        self.chart_container = tk.Frame(chart_frame, bg="#2d3748", padx=5, pady=5)
        self.chart_container.pack(fill=tk.BOTH, expand=True)
        
        # Placeholder for chart
        self.chart_placeholder = tk.Label(self.chart_container,
                                        text="📊\n\nBuilding distribution chart\nwill appear here after analysis",
                                        font=('Segoe UI', 12),
                                        fg="#8892b0",
                                        bg="#0f1419")
        self.chart_placeholder.pack(fill=tk.BOTH, expand=True)
        
    def create_status_bar(self, parent):
        status_frame = tk.Frame(parent, bg="#1a1f26", height=40)
        status_frame.pack(fill=tk.X, pady=(20, 0))
        status_frame.pack_propagate(False)
        
        self.status_var = tk.StringVar()
        self.status_var.set("🚀 Ready to analyze images")
        
        self.status_label = tk.Label(status_frame,
                                   textvariable=self.status_var,
                                   font=('Segoe UI', 10),
                                   fg="#8892b0",
                                   bg="#1a1f26",
                                   anchor=tk.W,
                                   padx=15)
        self.status_label.pack(fill=tk.X, pady=10)
        
    def create_pie_chart(self, residential_count, non_residential_count):
        # Clear existing chart
        for widget in self.chart_container.winfo_children():
            widget.destroy()
        
        if residential_count == 0 and non_residential_count == 0:
            # Show placeholder if no data
            self.chart_placeholder = tk.Label(self.chart_container,
                                            text="📊\n\nNo buildings detected\nto display in chart",
                                            font=('Segoe UI', 12),
                                            fg="#8892b0",
                                            bg="#0f1419")
            self.chart_placeholder.pack(fill=tk.BOTH, expand=True)
            return
        
        # Create matplotlib figure with dark theme
        plt.style.use('dark_background')
        fig = Figure(figsize=(6, 3), dpi=100, facecolor='#0f1419')
        ax = fig.add_subplot(111, facecolor='#0f1419')
        
        # Data for pie chart
        labels = []
        sizes = []
        colors_list = []
        
        if residential_count > 0:
            labels.append(f'Residential\n({residential_count})')
            sizes.append(residential_count)
            colors_list.append('#ff0088')
        
        if non_residential_count > 0:
            labels.append(f'Non-Residential\n({non_residential_count})')
            sizes.append(non_residential_count)
            colors_list.append('#00a5ff')
        
        # Create pie chart
        wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors_list, 
                                         autopct='%1.1f%%', startangle=90,
                                         textprops={'color': 'white', 'fontsize': 10})
        
        # Customize text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        ax.set_title('Building Type Distribution', color='white', fontsize=12, fontweight='bold', pad=20)
        
        # Remove axes
        ax.axis('equal')
        
        # Create canvas and embed in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def show_progress(self, show=True):
        if show:
            self.progress_frame.pack(pady=(10, 0))
            self.progress_bar.start(10)
        else:
            self.progress_bar.stop()
            self.progress_frame.pack_forget()
            
    def process_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image File",
            filetypes=[
                ("All Image files", "*.png *.jpg *.jpeg *.bmp *.gif *.tiff"),
                ("PNG files", "*.png"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("All files", "*.*")
            ]
        )
        
        if not file_path:
            return
            
        # Update file info
        filename = os.path.basename(file_path)
        self.file_info.config(text=f"Selected: {filename}")
        
        # Show progress
        self.status_var.set("🔍 Analyzing image with AI...")
        self.show_progress(True)
        self.root.update_idletasks()
        
        try:
            # Load and display original image first
            original_image = cv2.imread(file_path)
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_pil = Image.fromarray(original_rgb)
            
            # Resize original image to fit container
            container_width = 600
            container_height = 400
            original_pil = original_pil.resize((container_width, container_height), Image.Resampling.LANCZOS)

            
            original_tk = ImageTk.PhotoImage(original_pil)
            self.original_label.config(image=original_tk, text="")
            self.original_label.image = original_tk
            
            # Run YOLO detection
            results = model.predict(source=file_path, save=False, save_txt=False, imgsz=640)
            result = results[0]
            
            # Create a copy of the original image for processing
            processed_image = original_image.copy()
            height, width = processed_image.shape[:2]
            estimated_population = 0
            residential_count = 0
            non_residential_count = 0
            
            for box in result.boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                x_center, y_center, w, h = box.xywh[0]
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)
                
                # Debug print to see what classes are being detected
                # print(f"Detected class: {cls}, confidence: {confidence:.2f}, class_name: {class_names.get(cls, 'Unknown')}")
                
                # Draw boxes with modern styling
                color = colors.get(cls, (255, 255, 255))
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 3)
                
                # Add label background with smaller text
                label_text = f"{class_names.get(cls, str(cls))}"
                font_scale = 0.4
                thickness = 1
                
                (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # Create smaller background rectangle
                cv2.rectangle(processed_image, (x1, y1 - text_height - 8), (x1 + text_width + 8, y1), color, -1)
                cv2.putText(processed_image, label_text, (x1 + 4, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                
                # Count buildings
                if cls == 0:  # Residential buildings
                    residential_count += 1
                    area = w * h
                    if area < 800:
                        multiplier = 1
                    elif area < 1500:
                        multiplier = 2
                    else:
                        multiplier = 3
                    estimated_population += PEOPLE_PER_BUILDING * multiplier
                elif cls == 1:  # Non-residential buildings
                    non_residential_count += 1
                    
                    # You can add population estimation for non-residential if needed
                    # For now, we're only counting residential for population
                else:
                    # Handle any other classes that might be detected
                    print(f"Unknown class detected: {cls}")
                    non_residential_count += 1  # Count unknown classes as non-residential
                    
            estimated_population = math.ceil(estimated_population)
            total_buildings = residential_count + non_residential_count
            
            # Update results
            self.pop_value.config(text=str(estimated_population))
            self.residential_count.config(text=str(residential_count))
            self.non_residential_count.config(text=str(non_residential_count))
            self.total_count.config(text=str(total_buildings))
            
            # Display processed image
            processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            processed_pil = Image.fromarray(processed_rgb)
            
            # Resize processed image to fit container
            processed_pil = processed_pil.resize((container_width, container_height), Image.Resampling.LANCZOS)

            
            processed_tk = ImageTk.PhotoImage(processed_pil)
            self.result_label.config(image=processed_tk, text="")
            self.result_label.image = processed_tk
            
            # Create pie chart
            self.create_pie_chart(residential_count, non_residential_count)
            
            self.status_var.set(f"✅ Analysis complete! Found {total_buildings} buildings")
            
        except Exception as e:
            self.status_var.set("❌ Error during processing")
            messagebox.showerror("Error", f"Failed to process image:\n{str(e)}")
            
        finally:
            self.show_progress(False)

# ==== Main Application ====
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernYOLOApp(root)
    root.mainloop()