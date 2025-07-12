import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import os
import math
import sys

# Simple YOLO mock class for testing without actual model
class MockYOLO:
    def predict(self, source, save=False, save_txt=False, imgsz=640):
        # Mock detection results for testing
        class MockBox:
            def __init__(self, cls, conf, xywh):
                self.cls = [cls]
                self.conf = [conf]
                self.xywh = [xywh]
        
        class MockResult:
            def __init__(self):
                # Mock some detections for testing
                self.boxes = [
                    MockBox(0, 0.85, [100, 100, 80, 60]),  # Residential
                    MockBox(1, 0.92, [300, 200, 120, 90]), # Non-residential
                    MockBox(0, 0.78, [500, 150, 70, 50]),  # Residential
                ]
        
        return [MockResult()]

# Try to load actual YOLO model, fallback to mock if not available
try:
    from ultralytics import YOLO
    model_path = 'runs/detect/train/weights/best.pt'
    if os.path.exists(model_path):
        model = YOLO(model_path)
    else:
        print("YOLO model not found, using mock for testing")
        model = MockYOLO()
except ImportError:
    print("Ultralytics not available, using mock for testing")
    model = MockYOLO()

# Constants
colors = {0: (255, 0, 255), 1: (0, 165, 255)}
class_names = {0: 'Residential', 1: 'Non-Residential'}

class SimpleYOLOApp:
    def __init__(self, root):
        self.root = root
        self.people_per_building = 4.23
        self.file_path = None
        self.setup_window()
        self.create_widgets()

    def setup_window(self):
        self.root.title("Simple YOLO Building Detector")
        self.root.geometry("1000x700")
        self.root.configure(bg="#2c3e50")
        self.center_window()

    def center_window(self):
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (500)
        y = (self.root.winfo_screenheight() // 2) - (350)
        self.root.geometry(f"1000x700+{x}+{y}")

    def create_widgets(self):
        # Header
        header = tk.Label(
            self.root, 
            text="Simple Population Estimator", 
            font=('Arial', 24, 'bold'), 
            fg="white", 
            bg="#2c3e50"
        )
        header.pack(pady=20)

        # Main container
        main_frame = tk.Frame(self.root, bg="#2c3e50")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        # Left panel - Controls
        left_panel = tk.Frame(main_frame, bg="#34495e", width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)

        # File selection
        tk.Label(left_panel, text="Image Selection:", font=('Arial', 12, 'bold'), 
                fg="white", bg="#34495e").pack(pady=(20, 10))
        
        self.upload_btn = tk.Button(
            left_panel, 
            text="Choose Image File", 
            command=self.choose_image,
            bg="#3498db", 
            fg="white", 
            font=('Arial', 12), 
            padx=20, 
            pady=10
        )
        self.upload_btn.pack(pady=10)

        self.file_info = tk.Label(
            left_panel, 
            text="No file selected", 
            fg="#bdc3c7", 
            bg="#34495e", 
            wraplength=250
        )
        self.file_info.pack(pady=5)

        # People per building input
        tk.Label(left_panel, text="People per Residential Building:", 
                font=('Arial', 11, 'bold'), fg="white", bg="#34495e").pack(pady=(20, 5))
        
        self.people_entry = tk.Entry(left_panel, font=('Arial', 11))
        self.people_entry.insert(0, str(self.people_per_building))
        self.people_entry.pack(pady=5)

        # Analysis button
        self.analyze_button = tk.Button(
            left_panel, 
            text="START ANALYSIS", 
            command=self.analyze_image,
            bg="#e74c3c", 
            fg="white", 
            font=('Arial', 14, 'bold'), 
            padx=30, 
            pady=15
        )
        self.analyze_button.pack(pady=30)

        # Results section
        results_frame = tk.Frame(left_panel, bg="#34495e")
        results_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(results_frame, text="Results:", font=('Arial', 12, 'bold'), 
                fg="white", bg="#34495e").pack()

        # Result labels
        self.result_labels = {}
        results = [
            ("Estimated Population:", "#e74c3c"),
            ("Residential Buildings:", "#f39c12"),
            ("Non-Residential Buildings:", "#3498db"),
            ("Total Buildings:", "#2ecc71")
        ]

        for label_text, color in results:
            frame = tk.Frame(results_frame, bg="#34495e")
            frame.pack(fill=tk.X, pady=2)
            tk.Label(frame, text=label_text, font=('Arial', 10), 
                    fg="#bdc3c7", bg="#34495e").pack(side=tk.LEFT)
            value_label = tk.Label(frame, text="--", font=('Arial', 10, 'bold'), 
                                 fg=color, bg="#34495e")
            value_label.pack(side=tk.RIGHT)
            self.result_labels[label_text] = value_label

        # Right panel - Images
        right_panel = tk.Frame(main_frame, bg="#34495e")
        right_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Image display
        image_frame = tk.Frame(right_panel, bg="#34495e")
        image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.original_label = tk.Label(
            image_frame, 
            bg="#2c3e50", 
            fg="#bdc3c7", 
            text="Original Image\n(Select an image file)", 
            font=('Arial', 12)
        )
        self.original_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(0, 5))

        self.result_label = tk.Label(
            image_frame, 
            bg="#2c3e50", 
            fg="#bdc3c7", 
            text="Detection Results\n(Click START ANALYSIS)", 
            font=('Arial', 12)
        )
        self.result_label.pack(side=tk.LEFT, expand=True, fill=tk.BOTH, padx=(5, 0))

        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready - Select an image file and click START ANALYSIS")
        status_label = tk.Label(
            self.root, 
            textvariable=self.status_var, 
            bg="#34495e", 
            fg="#2ecc71", 
            font=('Arial', 10)
        )
        status_label.pack(fill=tk.X, pady=5)

    def choose_image(self):
        """Choose image file"""
        try:
            file_path = filedialog.askopenfilename(
                title="Select Image File", 
                filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tiff *.tif")]
            )
            if file_path:
                self.file_path = file_path
                filename = os.path.basename(file_path)
                self.file_info.config(text=f"Selected: {filename}")
                self.status_var.set(f"Image selected: {filename}")
                
                # Load and display the selected image
                self.display_original_image(file_path)
                
        except Exception as e:
            messagebox.showerror("Error", f"Error selecting image: {str(e)}")

    def display_original_image(self, file_path):
        """Display the original image"""
        try:
            # Load image
            image = cv2.imread(file_path)
            if image is None:
                raise ValueError("Could not load image")
                
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image to fit display
            height, width = image_rgb.shape[:2]
            max_size = 400
            if width > height:
                new_width = max_size
                new_height = int((max_size * height) / width)
            else:
                new_height = max_size
                new_width = int((max_size * width) / height)
            
            image_pil = Image.fromarray(image_rgb).resize((new_width, new_height))
            image_tk = ImageTk.PhotoImage(image_pil)
            
            self.original_label.config(image=image_tk, text="")
            self.original_label.image = image_tk  # Keep a reference
            
        except Exception as e:
            self.status_var.set(f"Error loading image: {str(e)}")

    def analyze_image(self):
        """Analyze the selected image"""
        try:
            # Validate input
            try:
                self.people_per_building = float(self.people_entry.get())
                if self.people_per_building <= 0:
                    raise ValueError("Value must be greater than 0.")
            except ValueError:
                messagebox.showerror("Invalid Input", 
                                   "Please enter a valid positive number for people per building.")
                return
            
            # Check if image is selected
            if not self.file_path:
                messagebox.showerror("No Image", "Please select an image file first.")
                return
            
            # Update status
            self.status_var.set("Analyzing image...")
            self.analyze_button.config(state='disabled', text="ANALYZING...")
            self.root.update()
            
            # Load image
            image = cv2.imread(self.file_path)
            if image is None:
                raise ValueError("Could not load image")
            
            # Run YOLO prediction
            results = model.predict(source=self.file_path, save=False, save_txt=False, imgsz=640)[0]
            
            # Process results
            processed_image = image.copy()
            residential_count = 0
            non_residential_count = 0
            
            # Process detections
            for box in results.boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                x_center, y_center, w, h = box.xywh[0]
                
                # Convert to corner coordinates
                x1 = int(x_center - w / 2)
                y1 = int(y_center - h / 2)
                x2 = int(x_center + w / 2)
                y2 = int(y_center + h / 2)
                
                # Draw bounding box
                color = colors.get(cls, (255, 255, 255))
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                
                # Add label
                label = f"{class_names.get(cls, str(cls))} ({confidence:.2f})"
                cv2.putText(processed_image, label, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Count buildings
                if cls == 0:  # Residential
                    residential_count += 1
                elif cls == 1:  # Non-residential
                    non_residential_count += 1
            
            # Calculate population
            estimated_population = residential_count * self.people_per_building
            total_buildings = residential_count + non_residential_count
            
            # Update results
            self.result_labels["Estimated Population:"].config(text=str(int(estimated_population)))
            self.result_labels["Residential Buildings:"].config(text=str(residential_count))
            self.result_labels["Non-Residential Buildings:"].config(text=str(non_residential_count))
            self.result_labels["Total Buildings:"].config(text=str(total_buildings))
            
            # Display processed image
            self.display_processed_image(processed_image)
            
            # Update status
            self.status_var.set(f"Analysis complete! Found {total_buildings} buildings, "
                              f"estimated population: {int(estimated_population)}")
            
        except Exception as e:
            self.status_var.set(f"Error during analysis: {str(e)}")
            messagebox.showerror("Analysis Error", f"An error occurred: {str(e)}")
        
        finally:
            # Re-enable button
            self.analyze_button.config(state='normal', text="START ANALYSIS")

    def display_processed_image(self, processed_image):
        """Display the processed image with detections"""
        try:
            # Convert to RGB
            image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Resize for display
            height, width = image_rgb.shape[:2]
            max_size = 400
            if width > height:
                new_width = max_size
                new_height = int((max_size * height) / width)
            else:
                new_height = max_size
                new_width = int((max_size * width) / height)
            
            image_pil = Image.fromarray(image_rgb).resize((new_width, new_height))
            image_tk = ImageTk.PhotoImage(image_pil)
            
            self.result_label.config(image=image_tk, text="")
            self.result_label.image = image_tk  # Keep a reference
            
        except Exception as e:
            self.status_var.set(f"Error displaying result: {str(e)}")

def main():
    """Main function to run the application"""
    try:
        root = tk.Tk()
        app = SimpleYOLOApp(root)
        root.mainloop()
    except Exception as e:
        print(f"Application error: {str(e)}")
        messagebox.showerror("Application Error", f"Failed to start application: {str(e)}")

if __name__ == "__main__":
    main()