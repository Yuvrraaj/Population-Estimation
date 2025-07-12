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
        self._add_stat(stats_frame, "Residential Buildings:", 'residential_count', "#ff0088")
        self._add_stat(stats_frame, "Non-Residential Buildings:", 'non_residential_count', "#00a5ff")
        self._add_stat(stats_frame, "Total Detections:", 'total_count', "#ffffff")

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

    def choose_image(self):
        self.file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")])
        if self.file_path:
            self.file_info.config(text=f"Selected: {os.path.basename(self.file_path)}")
            self.status_var.set("Set value, click Enter to update, then Analyze to proceed")

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
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid positive number for people per residential building.")

    def submit_and_process(self):
        if self.file_path:
            self.status_var.set("Analyzing image...")
            self.root.update_idletasks()
            threading.Thread(target=self.analyze_image, args=(self.file_path,), daemon=True).start()
        else:
            messagebox.showwarning("No Image", "Please upload an image first.")
            messagebox.showwarning("No Image", "Please upload an image first.")

    def analyze_image(self, file_path):
        try:
            original_image = cv2.imread(file_path)
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            original_pil = Image.fromarray(original_rgb).resize((600, 400))
            original_tk = ImageTk.PhotoImage(original_pil)
            self.original_label.config(image=original_tk, text="")
            self.original_label.image = original_tk

            results = model.predict(source=file_path, save=False, save_txt=False, imgsz=640)[0]
            processed_image = original_image.copy()
            residential_count = 0
            non_residential_count = 0
            estimated_population = 0

            for box in results.boxes:
                cls = int(box.cls[0])
                confidence = float(box.conf[0])
                x_center, y_center, w, h = box.xywh[0]
                x1, y1 = int(x_center - w / 2), int(y_center - h / 2)
                x2, y2 = int(x_center + w / 2), int(y_center + h / 2)
                color = colors.get(cls, (255, 255, 255))
                cv2.rectangle(processed_image, (x1, y1), (x2, y2), color, 2)
                label = f"{class_names.get(cls, str(cls))} ({confidence:.2f})"
                cv2.putText(processed_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                if cls == 0:
                    residential_count += 1
                    area = w * h
                    multiplier = 1 if area < 800 else 2 if area < 1500 else 3
                    estimated_population += self.people_per_building.get() * multiplier
                elif cls == 1:
                    non_residential_count += 1
            estimated_population = math.ceil(estimated_population)
            total = residential_count + non_residential_count
            self.pop_value.config(text=str(estimated_population))
            self.residential_count.config(text=str(residential_count))
            self.non_residential_count.config(text=str(non_residential_count))
            self.total_count.config(text=str(total))

            processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            processed_pil = Image.fromarray(processed_rgb).resize((600, 400))
            processed_tk = ImageTk.PhotoImage(processed_pil)
            self.result_label.config(image=processed_tk, text="")
            self.result_label.image = processed_tk

            self.create_charts(residential_count, non_residential_count)
            self.status_var.set(f"Analysis complete. {total} buildings found.")

        except Exception as e:
            self.status_var.set("Error during processing")
            messagebox.showerror("Error", str(e))

    def create_charts(self, residential_count, non_residential_count):
        for widget in self.chart_container.winfo_children():
            widget.destroy()

        fig = Figure(figsize=(6, 4), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        labels = ['Residential', 'Non-Residential']
        values = [residential_count, non_residential_count]
        colors_list = ['#ff0088', '#00a5ff']

        ax1.pie(values, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors_list)
        ax1.set_title("Pie Chart")
        ax1.axis('equal')

        ax2.bar(labels, values, color=colors_list)
        ax2.set_title("Bar Chart")
        ax2.set_ylabel("Count")

        canvas = FigureCanvasTkAgg(fig, master=self.chart_container)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernYOLOApp(root)
    root.mainloop()
