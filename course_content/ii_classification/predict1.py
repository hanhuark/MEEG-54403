import cv2
import numpy as np
import tensorflow as tf
import serial
import serial.tools.list_ports
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import os

class RobotControlApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Controller")

        # Serial communication
        self.serialInst = serial.Serial()
        self.arduino_connected = False

        # Model
        self.model = None

        # Camera
        self.cap = None
        self.camera_index = 0

        # Setup GUI
        self.setup_gui()

        # Start updating camera
        self.update_frame()

    def setup_gui(self):
        # Top frame for settings
        top_frame = tk.Frame(self.root)
        top_frame.pack(pady=10)

        tk.Label(top_frame, text="Select COM Port:").grid(row=0, column=0, padx=5)
        self.combobox_ports = ttk.Combobox(top_frame, values=self.get_ports(), width=10)
        self.combobox_ports.grid(row=0, column=1, padx=5)

        self.button_connect = tk.Button(top_frame, text="Connect", command=self.connect_arduino)
        self.button_connect.grid(row=0, column=2, padx=5)

        tk.Label(top_frame, text="Select Camera:").grid(row=1, column=0, padx=5)
        self.combobox_cameras = ttk.Combobox(top_frame, values=[0, 1, 2, 3], width=10)
        self.combobox_cameras.current(0)
        self.combobox_cameras.grid(row=1, column=1, padx=5)
        self.combobox_cameras.bind("<<ComboboxSelected>>", self.change_camera)

        tk.Label(top_frame, text="Select Model:").grid(row=2, column=0, padx=5)
        self.model_dir = './class_models'
        self.combobox_models = ttk.Combobox(top_frame, values=self.get_model_list(), width=10)
        self.combobox_models.grid(row=2, column=1, padx=5)

        self.button_load_model = tk.Button(top_frame, text="Load Model", command=self.load_model)
        self.button_load_model.grid(row=2, column=2, padx=5)

        # Middle frame for live and captured image
        middle_frame = tk.Frame(self.root)
        middle_frame.pack(pady=10)

        self.label_camera = tk.Label(middle_frame)
        self.label_camera.pack(side=tk.LEFT, padx=10)

        captured_frame = tk.Frame(middle_frame)
        captured_frame.pack(side=tk.LEFT, padx=10)

        self.label_captured = tk.Label(captured_frame)
        self.label_captured.pack()

        self.label_captured_prediction = tk.Label(captured_frame, text="Prediction: None", font=("Helvetica", 14))
        self.label_captured_prediction.pack(pady=10)

        # Bottom frame for capture button
        bottom_frame = tk.Frame(self.root)
        bottom_frame.pack(pady=10)

        self.button_capture = tk.Button(bottom_frame, text="Capture and Predict", font=("Helvetica", 12), command=self.capture_and_predict)
        self.button_capture.pack()

    def get_ports(self):
        ports = serial.tools.list_ports.comports()
        return [str(port.device) for port in ports]

    def connect_arduino(self):
        port = self.combobox_ports.get()
        if port:
            self.serialInst.baudrate = 9600
            self.serialInst.port = port
            self.serialInst.open()
            self.arduino_connected = True
            print(f"Connected to {port}")

    def get_model_list(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        models = [file.split('_')[-1].replace('.keras', '') for file in os.listdir(self.model_dir) if file.endswith('.keras')]
        return models

    def load_model(self):
        last_name = self.combobox_models.get()
        if last_name:
            model_path = os.path.join(self.model_dir, f"nutsandboltsmodel_{last_name}.keras")
            self.model = tf.keras.models.load_model(model_path)
            print(f"Loaded model: {model_path}")

    def change_camera(self, event):
        new_index = int(self.combobox_cameras.get())
        if new_index == self.camera_index:
            return

        new_cap = cv2.VideoCapture(new_index)
        ret, _ = new_cap.read()
        if ret:
            if self.cap is not None:
                self.cap.release()
            self.cap = new_cap
            self.camera_index = new_index
            print(f"Switched to camera {new_index}")
        else:
            print(f"Failed to open camera {new_index}")
            new_cap.release()

    def update_frame(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_index)

        ret, frame = self.cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame_rgb)
            img = img.resize((400, 400))
            imgtk = ImageTk.PhotoImage(image=img)
            self.label_camera.imgtk = imgtk
            self.label_camera.configure(image=imgtk)

        self.root.after(10, self.update_frame)

    def capture_and_predict(self):
        if self.cap is None or not self.cap.isOpened():
            print("Camera not available!")
            return
        if self.model is None:
            print("Model not loaded!")
            return

        ret, frame = self.cap.read()
        if ret:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            height, width = frame_gray.shape
            print(height,width)
            left = (width - 360) // 2
            cropped_image = frame_gray[60:420, left:left+360]
            resized_image = cv2.resize(cropped_image, (300, 300))

            normalized_image = resized_image / 255.0
            normalized_image = np.expand_dims(normalized_image, axis=-1)
            input_image = np.expand_dims(normalized_image, axis=0)

            prediction = self.model.predict(input_image)
            confidence = float(prediction[0][0])

            print(f"Prediction Raw: {confidence}")

            if confidence > 0.5:
                prediction_text = "Nut"
                if self.arduino_connected:
                    self.serialInst.write(b'1\n')
            else:
                prediction_text = "Bolt"
                if self.arduino_connected:
                    self.serialInst.write(b'2\n')

            # Update captured image display
            img = Image.fromarray(cropped_image)
            img = img.resize((400, 400))
            imgtk = ImageTk.PhotoImage(image=img)
            self.label_captured.imgtk = imgtk
            self.label_captured.configure(image=imgtk)

            # Update prediction label
            self.label_captured_prediction.config(
                text=f"Prediction: {prediction_text} ({confidence:.2f})"
            )
        else:
            print("Failed to capture frame.")

# Main
root = tk.Tk()
app = RobotControlApp(root)
root.mainloop()
