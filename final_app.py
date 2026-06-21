import customtkinter as ctk
import cv2
from PIL import Image
from ultralytics import YOLO
import threading
import time
from collections import deque

# --- KONFIGURASI TAMPILAN ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

# --- KONFIGURASI MODEL ---
CONF_DEFAULT = 0.7
VOTE_WINDOW = 7
LOCK_THRESHOLD = 3

CLASS_NAMES = {
    0: "LAYAK",
    1: "PENYOK",
    2: "SOBEK"
}

CLASS_COLORS = {
    0: "#00ff00",
    1: "orange",
    2: "red"
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("SISTEM INSPEKSI KEMASAN (STABIL)")
        self.geometry("1200x720")

        print("⏳ Loading model...")
        self.model = YOLO("Hasil_Skripsi/percobaan_1/weights/best.pt")
        print("✅ Model siap")

        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()

        # --- BUFFER & STATE ---
        self.cls_buffer = deque(maxlen=VOTE_WINDOW)
        self.last_class = None
        self.stable_count = 0
        self.locked_class = None

        # --- GUI ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.sidebar = ctk.CTkFrame(self, width=260, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        ctk.CTkLabel(
            self.sidebar, text="YOLOv10 QC", font=("Arial", 20, "bold")
        ).pack(pady=20)

        self.lbl_status = ctk.CTkLabel(
            self.sidebar, text="STATUS: -",
            font=("Arial", 18, "bold")
        )
        self.lbl_status.pack(pady=20)

        self.lbl_conf = ctk.CTkLabel(
            self.sidebar, text=f"Sensitivitas: {int(CONF_DEFAULT*100)}%"
        )
        self.lbl_conf.pack()

        self.slider_conf = ctk.CTkSlider(
            self.sidebar, from_=0.4, to=0.9,
            command=self.update_conf_text
        )
        self.slider_conf.set(CONF_DEFAULT)
        self.slider_conf.pack(padx=20, pady=10)

        self.video_area = ctk.CTkLabel(self, text="Menunggu Kamera...")
        self.video_area.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)

        self.available_cameras = self.scan_cameras()
        if self.available_cameras:
            self.start_camera(self.available_cameras[-1])

        self.update_video_loop()

    # --- KAMERA ---
    def scan_cameras(self):
        cams = []
        for i in range(5):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cams.append(i)
                cap.release()
        return cams

    def start_camera(self, idx):
        self.cap = cv2.VideoCapture(idx)
        self.is_running = True
        print(f"📷 Camera {idx} aktif")

    # --- UI ---
    def update_conf_text(self, val):
        self.lbl_conf.configure(text=f"Sensitivitas: {int(val*100)}%")

    # --- LOGIKA STABIL ---
    def stabilize_class(self, cls, conf):
        if cls is None:
            self.cls_buffer.clear()
            self.locked_class = None
            self.stable_count = 0
            return None

        self.cls_buffer.append(cls)

        voted = max(set(self.cls_buffer), key=self.cls_buffer.count)

        if voted == self.last_class:
            self.stable_count += 1
        else:
            self.stable_count = 1
            self.last_class = voted

        if self.stable_count >= LOCK_THRESHOLD and conf > self.slider_conf.get():
            self.locked_class = voted

        return self.locked_class

    # --- VIDEO LOOP ---
    def update_video_loop(self):
        if self.is_running and self.cap:
            ret, frame = self.cap.read()
            if ret:
                results = self.model(
                    frame,
                    conf=self.slider_conf.get(),
                    verbose=False,
                    agnostic_nms=False
                )

                best_cls = None
                best_conf = 0

                if results[0].boxes is not None:
                    boxes = results[0].boxes
                    if len(boxes) > 0:
                        idx = boxes.conf.argmax()
                        best_cls = int(boxes.cls[idx])
                        best_conf = float(boxes.conf[idx])

                final_class = self.stabilize_class(best_cls, best_conf)

                annotated = results[0].plot()
                img = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)

                self.video_area.configure(
                    image=ctk.CTkImage(img, size=(900, 540)),
                    text=""
                )

                if final_class is not None:
                    name = CLASS_NAMES[final_class]
                    color = CLASS_COLORS[final_class]
                    self.lbl_status.configure(
                        text=f"STATUS: {name}",
                        text_color=color
                    )
                else:
                    self.lbl_status.configure(text="STATUS: -")

        self.after(15, self.update_video_loop)


if __name__ == "__main__":
    App().mainloop()
