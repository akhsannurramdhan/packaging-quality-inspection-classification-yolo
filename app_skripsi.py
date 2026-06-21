import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import threading
import time
from tkinter import filedialog
import os

# --- KONFIGURASI TAMPILAN ---
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("SISTEM INSPEKSI KEMASAN - YOLOv10")
        self.geometry("1200x720")

        # --- LOAD MODEL ---
        print("⏳ Sedang memuat model...")
        try:
            # PASTIKAN PATH INI SESUAI DENGAN LOKASI FILE .PT ANDA
            model_path = 'Hasil_Skripsi/percobaan_1/weights/best.pt'
            
            if os.path.exists(model_path):
                self.model = YOLO(model_path) 
                print(f"✅ Model ditemukan: {model_path}")
            else:
                print(f"⚠️ File model tidak ditemukan di: {model_path}")
                print("   -> Menggunakan model standar 'yolov10n.pt' (jika ada) atau error akan muncul.")
                self.model = YOLO('yolov8n.pt') # Fallback jika model skripsi tidak ada (opsional)

        except Exception as e:
            print(f"Error Critical Model: {e}")
            self.model = None

        # Variabel Global
        self.cap = None
        self.is_running = False
        self.display_mode = "camera" # Opsi: 'camera' atau 'image'
        self.lock = threading.Lock()

        # --- GUI LAYOUT ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # 1. SIDEBAR (Menu Kiri)
        self.sidebar = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar.grid(row=0, column=0, sticky="nsew")

        self.logo = ctk.CTkLabel(self.sidebar, text="INSPEKSI KEMASAN", font=("Arial", 20, "bold"))
        self.logo.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Label Menu
        self.lbl_input = ctk.CTkLabel(self.sidebar, text="Sumber Input:", anchor="w", font=("Arial", 14, "bold"))
        self.lbl_input.grid(row=1, column=0, padx=20, pady=(20, 0))

        # Dropdown Kamera
        self.available_cameras = self.scan_cameras()
        self.cam_menu = ctk.CTkOptionMenu(self.sidebar, values=self.available_cameras, command=self.trigger_change_camera)
        self.cam_menu.grid(row=2, column=0, padx=20, pady=10)

        # Tombol Upload
        self.btn_upload = ctk.CTkButton(self.sidebar, text="📂 Upload Gambar", command=self.upload_image, fg_color="green", font=("Arial", 12, "bold"))
        self.btn_upload.grid(row=3, column=0, padx=20, pady=10)

        # Tombol Live Camera
        self.btn_live = ctk.CTkButton(self.sidebar, text="📹 Live Kamera", command=self.reset_to_camera, fg_color="#1f6aa5", font=("Arial", 12, "bold"))
        self.btn_live.grid(row=4, column=0, padx=20, pady=5)

        # 2. LAYAR UTAMA (Video Area)
        self.video_area = ctk.CTkFrame(self, corner_radius=0)
        self.video_area.grid(row=0, column=1, sticky="nsew")
        
        # Label Gambar/Video
        self.lbl_video = ctk.CTkLabel(self.video_area, text="Menunggu Input...", font=("Arial", 24))
        self.lbl_video.pack(expand=True, fill="both", padx=10, pady=10)

        # --- STARTUP ---
        # Mulai Loop Update Video
        self.update_video_loop()

        # Otomatis Pilih Kamera Terakhir (Biasanya Webcam Eksternal/DroidCam ada di index terakhir)
        if self.available_cameras and self.available_cameras[0] != "Tidak Ada Kamera":
            best_cam = self.available_cameras[-1] 
            self.cam_menu.set(best_cam)
            self.trigger_change_camera(best_cam)

    def scan_cameras(self):
        """Mencari kamera yang aktif (Index 0-5)"""
        print("🔍 Scanning kamera...")
        found = []
        for i in range(5): 
            cap = cv2.VideoCapture(i) 
            if cap.isOpened():
                ret, _ = cap.read()
                if ret:
                    found.append(f"Camera {i}")
                cap.release()
        return found if found else ["Tidak Ada Kamera"]

    def trigger_change_camera(self, value):
        """Callback saat dropdown kamera dipilih"""
        if value == "Tidak Ada Kamera": return
        
        self.display_mode = "camera" # Paksa mode kamera
        idx = int(value.split()[-1]) # Ambil angka dari string "Camera 1"
        
        # Jalankan di thread terpisah agar GUI tidak macet saat ganti kamera
        threading.Thread(target=self.change_camera_process, args=(idx,)).start()

    def change_camera_process(self, idx):
        """Proses teknis penggantian kamera"""
        with self.lock:
            self.is_running = False
            if self.cap:
                self.cap.release()
                time.sleep(0.5) # Beri jeda sedikit
            
            print(f"Mencoba membuka Camera {idx}...")
            new_cap = cv2.VideoCapture(idx)
            
            if new_cap.isOpened():
                self.cap = new_cap
                self.is_running = True
                print(f"Camera {idx} Berhasil Aktif!")
            else:
                print(f"Gagal membuka Camera {idx}")

    def upload_image(self):
        """Handler tombol Upload Gambar"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
        )
        
        if file_path:
            self.display_mode = "image" # Ubah mode ke image static
            
            # Baca gambar dengan OpenCV
            img = cv2.imread(file_path)
            if img is not None:
                self.process_and_display_image(img)
            else:
                print("Gagal membaca file gambar.")

    def reset_to_camera(self):
        """Handler tombol Live Kamera"""
        self.display_mode = "camera"
        self.lbl_video.configure(text="") # Bersihkan text placeholder

    def process_and_display_image(self, frame):
        """
        Fungsi inti: Deteksi YOLO -> Gambar Kotak -> Tampilkan di GUI.
        Sekarang dilengkapi LOGIKA DINAMIS untuk ukuran font/garis.
        """
        if self.model:
            # --- 1. HITUNG SKALA GAMBAR (Baru) ---
            # Kita bandingkan ukuran gambar saat ini dengan ukuran standar (misal 640px)
            h, w = frame.shape[:2]
            # Jika gambar dari HP (misal 3000px), scale akan jadi sekitar 4.5
            scale = max(h, w) / 640 
            
            # --- 2. TENTUKAN KETEBALAN BERDASARKAN SKALA ---
            # Base line_width=2 dikali skala. Jika gambar besar, garis jadi tebal (misal 10).
            dyn_lw = max(int(2 * scale), 2)  
            # Base font_size=1 dikali skala.
            dyn_fs = max(1 * scale, 1)      

            # --- 3. PARAMETER DETEKSI ---
            # max_det=1 : Tetap pakai ini agar tidak tumpuk
            results = self.model(frame, conf=0.40, iou=0.5, max_det=1, verbose=False, agnostic_nms=True)
            
            # --- 4. VISUALISASI HASIL ---
            # Masukkan dyn_lw dan dyn_fs ke sini
            annotated_bgr = results[0].plot(line_width=dyn_lw, font_size=dyn_fs, labels=True, conf=True)
            
            display_img = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        else:
            display_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # --- RESIZE LOGIC (Agar muat di layar Laptop) ---
        img_pil = Image.fromarray(display_img)
        
        h, w, _ = frame.shape
        aspect_ratio = w / h
        
        # Tinggi tampilan di layar fix 540px
        target_h = 540
        target_w = int(target_h * aspect_ratio)

        img_tk = ctk.CTkImage(light_image=img_pil, dark_image=img_pil, size=(target_w, target_h))
        self.lbl_video.configure(image=img_tk, text="")

    def update_video_loop(self):
        """Looping utama untuk mengambil frame kamera secara real-time"""
        if self.display_mode == "camera":
            if self.is_running and self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    self.process_and_display_image(frame)
                else:
                    self.lbl_video.configure(image=None, text="Kamera Terputus / Tidak Ada Frame")
            
        # Panggil fungsi ini lagi setelah 10ms (sekitar 100 FPS cap)
        self.after(10, self.update_video_loop)

if __name__ == "__main__":
    # Inisialisasi Aplikasi
    app = App()
    # Handle penutupan aplikasi yang bersih
    try:
        app.mainloop()
    except KeyboardInterrupt:
        print("Aplikasi ditutup.")
    finally:
        if app.cap:
            app.cap.release()
        cv2.destroyAllWindows()