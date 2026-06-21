import cv2
import time
import os
from ultralytics import YOLO

def main():
    # --- KONFIGURASI ---
    # Ganti index kamera jika pakai DroidCam (biasanya 0 atau 1)
    CAMERA_INDEX = 0 
    
    # Path model kamu
    MODEL_PATH = 'Hasil_Skripsi/percobaan_1/weights/best.pt'
    
    # Minimal yakin 60% baru boleh muncul (Filter "Halu")
    CONF_THRESHOLD = 0.60 
    # -------------------

    print(f"🚀 Memuat model: {MODEL_PATH}...")
    try:
        model = YOLO(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error memuat model: {e}")
        return

    # Ambil nama kelas langsung dari file model
    class_names = model.names
    print(f"✅ Daftar Kelas Model: {class_names}")

    # Buka Kamera dengan DirectShow (Wajib buat Windows)
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    
    # Paksa resolusi HD (Supaya mirip dataset training)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Cek apakah kamera terbuka
    if not cap.isOpened():
        print("❌ Kamera tidak terdeteksi! Coba ganti CAMERA_INDEX ke 1 atau 2.")
        return

    print("-" * 50)
    print("🎥 KAMERA AKTIF!")
    print("⌨️  Tekan 's' untuk SIMPAN FOTO (Snapshot) buat dicek.")
    print("⌨️  Tekan 'q' untuk KELUAR.")
    print("-" * 50)

    while True:
        start_time = time.time()
        success, frame = cap.read()
        
        if not success:
            print("❌ Gagal membaca frame kamera.")
            break

        # --- LANGKAH 1: PREDIKSI ---
        # conf=CONF_THRESHOLD: Filter keyakinan rendah
        # agnostic_nms=True: Cegah kotak menumpuk (double detection)
        results = model(frame, conf=CONF_THRESHOLD, verbose=False, agnostic_nms=True)

        # --- LANGKAH 2: VISUALISASI & LOGGING ---
        annotated_frame = frame.copy()
        
        # Cek apakah ada deteksi?
        deteksi_found = False
        
        for r in results:
            boxes = r.boxes
            for box in boxes:
                deteksi_found = True
                
                # Ambil data
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                label_name = class_names[cls_id]

                # LOG KE TERMINAL (PENTING!)
                # Perhatikan angka conf-nya. Kalau tinggi (>0.8) tapi salah label,
                # berarti fix masalah kualitas kamera (Domain Shift).
                print(f"👀 Deteksi: {label_name.upper()} | Yakin: {conf:.2f}")

                # Tentukan Warna (BGR format untuk OpenCV)
                # Layak = Hijau, Penyok = Biru Langit, Sobek = Merah
                if label_name == 'layak_edar':
                    color = (0, 255, 0)
                elif label_name == 'penyok':
                    color = (255, 191, 0) 
                else:
                    color = (0, 0, 255)

                # Gambar Kotak
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                # Gambar Label
                text = f"{label_name} {conf:.2f}"
                (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + w, y1), color, -1)
                cv2.putText(annotated_frame, text, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # --- LANGKAH 3: TAMPILKAN ---
        # Hitung FPS
        fps = 1.0 / (time.time() - start_time)
        cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        cv2.imshow("Debug Mode - Skripsi (Tekan 'q' keluar, 's' simpan)", annotated_frame)

        # --- LANGKAH 4: INPUT KEYBOARD ---
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'): # Quit
            break
        elif key == ord('s'): # Snapshot
            # Simpan 2 gambar: Asli (tanpa kotak) dan Hasil (ada kotak)
            timestamp = int(time.time())
            filename_raw = f"DEBUG_Raw_{timestamp}.jpg"
            filename_res = f"DEBUG_Result_{timestamp}.jpg"
            
            cv2.imwrite(filename_raw, frame)           # Gambar murni dari kamera
            cv2.imwrite(filename_res, annotated_frame) # Gambar dengan deteksi
            
            print(f"\n📸 SNAPSHOT DISIMPAN!")
            print(f"   1. {filename_raw} (Cek ini: Buram/Gelap gak?)")
            print(f"   2. {filename_res} (Hasil deteksinya)\n")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()