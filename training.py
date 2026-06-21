from ultralytics import YOLO
import torch

def main():
    # 1. Cek GPU RTX 4060
    print("-" * 50)
    if torch.cuda.is_available():
        print(f"✅ SUKSES: Menggunakan GPU {torch.cuda.get_device_name(0)}")
    else:
        print("❌ WARNING: GPU tidak terbaca! Cek instalasi CUDA.")
    print("-" * 50)

    # 2. Load Model
    # Menggunakan YOLOv10 Small (Seimbang untuk Skripsi)
    model = YOLO('yolov10s.pt') 

    # 3. Mulai Training
    model.train(
        data='data.yaml',
        epochs=100,         # 100 putaran
        imgsz=640,          # Ukuran gambar standar
        batch=16,           # Aman untuk VRAM 8GB RTX 4060
        device=0,           # Pakai NVIDIA GPU
        workers=8,          # Pakai 8 Core i9 kamu untuk loading data
        project='Hasil_Skripsi', # Nama folder output utama
        name='percobaan_1',      # Nama folder hasil training ini
        exist_ok=True       # Timpa jika folder sudah ada (biar gak error)
    )

if __name__ == '__main__':
    main()