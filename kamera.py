import cv2

print("🔍 MENCARI KAMERA (METODE MSMF)...")
print("-" * 40)

# Cek index 0 sampai 3
for i in range(4):
    print(f"Mengakses Index {i}...", end=" ")
    
    # Hapus cv2.CAP_DSHOW, biarkan OpenCV memilih otomatis
    cap = cv2.VideoCapture(i) 
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            print(f"✅ BERHASIL! (Resolusi: {int(cap.get(3))}x{int(cap.get(4))})")
        else:
            print("⚠️ Terbuka tapi Layar Hitam.")
    else:
        print("❌ Gagal / Kosong")
        
    cap.release()

print("-" * 40)
input("Tekan Enter untuk keluar...")