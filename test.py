from ultralytics import YOLO
import os
import glob
import time

def main():
    # --- KONFIGURASI ---
    path_model = 'Hasil_Skripsi/percobaan_1/weights/best.pt'
    folder_test = 'test/images'
    folder_output = 'Hasil_Skripsi'
    nama_project = 'hasil_test_full' # Folder baru khusus hasil full
    # -------------------

    # 1. Cek Model
    if not os.path.exists(path_model):
        print(f"❌ Error: Model tidak ditemukan di {path_model}")
        return
    
    print(f"🚀 Memuat model dari: {path_model}")
    model = YOLO(path_model)

    # 2. Ambil SEMUA gambar
    print(f"📂 Membaca gambar dari: {folder_test}")
    # Mengambil ekstensi jpg, jpeg, dan png
    list_gambar = glob.glob(os.path.join(folder_test, '*.jpg')) + \
                  glob.glob(os.path.join(folder_test, '*.jpeg')) + \
                  glob.glob(os.path.join(folder_test, '*.png'))
    
    jumlah_gambar = len(list_gambar)
    print(f"📸 Ditemukan {jumlah_gambar} gambar. Siap melakukan testing...")

    if jumlah_gambar == 0:
        print("❌ Tidak ada gambar ditemukan! Cek path folder test kamu.")
        return

    # 3. Mulai Prediksi (Inference)
    t_start = time.time()
    
    # Kita tidak perlu loop manual, ultralytics bisa terima list atau folder langsung
    # stream=True agar hemat memori jika gambarnya ada ribuan
    results = model.predict(
        source=folder_test, # Langsung arahkan ke folder, otomatis baca semua
        conf=0.25,          # Minimal keyakinan 25%
        save=True,          # Simpan gambar hasil deteksi
        save_txt=True,      # (Opsional) Simpan label hasil deteksi ke txt juga
        project=folder_output,
        name=nama_project,
        device=0,           # Pakai RTX 4060
        exist_ok=True       # Timpa jika folder sudah ada
    )

    t_end = time.time()
    durasi = t_end - t_start
    
    print("-" * 50)
    print(f"✅ SELESAI! Berhasil mengetes {jumlah_gambar} gambar.")
    print(f"⏱️ Waktu total: {durasi:.2f} detik ({durasi/jumlah_gambar:.4f} detik/gambar)")
    print(f"📂 Cek hasilnya di folder: D:/dataset1/{folder_output}/{nama_project}")
    print("-" * 50)

if __name__ == '__main__':
    main()