from ultralytics import YOLO
import cv2
import math

def main():
    # 1. Load Model
    path_model = 'Hasil_Skripsi/percobaan_1/weights/best.pt'
    model = YOLO(path_model)

    # --- PERBAIKAN 1: Ambil nama kelas langsung dari model (Anti-Salah) ---
    classNames = model.names
    print(f"✅ Kelas yang dikenali model: {classNames}")

    # 2. Buka Webcam
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(3, 1280)
    cap.set(4, 720)

    while True:
        success, img = cap.read()
        if not success:
            break

        # --- PERBAIKAN 2: Konversi BGR ke RGB ---
        # OpenCV pakai BGR, YOLO butuh RGB biar warnanya akurat
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # 3. Prediksi
        results = model(img_rgb, stream=True, conf=0.5, verbose=False)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Ambil ID Kelas
                cls_id = int(box.cls[0])
                currentClass = classNames[cls_id] # Panggil dari dictionary model
                conf = math.ceil((box.conf[0] * 100)) / 100

                # Logika Warna: Hijau jika Layak, Merah jika Cacat
                # Perhatikan: nama kelas harus PERSIS sama dengan di data.yaml
                if currentClass == 'layak_edar':
                    color = (0, 255, 0) 
                else:
                    color = (0, 0, 255)

                # Gambar di frame ASLI (img), bukan img_rgb
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)

                label = f'{currentClass} {conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1, y1), c2, color, -1, cv2.LINE_AA)  
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

        cv2.imshow('Demo Skripsi Fix', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()