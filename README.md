# AppPresensiku - Face Recognition Attendance (Flask + CNN)

## Gambaran Sistem
Sistem ini memproses dataset wajah, melatih model CNN, lalu melakukan absensi otomatis dari kamera atau upload gambar.

Alur end-to-end:
1. Capture dataset dari website Laravel ke folder `dataset/Dataset_Raw/<nama_siswa>/`.
2. Preprocess (satu langkah): deteksi wajah, crop, resize ke 224x224, lalu augmentasi ke folder `dataset/Dataset_Preprocessed`.
3. Training CNN: langsung dari `dataset/Dataset_Preprocessed` dengan split internal per kelas (default 70/15/15).
4. Inference absensi: kamera mendeteksi wajah -> prediksi identitas -> simpan absensi ke MySQL.

## Struktur Proyek
- `app/api.py`: API Flask (pipeline training + endpoint absensi)
- `app/config.py`: konfigurasi terpusat dari environment variable
- `app/database.py`: koneksi DB + helper query absensi
- `app/services/preprocess.py`: preprocess + augmentasi dalam satu file
- `app/services/train_cnn.py`: training model CNN
- `app/services/inference.py`: load model + prediksi wajah
- `app/services/attendance.py`: aturan bisnis absensi
- `app/services/camera_absensi.py`: loop kamera realtime

## Setup Cepat
1. Install dependency:
   - `pip install -r requirements.txt`
2. Salin env:
   - `copy .env.example .env`
3. Sesuaikan kredensial MySQL dan mapping tabel Laravel di `.env`.
4. Jalankan API:
   - `python app/api.py`

## Endpoint API
- `GET /health`
- `POST /pipeline/run`
  - body JSON opsional:
    - `preprocess` (bool)
    - `train` (bool)
    - `overwrite` (bool)
    - `epochs` (int)
    - `min_images_per_class` (int)
    - `train_ratio` (float, default 0.70)
    - `val_ratio` (float, default 0.15)
    - `test_ratio` (float, default 0.15)
- `POST /attendance/recognize`
  - multipart file field: `image`
- `POST /attendance/camera/start`
- `POST /attendance/camera/stop`
- `GET /attendance/camera/status`

## Contoh Menjalankan Pipeline Training
Request:
```json
{
  "preprocess": true,
  "train": true,
  "overwrite": false,
  "min_images_per_class": 30,
  "train_ratio": 0.70,
  "val_ratio": 0.15,
  "test_ratio": 0.15,
  "epochs": 20
}
```

## Integrasi Laravel
Agar mulus dengan sistem teman Anda:
1. Laravel simpan dataset ke `dataset/Dataset_Raw/<nama_siswa>/`.
2. Laravel trigger endpoint `POST /pipeline/run` saat admin selesai upload dataset.
3. Saat absensi aktif, backend Flask jalankan `POST /attendance/camera/start`.
4. Flask tulis ke tabel absensi MySQL yang sama dengan Laravel.

## Catatan Penting Produksi
- Jalankan API Flask di service terpisah (misal PM2/Gunicorn + reverse proxy).
- Simpan model versi per training agar bisa rollback.
- Tambahkan anti-spoofing (liveness) untuk keamanan lebih baik.
- Untuk skala besar, pindah dari Haar Cascade ke detector yang lebih robust (MTCNN/RetinaFace).
