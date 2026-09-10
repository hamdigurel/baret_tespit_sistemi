"""
train_finetune.py
------------------
Amac: Mevcut "models/best.pt" (kafa/baret modelin) uzerinden devam ederek
(fine-tune), yeni etiketlenmis + oversample edilmis veriyle YOLO11s modelini
GPU'nda egitir. Sifirdan egitmiyoruz - onceki ogrendiklerini koruyarak
uzerine ekliyoruz.

ONEMLI - calistirmadan once:
  1) oversample_head.py'yi calistirmis olman lazim (train klasoru
     genisletilmis olmali).
  2) Bilgisayarinda ultralytics kurulu olmali:
         pip install ultralytics
  3) NVIDIA GPU + CUDA'li PyTorch kurulu olmali (torch.cuda.is_available()
     True donmeli, panel.py'yi acarken de zaten bunu kontrol ediyorduk).

Kullanim (ornek):
    python train_finetune.py ^
        --data "C:\\Users\\90545\\Desktop\\helmet-detection-shipyard.v6-ppebaret.yolov11\\data.yaml" ^
        --weights "C:\\Users\\90545\\Desktop\\baret_tespit_sistemi\\models\\best.pt" ^
        --epochs 100 --imgsz 1280 --batch 8

Egitim bitince en iyi agirliklar su klasore duser:
    runs/detect/<name>/weights/best.pt
Bu dosyayi models/best.pt ile DEGISTIRMEDEN once eskisini yedekle
(orn. models/best_eski.pt olarak kopyala), sonra yenisini kopyala ve
paneli yeniden baslat. Kodda baska hicbir degisiklik gerekmiyor.
"""
import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Export edilen datasetin data.yaml yolu")
    ap.add_argument("--weights", required=True, help="Mevcut models/best.pt yolu (bu agirliklardan devam edilecek)")
    ap.add_argument("--epochs", type=int, default=100, help="Egitim tur sayisi (varsayilan 100)")
    ap.add_argument("--imgsz", type=int, default=1280, help="Egitim goruntu boyutu (varsayilan 1280 - kucuk kafalar icin yuksek tutuldu)")
    ap.add_argument("--batch", type=int, default=8, help="Batch boyutu (4GB VRAM icin 8 guvenli baslangic, OOM alirsan 4'e dusur)")
    ap.add_argument("--device", default="0", help="GPU id (varsayilan 0), CPU icin 'cpu' yaz")
    ap.add_argument("--project", default="runs/detect", help="Ciktilarin kaydedilecegi ana klasor")
    ap.add_argument("--name", default="baret_finetune", help="Bu egitimin klasor adi")
    ap.add_argument("--patience", type=int, default=20, help="Bu kadar epoch boyunca iyilesme olmazsa erken durdur")
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch

    print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "YOK (CPU) - bu cok yavas olur, devam etmeden once GPU/CUDA kurulumunu kontrol et!")

    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise SystemExit(f"HATA: {weights_path} bulunamadi. --weights yolunu kontrol et.")

    data_path = Path(args.data)
    if not data_path.exists():
        raise SystemExit(f"HATA: {data_path} bulunamadi. --data yolunu kontrol et.")

    model = YOLO(str(weights_path))  # mevcut best.pt'den devam

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        exist_ok=True,
        # close_mosaic varsayilan (10) kaliyor -> son 10 epoch mosaic kapanir,
        # kucuk nesne tespiti icin bu Ultralytics'in kendi onerisi.
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("EGITIM BITTI.")
    print(f"En iyi agirliklar: {args.project}/{args.name}/weights/best.pt")
    print("Simdi yapman gerekenler:")
    print("  1) models/best.pt dosyasini yedekle (orn. best_eski.pt olarak kopyala)")
    print(f"  2) {args.project}/{args.name}/weights/best.pt dosyasini models/best.pt olarak kopyala")
    print("  3) Paneli yeniden baslat (Paneli_Baslat.bat)")
    print("=" * 60)


if __name__ == "__main__":
    main()
