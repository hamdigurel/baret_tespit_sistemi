# ============================================================
# BARET TESPIT SISTEMI - Docker imaji
# ============================================================
# TABAN IMAJ: resmi PyTorch imaji, icinde CUDA 12.1 + cuDNN + torch
# ZATEN GPU'YA GORE DERLENMIS halde geliyor. Boylece pip'in torch'u
# yanlislikla CPU surumuyle degistirmesi riski ortadan kalkiyor.
#
# CUDA SURUMU HAKKINDA ONEMLI NOT:
#   Bu imaj CUDA 12.1 RUNTIME'ini kendi icinde tasir - hedef makinede
#   ayrica CUDA kurulu olmasina GEREK YOK. Tek sart: hedef makinenin
#   NVIDIA EKRAN KARTI SURUCUSU yeterince guncel olsun (CUDA 12.1'i
#   destekleyen bir surucu - Ekim 2023 sonrasi surucu surumleri genelde
#   yeterlidir). Surucu daha eskiyse (sadece CUDA 11.x destekliyorsa),
#   asagidaki FROM satirini "pytorch/pytorch:2.4.0-cuda11.8-cudnn9-runtime"
#   ile degistirip yeniden derle (docker build) - baska hicbir sey
#   degismez. Hangisinin gerektigini ogrenmek icin hedef makinede
#   "nvidia-smi" calistir, sag ust kosede yazan "CUDA Version" o
#   makinenin DESTEKLEDIGI EN YUKSEK surumdur (kurulu olan degil).
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

WORKDIR /app

# OpenCV (opencv-python, headless olmayan surum) bu sistem
# kutuphaneleri olmadan calismaz. ffmpeg RTSP/video okumak icin.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# --no-deps DEGIL: ultralytics'in kendi bagimliliklarini kursun ama
# torch zaten taban imajda GPU'ya gore kurulu oldugu icin pip onu
# TEKRAR KURMAYA calismaz (surum zaten tatmin ediyor) - GPU destegi
# boylece korunur.
RUN pip install --no-cache-dir -r requirements.txt

# Kod dosyalari kopyalanir. Video dosyalari, egitim ciktilari, model
# agirliklari, config.yaml ve veritabani BILEREK KOPYALANMAZ (bkz.
# .dockerignore) - bunlar "docker run -v" ile disaridan baglanir,
# boylece: (1) imaj kucuk kalir (birkac GB'lik video yuklenmez),
# (2) config/model degistiginde imaji YENIDEN DERLEMEK GEREKMEZ.
COPY . .

EXPOSE 8000

CMD ["python", "panel.py"]
