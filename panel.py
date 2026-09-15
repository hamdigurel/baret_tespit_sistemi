"""
BARET TESPIT PANELI
====================
Iki asamali sistem + canli web paneli.

KULLANIM:
    python panel.py

    Panel:  http://localhost:8000

Once config.yaml'i duzenle (kamera adresleri).
Modeli models/best.pt olarak koy.
"""

import asyncio
import subprocess
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

CFG_PATH = Path("config.yaml")
if not CFG_PATH.exists():
    raise SystemExit("config.yaml bulunamadi!")
CFG = yaml.safe_load(CFG_PATH.read_text(encoding="utf-8"))

head_model = Path(CFG["model"]["head_model"])
if not head_model.exists():
    raise SystemExit(
        f"\nKafa modeli bulunamadi: {head_model}\n"
        f"Egitilen best.pt dosyasini models/ klasorune kopyala.\n"
    )

SNAP_DIR = Path(CFG["storage"]["snapshots_dir"])
SNAP_DIR.mkdir(parents=True, exist_ok=True)

print("Modeller yukleniyor...")
from camera_worker import CameraWorker   # noqa: E402
from database import Database            # noqa: E402
from detector import Detector            # noqa: E402

import torch  # noqa: E402
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "yok (CPU)")

detector = Detector(CFG)
db = Database(CFG["storage"]["database"])

# ---- AYNI ANDA CALISACAK KAMERA SAYISI ----
# 0 = sinir yok (tum etkin kameralar ayni anda calisir - eski davranis).
# Kullanici bir sayi girerse (orn. 5), sadece ilk 5 kamera hemen baslar,
# geri kalanlar 'workers' disinda 'bekleyen_kameralar' icinde SIRADA
# bekler - GPU/CPU kapasitesini asmadan kac kameranin ayni anda
# calisacagini kullanicinin kendisi belirleyebilsin diye.
max_aktif_kamera = int(CFG.get("max_aktif_kamera", 0) or 0)

_tum_kameralar = [c for c in CFG.get("cameras", []) if c.get("enabled") and c.get("url")]
if max_aktif_kamera > 0:
    _baslangic_aktif = _tum_kameralar[:max_aktif_kamera]
else:
    _baslangic_aktif = _tum_kameralar

workers = {}
for cam in _baslangic_aktif:
    workers[cam["id"]] = CameraWorker(cam, detector, CFG, db)

# Kapasite disinda kalanlar - henuz CameraWorker'i bile olusturulmadi,
# sadece config bilgileri (id/ad/vb.) tutuluyor. Sirasi gelince
# _kamera_havuzunu_uygula() bunlardan gercek worker yaratip baslatacak.
bekleyen_kameralar = {c["id"]: c for c in _tum_kameralar[len(_baslangic_aktif):]}

# Video uzerindeki teknik hata-ayiklama satiri (kafa/kisi/govde sayaclari,
# kalibrasyon degerleri vb.) ISG gorevlisi icin anlamsiz gorsel gurultu -
# panel HER ACILISTA kapali baslasin ki ekran temiz/profesyonel gorunsun.
# Sadece bu oturum icin gecerli (config.yaml'a yazilmaz) - istersen
# panelden 'Teknik Gorunum' butonuyla acarsin, bir sonraki acilista yine
# varsayilan olarak kapali baslar.
for _w in workers.values():
    _w.show_debug = False

if not workers and not bekleyen_kameralar:
    print("\n! Hicbir kamera etkin degil. config.yaml -> cameras -> enabled: true\n")

# Panelden calisirken kamera ekleme/degistirme icin - config.yaml'a
# ES ZAMANLI erisimi (birden fazla istek ayni anda gelirse) korur.
workers_lock = threading.Lock()


def _config_kaydet():
    """CFG'yi diske yazar - panelden eklenen/degistirilen kameralar
    programi yeniden baslatinca da kalsin diye."""
    CFG_PATH.write_text(
        yaml.safe_dump(CFG, allow_unicode=True, sort_keys=False),
        encoding="utf-8")


def _kamera_havuzunu_uygula():
    """max_aktif_kamera degistiginde (ya da kamera eklenip/silindiginde)
    hangi kameralarin CALISACAGINI, hangilerinin SIRADA bekleyecegini
    yeniden hesaplar. cagiran fonksiyon workers_lock'u zaten tutuyor
    olmali. Sira, config.yaml'daki kamera sirasina gore belirlenir - ilk
    eklenen/tanimlanan kamera once calismaya baslar."""
    global workers, bekleyen_kameralar
    tum_sirali = [c for c in CFG.get("cameras", []) if c.get("enabled") and c.get("url")]
    if max_aktif_kamera > 0:
        aktif_olmali = tum_sirali[:max_aktif_kamera]
    else:
        aktif_olmali = tum_sirali
    aktif_id = {c["id"] for c in aktif_olmali}

    # Kapasite disina cikanlari durdur (thread bir daha baslatilamaz -
    # tekrar aktif olmasi gerekirse asagida YENI bir CameraWorker kurulur)
    for cam_id in list(workers.keys()):
        if cam_id not in aktif_id:
            w = workers.pop(cam_id)
            w.stop()

    # Kapasiteye yeni giren (sirada bekleyen ya da hic worker'i olmayan)
    # kameralari olustur/baslat
    for cam_cfg in aktif_olmali:
        if cam_cfg["id"] not in workers:
            w = CameraWorker(cam_cfg, detector, CFG, db)
            w.show_debug = False
            w.start()
            workers[cam_cfg["id"]] = w

    bekleyen_kameralar = {c["id"]: c for c in tum_sirali if c["id"] not in aktif_id}

app = FastAPI(title="Baret Tespit Sistemi")
STATIC = Path("static")
if STATIC.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.on_event("startup")
def _start():
    for w in workers.values():
        w.start()
    print(f"\n{'='*52}")
    print(f"  Panel:  http://localhost:{CFG['web']['port']}")
    print(f"  Kamera: {len(workers)} aktif")
    print(f"{'='*52}\n")


@app.on_event("shutdown")
def _stop():
    for w in workers.values():
        w.stop()


@app.get("/", response_class=HTMLResponse)
def index():
    p = STATIC / "index.html"
    if not p.exists():
        return HTMLResponse("<h3>static/index.html yok</h3>", status_code=500)
    # Cache-Control YOK ise tarayici bu sayfayi (ozellikle F5 ile) eskiden
    # kalma bir kopyadan gosterebiliyordu - panel guncellendiginde kullanici
    # degisiklikleri hic gormeden "duzelmedi" saniyordu. Panel her acilista
    # zaten tazeden okundugu icin bunu her seferinde tarayiciya da acikca
    # soyluyoruz: hic onbelleklemesin.
    return HTMLResponse(p.read_text(encoding="utf-8"),
                        headers={"Cache-Control": "no-store, must-revalidate"})


def mjpeg(cam_id, quality=None, width=None, ham=False):
    """NOT: worker'i HER DONGUDE workers sozlugunden yeniden okur (bir kez
    disariya alip 'while w:' ile sonsuza kadar donmek yerine). Eskiden bir
    kamera silinip AYNI id ile yeniden eklendiginde, o kamerayi izlemekte
    olan acik bir tarayici sekmesi eski (artik olu) worker nesnesinde
    sonsuza kadar takili kaliyordu - hicbir hata gostermeden, sadece son
    kareyi tekrar tekrar donduruyordu. Simdi kamera silinirse akis hemen
    biter, yeniden eklenirse (ayni id) akis otomatik yeni goruntuye gecer."""
    q = quality or CFG["web"].get("stream_quality", 70)
    delay = 1 / max(1, CFG["detection"].get("target_fps", 3))
    while True:
        w = workers.get(cam_id)
        if not w:
            break
        j = w.get_jpeg(q, width, ham=ham)
        if j:
            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n")
        time.sleep(delay)


@app.get("/stream/{cam_id}")
def stream(cam_id: str, genislik: int = 0, kalite: int = 0, ham: int = 0):
    """genislik: kucultulmus akis (panel onizlemeleri icin).
    Dort kamerayi tam cozunurlukte akitmak saniyede ~1.6 MB; onizlemeler
    320 px'e inince bu yuk ~%95 duser.
    ham=1: panelin 'Kamera' goruntuleme modu - kutu/etiket cizilmemis
    temiz goruntu. Tespit yine arka planda calisir, sadece gorunum degisir."""
    if cam_id not in workers:
        raise HTTPException(404, "Kamera yok")
    if not CFG["web"].get("live_view", True):
        raise HTTPException(403, "Canli goruntu kapali")
    g = genislik if 80 <= genislik <= 1920 else None
    k = kalite if 20 <= kalite <= 95 else None
    return StreamingResponse(mjpeg(cam_id, k, g, ham=bool(ham)),
                             media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/api/cameras")
def api_cams():
    """Calisan kameralarin gercek durumuna ek olarak, kapasite (max_aktif_
    kamera) yuzunden SIRADA bekleyen VE kullanicinin elle DURDURDUGU
    kameralari da doner - panelde bunlar 'Bağlantı yok' degil sirasiyla
    'Sırada bekliyor' / 'Durduruldu' olarak ayirt edilsin diye (bkz.
    index.html: sirada / durduruldu alanlari)."""
    aktif = [w.status() for w in workers.values()]
    sirada = [{
        "id": c["id"], "name": c.get("name", c["id"]), "connected": False,
        "sirada": True, "durduruldu": False, "fps": 0, "frames": 0,
        "helmet": 0, "no_helmet": 0, "gecikme": 0.0, "error": None,
        "mesafe": c.get("mesafe", "normal"), "hassasiyet": 75, "show_debug": False,
    } for c in bekleyen_kameralar.values()]
    kapali = [{
        "id": c["id"], "name": c.get("name", c["id"]), "connected": False,
        "sirada": False, "durduruldu": True, "fps": 0, "frames": 0,
        "helmet": 0, "no_helmet": 0, "gecikme": 0.0, "error": None,
        "mesafe": c.get("mesafe", "normal"), "hassasiyet": 75, "show_debug": False,
    } for c in CFG.get("cameras", []) if not (c.get("enabled") and c.get("url"))]
    return aktif + sirada + kapali


@app.post("/api/cameras/{cam_id}/durdur")
def api_camera_durdur(cam_id: str):
    """Kenar cubugundaki 'Durdur' (⏸) butonu bunu cagirir. Kamerayi config'den
    SILMEZ, sadece enabled:false yapar - worker'i hemen kapatir, GPU/CPU
    yuku biter. 'Baslat' ile istedigi an geri acabilir."""
    with workers_lock:
        for c in CFG.get("cameras", []):
            if c.get("id") == cam_id:
                c["enabled"] = False
                break
        else:
            raise HTTPException(404, "Kamera config.yaml'da yok")
        _config_kaydet()
        # Bu kamera aktifse durdurulmasi bir slot bosaltir - sirada
        # bekleyen varsa devreye girsin diye havuzu yeniden hesapla.
        _kamera_havuzunu_uygula()
    return {"ok": True}


@app.post("/api/cameras/{cam_id}/baslat")
def api_camera_baslat(cam_id: str):
    """Elle durdurulmus bir kamerayi tekrar etkinlestirir. Kapasite
    musaitse hemen calismaya baslar, degilse sirada bekler (diger
    kameralar gibi). NOT: bu, kapasite doluyken zorla ONE GECIRMEZ -
    onun icin bkz. /aktif-yap."""
    with workers_lock:
        bulundu = False
        for c in CFG.get("cameras", []):
            if c.get("id") == cam_id:
                c["enabled"] = True
                bulundu = True
                break
        if not bulundu:
            raise HTTPException(404, "Kamera config.yaml'da yok")
        _config_kaydet()
        _kamera_havuzunu_uygula()
    return {"ok": True}


class AktifYapGirdi(BaseModel):
    onayla: bool = False   # True olmadan, kapasite doluysa SADECE hangi
                           # kameranin durdurulacagini bildirir, hicbir
                           # sey degistirmez - kullanici onaylayinca gercek
                           # islem yapilir. Boylece kimse "diger kamera
                           # birden kapandi" diye saskina donmuyor.


@app.post("/api/cameras/{cam_id}/aktif-yap")
def api_camera_aktif_yap(cam_id: str, g: AktifYapGirdi = AktifYapGirdi()):
    """Panelde secili/sirada bekleyen bir kameranin yanindaki '▶ Aktif Yap'
    butonu bunu cagirir. Sadece enabled:true yapmakla yetinmez (o zaten
    /baslat'in isi) - kapasite (max_aktif_kamera) doluysa, su an aktif
    olan kameralarin EN SONDAKI (config sirasinda en dusuk oncelikli)
    olanini otomatik olarak siraya geri gonderip yerine bunu gecirir.
    Boylece kullanici "otomatik ilk-N kamera calisir" kuralini gecersiz
    kilip, hangi kameranin gorulecegini kendisi secebiliyor.

    onayla=False (ilk cagri) iken kapasite dolu VE bu bir baskasini
    durduracaksa, HICBIR SEY DEGISTIRMEDEN hangi kameranin durdurulacagini
    bildirir (onay_gerekli:true) - frontend kullaniciya sorup, evet derse
    onayla=True ile tekrar cagirir."""
    with workers_lock:
        kameralar = CFG.get("cameras", [])
        idx = next((i for i, c in enumerate(kameralar) if c.get("id") == cam_id), None)
        if idx is None:
            raise HTTPException(404, "Kamera config.yaml'da yok")

        if cam_id in workers:
            return {"ok": True, "aktif": True}   # zaten calisiyor, yapacak bir sey yok

        if max_aktif_kamera > 0 and len(workers) >= max_aktif_kamera:
            aktif_indeksler = [i for i, c in enumerate(kameralar) if c.get("id") in workers]
            son = max(aktif_indeksler)
            if not g.onayla:
                return {"ok": False, "onay_gerekli": True,
                        "durdurulacak_id": kameralar[son]["id"],
                        "durdurulacak_ad": kameralar[son].get("name", kameralar[son]["id"])}
            kameralar[idx]["enabled"] = True
            kameralar[idx], kameralar[son] = kameralar[son], kameralar[idx]
        else:
            kameralar[idx]["enabled"] = True

        _config_kaydet()
        _kamera_havuzunu_uygula()
    return {"ok": True, "aktif": cam_id in workers}


class AktifKameraSayisiGirdi(BaseModel):
    sayi: int   # 0 = sinir yok (tum etkin kameralar ayni anda calisir)


@app.get("/api/ayarlar/aktif-kamera-sayisi")
def api_aktif_kamera_sayisi_oku():
    """Sistem penceresi acildiginda mevcut siniri gostermek icin."""
    toplam = len([c for c in CFG.get("cameras", []) if c.get("enabled") and c.get("url")])
    return {"sayi": max_aktif_kamera, "toplam_kamera": toplam}


@app.post("/api/ayarlar/aktif-kamera-sayisi")
def api_aktif_kamera_sayisi_ayarla(g: AktifKameraSayisiGirdi):
    """Kullanicinin 'Sistem' penceresinden girdigi 'ayni anda calisacak
    kamera sayisi'. 0 = sinirsiz. Degisiklik ANINDA uygulanir (panel
    yeniden baslatilmaz): kapasiteyi asan kameralar sirada bekletilir,
    kapasite artinca sirada bekleyenler config sirasina gore devreye girer."""
    global max_aktif_kamera
    if g.sayi < 0:
        raise HTTPException(400, "sayi negatif olamaz")
    with workers_lock:
        max_aktif_kamera = g.sayi
        CFG["max_aktif_kamera"] = g.sayi
        _config_kaydet()
        _kamera_havuzunu_uygula()
    return {"ok": True, "sayi": max_aktif_kamera,
            "aktif": len(workers), "sirada": len(bekleyen_kameralar)}


def _gpu_bilgisi():
    """nvidia-smi'yi cagirip guncel GPU durumunu okur. Ekstra bir kutuphane
    kurmaya gerek yok - nvidia-smi zaten NVIDIA suruculeriyle birlikte gelir.
    Sistemde GPU yoksa / nvidia-smi bulunamazsa hata mesajiyla doner, panel
    kilitlenmez."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=name,memory.used,memory.total,utilization.gpu,"
             "temperature.gpu,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            encoding="utf-8", timeout=3,
        )
        def _f(deger, varsayilan=0.0):
            """nvidia-smi bazi kartlarda (ozellikle laptop GPU) sayisal
            olmayan bir deger dondurebilir: 'N/A', '[N/A]', '[Not Supported]'
            gibi. Boyle bir deger gelirse cokmek yerine varsayilana don."""
            try:
                return float(deger)
            except (ValueError, TypeError):
                return varsayilan

        # Birden fazla GPU olabilir - her satir bir GPU. Coklu GPU kurulumunda
        # hepsi listelenir, panel simdilik ilkini (index 0) esas alir ama
        # tumunu de dondurur (ileride secim gerekirse hazir olsun).
        gpular = []
        for satir in out.strip().splitlines():
            p = [x.strip() for x in satir.split(",")]
            gpular.append({
                "ad": p[0],
                "vram_kullanilan_mb": _f(p[1]),
                "vram_toplam_mb": _f(p[2], 1.0),   # 0'a bolme olmasin diye
                "gpu_yuzde": _f(p[3]),
                "sicaklik_c": _f(p[4]),
                "guc_w": _f(p[5], None),
                "guc_limit_w": _f(p[6], None),
            })
        return {"var": True, "gpular": gpular}
    except FileNotFoundError:
        return {"var": False, "hata": "nvidia-smi bulunamadi (NVIDIA GPU/surucusu yok mu?)"}
    except Exception as e:
        return {"var": False, "hata": str(e)}


@app.get("/api/sistem")
def api_sistem():
    """Panelin 'Sistem Kullanimi' penceresi bunu cagirir - GPU/VRAM/CPU/RAM
    anlik durumunu doner. Coklu kamera kurulumunda (8-10+ kamera) GPU'nun
    ne kadar dolu oldugunu gormek icin - 12GB VRAM'lik kartta rahat calisip
    calismadigini izlemeye yarar."""
    cpu_ram = {"cpu_yuzde": None, "ram_kullanilan_gb": None, "ram_toplam_gb": None}
    try:
        import psutil
        cpu_ram["cpu_yuzde"] = psutil.cpu_percent(interval=0.15)
        vm = psutil.virtual_memory()
        cpu_ram["ram_kullanilan_gb"] = round(vm.used / (1024 ** 3), 1)
        cpu_ram["ram_toplam_gb"] = round(vm.total / (1024 ** 3), 1)
    except ImportError:
        pass   # psutil kurulu degilse CPU/RAM gosterilmez, GPU yine de gosterilir

    aktif = sum(1 for w in workers.values() if w.connected)
    return {
        "gpu": _gpu_bilgisi(),
        **cpu_ram,
        "kamera_toplam": len(workers),
        "kamera_aktif": aktif,
    }


class TeknikGorunumGirdi(BaseModel):
    acik: bool


@app.post("/api/teknik-gorunum")
def api_teknik_gorunum(g: TeknikGorunumGirdi):
    """Panelin ust cubugundaki 'Teknik Gorunum' anahtari - video uzerindeki
    kafa/kisi/govde sayaclari ve kalibrasyon degerlerini gosterir/gizler.
    Sadece bu oturum icin gecerlidir, config.yaml'a yazilmaz (panel her
    acilista temiz/profesyonel gorunumle baslasin diye)."""
    with workers_lock:
        for w in workers.values():
            w.show_debug = g.acik
    return {"ok": True, "acik": g.acik}


class IdTakipGirdi(BaseModel):
    acik: bool


@app.get("/api/id-takip")
def api_id_takip_durum():
    """Sayfa ilk acildiginda/yenilendiginde buton durumunu senkronlamak icin."""
    return {"acik": detector.reid_enabled}


@app.post("/api/id-takip")
def api_id_takip(g: IdTakipGirdi):
    """Panelin ust cubugundaki 'ID Takip' anahtari - kameralar arasi kimlik
    eslestirmeyi (basit renk-histogrami tabanli) acar/kapatir.

    KAPALIYKEN (varsayilan): sistem bugunku gibi calisir, her kamera kendi
    #track_id numaralarini kullanir, ekstra maliyet yoktur.
    ACIKKEN: ayni kisi baska bir kameraya gectiginde ayni [G..] kimligini
    almaya calisir (garanti degil - bkz. detector.py GlobalReID docstring'i).
    Sadece bu oturum icin gecerlidir, config.yaml'a yazilmaz."""
    detector.reid_enabled = g.acik
    if g.acik:
        detector.reid.reset()   # her acilista temiz bir galeriyle basla
    return {"ok": True, "acik": g.acik}


class KameraGirdi(BaseModel):
    id: str
    name: Optional[str] = None
    url: str
    target_fps: Optional[float] = None
    mesafe: Optional[str] = None   # yakin | normal | uzak - bkz detector.py MESAFE_PRESETLERI


@app.post("/api/cameras")
def api_camera_ekle(k: KameraGirdi):
    """Yeni kamera ekler VEYA mevcut kameranin linkini/adini degistirir
    (ayni id ile gonderilirse). Panelin sag ustundeki 'Kamera / Link'
    penceresi bunu cagirir - RTSP linki degistiginde eski baglanti
    kapanir, yenisiyle aninda yeniden baslar, config.yaml'a da yazilir."""
    cam_id = (k.id or "").strip()
    url = (k.url or "").strip()
    if not cam_id or not url:
        raise HTTPException(400, "id ve url gerekli")
    name = (k.name or cam_id).strip()

    with workers_lock:
        eski = workers.pop(cam_id, None)
        if eski:
            eski.stop()   # eski akis (varsa) kapanir, thread kendini sonlandirir
        bekleyen_kameralar.pop(cam_id, None)   # sirada bekliyorsa da temizle, asagida yeniden hesaplanacak

        cam_cfg = {"id": cam_id, "name": name, "url": url, "enabled": True}
        if k.target_fps:
            cam_cfg["target_fps"] = k.target_fps
        if k.mesafe in ("yakin", "normal", "uzak"):
            cam_cfg["mesafe"] = k.mesafe

        # Kamera basina tespit ayarlarini (mesafe on-ayari) paylasilan
        # Detector'a hemen bildir - yeniden baslatma gerekmez.
        detector.kamera_ayarla(cam_cfg)

        kameralar = CFG.setdefault("cameras", [])
        for c in kameralar:
            if c.get("id") == cam_id:
                c.clear()
                c.update(cam_cfg)
                break
        else:
            kameralar.append(cam_cfg)
        _config_kaydet()

        # max_aktif_kamera kapasitesini asiyorsa bu kamera dogrudan
        # baslatilmaz, siraya alinir - havuzu yeniden hesapla.
        _kamera_havuzunu_uygula()

        yeni = workers.get(cam_id)
        if yeni:
            return {"ok": True, "kamera": yeni.status()}
        return {"ok": True, "sirada": True,
                "kamera": {"id": cam_id, "name": name, "sirada": True}}


class HassasiyetGirdi(BaseModel):
    deger: float   # 0-100, panelin ust tarafindaki kaydirici


@app.post("/api/cameras/{cam_id}/hassasiyet")
def api_hassasiyet(cam_id: str, h: HassasiyetGirdi):
    """Panelin ust tarafindaki hassasiyet kaydiricisi bunu cagirir.
    Kamerayi/akisi YENIDEN BASLATMAZ - sadece o kameranin head_conf
    esigini gunceller, bir sonraki islenen karede gecerli olur. Boylece
    kaydiriciyi suruklerken goruntu kesilmez."""
    if cam_id not in workers:
        raise HTTPException(404, "Kamera yok")
    with workers_lock:
        yeni_conf = detector.cam_hassasiyet_ayarla(cam_id, h.deger)
        for c in CFG.get("cameras", []):
            if c.get("id") == cam_id:
                c["head_conf"] = yeni_conf
                break
        _config_kaydet()
    return {"ok": True, "hassasiyet": h.deger, "head_conf": yeni_conf}


@app.post("/api/cameras/{cam_id}/hassasiyet/sifirla")
def api_hassasiyet_sifirla(cam_id: str):
    """Kaydiricinin yanindaki '↺ Optimuma dondur' butonu bunu cagirir.
    Elle girilen head_conf'u siler, kamera tekrar kendi mesafe on-ayarinin
    (yakin/normal/uzak) varsayilan hassasiyetine doner."""
    if cam_id not in workers:
        raise HTTPException(404, "Kamera yok")
    with workers_lock:
        for c in CFG.get("cameras", []):
            if c.get("id") == cam_id:
                c.pop("head_conf", None)
                detector.kamera_ayarla(c)
                break
        else:
            raise HTTPException(404, "Kamera config.yaml'da yok")
        _config_kaydet()
        yeni_hassasiyet = round(detector.cam_head_conf_hassasiyet(cam_id), 1)
    return {"ok": True, "hassasiyet": yeni_hassasiyet}


@app.delete("/api/cameras/{cam_id}")
def api_camera_sil(cam_id: str):
    with workers_lock:
        w = workers.pop(cam_id, None)
        if w:
            w.stop()
        bekleyen_kameralar.pop(cam_id, None)
        CFG["cameras"] = [c for c in CFG.get("cameras", []) if c.get("id") != cam_id]
        _config_kaydet()
        # Bu kamera aktifse silinmesi bir slot bosaltir - sirada bekleyen
        # varsa devreye girsin diye havuzu yeniden hesapla.
        _kamera_havuzunu_uygula()
    return {"ok": True}


@app.get("/api/violations")
def api_viol(limit: int = 50, offset: int = 0, camera_id: str = None,
             only_unreviewed: bool = False, durum: str = None):
    """durum: bekleyen | onayli | reddedilen | (bos = hepsi)"""
    return db.get_violations(limit=limit, offset=offset, camera_id=camera_id,
                             only_unreviewed=only_unreviewed, durum=durum)


@app.get("/api/kuyruk")
def api_kuyruk():
    """Inceleme kuyrugu sekmelerindeki sayilar."""
    return db.durum_sayilari()


@app.get("/api/bildirimler")
def api_bildirimler(since: str = "", pencere_dk: int = 15):
    """Panelin ust tarafinda cikacak 'akilli' bildirimler.

    Ayni kisi (kamera+track_id) bareti takana kadar tekrar tekrar
    tespit edilse bile, pencere_dk dakikada bir defadan fazla
    bildirilmez - bkz. Database.bildirim_adaylari(). `since` bos
    verilirse (ilk cagri) hicbir eski kayit "yeni" sayilmaz, sadece
    bundan sonraki gercekten yeni olaylar doner - boylece panel
    acilir acilmaz eski kuyruk bildirim selinie yol acmaz.
    """
    if not since:
        since = datetime.now().isoformat(timespec="seconds")
    return db.bildirim_adaylari(since, pencere_dk)


@app.get("/api/summary")
def api_sum(days: int = 7):
    return db.get_summary(days)


@app.get("/api/stats/hourly")
def api_hourly(hours: int = 24):
    return db.get_hourly_stats(hours)


@app.post("/api/violations/{vid}/review")
def api_review(vid: int, valid: bool = True, note: str = None):
    """Onayla/Reddet karari. Onaylanan kayit normal sekilde isaretlenir
    ve ihlaller/ klasorunde kalir. Reddedilen kayit ise (operatorun
    acikca istedigi uzere) goruntusuyle birlikte ANINDA SILINIR - sadece
    gercek/onaylanmis ihlallerin izi birikir, yanlis alarmlar hic
    saklanmaz."""
    if valid:
        db.mark_reviewed(vid, valid, note)
    else:
        db.reddi_sil(vid, SNAP_DIR)
    return {"ok": True}


@app.post("/api/violations/{vid}/geri")
def api_geri(vid: int):
    """Karari geri al - kayit bekleyen kuyruguna doner."""
    db.geri_al(vid)
    return {"ok": True}


@app.post("/api/violations/temizle-hepsi")
def api_temizle_hepsi(onay: str = ""):
    """TUM kayitlari (bekleyen+onaylanan+reddedilen) siler.

    Panelin 'Tumunu Temizle' butonu buraya bagli - guvenlik icin
    `onay=SIL` gonderilmeden calismaz (yanlislikla tetiklenmesin diye,
    frontend zaten kullaniciya yaziyla onaylatiyor)."""
    if onay != "SIL":
        raise HTTPException(400, "onay=SIL parametresi gerekli")
    n = db.tumunu_temizle(SNAP_DIR)
    return {"ok": True, "silinen": n}


@app.get("/snapshot/{filename}")
def snapshot(filename: str):
    p = SNAP_DIR / Path(filename).name
    if not p.exists():
        raise HTTPException(404, "Goruntu yok")
    return FileResponse(p)


@app.on_event("startup")
async def _cleanup():
    async def loop():
        while True:
            await asyncio.sleep(6 * 3600)
            try:
                n = db.cleanup(CFG["storage"].get("retention_days", 90), SNAP_DIR)
                if n:
                    print(f"[Bakim] {n} eski kayit silindi")
            except Exception as e:
                print("[Bakim] Hata:", e)
    asyncio.create_task(loop())


if __name__ == "__main__":
    uvicorn.run(app, host=CFG["web"]["host"], port=CFG["web"]["port"],
                log_level="warning")
