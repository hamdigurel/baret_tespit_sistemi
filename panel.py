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

workers = {}
for cam in CFG["cameras"]:
    if cam.get("enabled") and cam.get("url"):
        workers[cam["id"]] = CameraWorker(cam, detector, CFG, db)

# Video uzerindeki teknik hata-ayiklama satiri (kafa/kisi/govde sayaclari,
# kalibrasyon degerleri vb.) ISG gorevlisi icin anlamsiz gorsel gurultu -
# panel HER ACILISTA kapali baslasin ki ekran temiz/profesyonel gorunsun.
# Sadece bu oturum icin gecerli (config.yaml'a yazilmaz) - istersen
# panelden 'Teknik Gorunum' butonuyla acarsin, bir sonraki acilista yine
# varsayilan olarak kapali baslar.
for _w in workers.values():
    _w.show_debug = False

if not workers:
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
    w = workers.get(cam_id)
    q = quality or CFG["web"].get("stream_quality", 70)
    delay = 1 / max(1, CFG["detection"].get("target_fps", 3))
    while w:
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
    return [w.status() for w in workers.values()]


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
        eski = workers.get(cam_id)
        if eski:
            eski.stop()   # eski akis (varsa) kapanir, thread kendini sonlandirir

        cam_cfg = {"id": cam_id, "name": name, "url": url, "enabled": True}
        if k.target_fps:
            cam_cfg["target_fps"] = k.target_fps
        if k.mesafe in ("yakin", "normal", "uzak"):
            cam_cfg["mesafe"] = k.mesafe

        # Kamera basina tespit ayarlarini (mesafe on-ayari) paylasilan
        # Detector'a hemen bildir - yeniden baslatma gerekmez.
        detector.kamera_ayarla(cam_cfg)

        yeni = CameraWorker(cam_cfg, detector, CFG, db)
        yeni.start()
        workers[cam_id] = yeni

        kameralar = CFG.setdefault("cameras", [])
        for c in kameralar:
            if c.get("id") == cam_id:
                c.clear()
                c.update(cam_cfg)
                break
        else:
            kameralar.append(cam_cfg)
        _config_kaydet()

    return {"ok": True, "kamera": yeni.status()}


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
        CFG["cameras"] = [c for c in CFG.get("cameras", []) if c.get("id") != cam_id]
        _config_kaydet()
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
