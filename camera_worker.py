"""
Kamera Isleyici - her kamera icin ayri thread
"""

import os
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# RTSP'yi TCP uzerinden tasi. FFMPEG varsayilani UDP'dir; paket kaybinda
# goruntu bozulur, yarim kareler gelir ve model bozuk goruntude sacmalar.
# stimeout: kamera susarsa FFMPEG sonsuza kadar beklemesin (5 sn, mikrosaniye).
# cv2 ICE AKTARILMADAN ONCE ayarlanmali - sonrasinda etkisi olmaz.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "rtsp_transport;tcp|stimeout;5000000")

import cv2  # noqa: E402

from detector import CLS_HEAD, CLS_HELMET, CLS_UNKNOWN

COLORS = {CLS_HELMET: (0, 200, 0), CLS_HEAD: (0, 0, 255),
          CLS_UNKNOWN: (140, 140, 140)}
NAMES = {CLS_HELMET: "baretli", CLS_HEAD: "BARETSIZ",
         CLS_UNKNOWN: "?"}


class CameraWorker(threading.Thread):
    def __init__(self, cam_cfg, detector, cfg, db):
        super().__init__(daemon=True)
        self.cam_id = cam_cfg["id"]
        self.cam_name = cam_cfg.get("name", cam_cfg["id"])
        self.url = cam_cfg["url"]
        # Kafa tespiti icin bu kameraya ozel mesafe on-ayari (yakin/normal/
        # uzak) - Detector.kamera_ayarla() gercek ayarlari uygular, burada
        # sadece panelde gosterebilmek icin saklanir.
        self.mesafe = cam_cfg.get("mesafe", "normal")
        self.det = detector
        self.cfg = cfg
        self.db = db

        self.running = False
        self.connected = False
        self.latest_frame = None
        # Isaretlemesiz (kutu/yazi cizilmemis) ham goruntu - panelde
        # 'Kamera' goruntuleme modu icin. Tespit hala calisir, sadece
        # kullanicinin gordugu goruntu farkli.
        self.latest_frame_raw = None
        self._lock = threading.Lock()
        self.stats = {"helmet": 0, "no_helmet": 0, "frames": 0,
                      "fps": 0.0, "gecikme": 0.0}
        self.last_error = None

        d = cfg["detection"]
        # Hedef fps KAMERA BASINA ezilebilir. Yogun bir giris kapisi 4 fps
        # isteyebilir, sakin bir kose 1 fps yeterlidir. Kamera girdisinde
        # target_fps varsa o kullanilir, yoksa genel ayar.
        self.target_fps = float(cam_cfg.get("target_fps",
                                            d.get("target_fps", 3)))
        self.tracking = d.get("tracking_enabled", True)
        # Kacan kisiyi teshis ederken belirsiz (?) kutulari gormek sart:
        # "kisi hic bulunmadi" ile "kisi bulundu ama kafasi eslesmedi"
        # ancak boyle ayirt edilir.
        self.show_unknown = d.get("show_unknown", False)
        self.show_debug = d.get("show_debug", True)
        # Akis bu kadar sure kare vermezse baglanti kopmus sayilir ve
        # yeniden baglanilir. RTSP bazen hata vermeden susar.
        self.stall_seconds = d.get("stall_seconds", 20.0)

        s = cfg["storage"]
        self.snap_dir = Path(s.get("snapshots_dir", "ihlaller"))
        self.snap_dir.mkdir(parents=True, exist_ok=True)
        self.snap_mode = s.get("snapshot_mode", "both")

        self._times = deque(maxlen=30)

    def run(self):
        self.running = True
        while self.running:
            try:
                self._stream()
            except Exception as e:
                self.last_error = str(e)
                self.connected = False
                print(f"[{self.cam_id}] Hata: {e} - 5 sn sonra tekrar")
                time.sleep(5)

    def stop(self):
        self.running = False

    def _src_fps(self, cap):
        """Kaynak fps'i guvenli araliga cek.

        RTSP kameralar CAP_PROP_FPS icin sacma degerler dondurebiliyor
        (0, 90000, 1000...). Bu deger dogrudan kare atlamaya girdigi icin
        yanlis olursa sistem ya donar ya da hicbir kareyi atlamaz.
        """
        try:
            f = float(cap.get(cv2.CAP_PROP_FPS))
        except Exception:
            f = 0.0
        if not (1.0 <= f <= 120.0):
            print(f"[{self.cam_id}] CAP_PROP_FPS={f} makul degil, 25 varsayiliyor")
            f = 25.0
        return f

    def _stream(self):
        cap = cv2.VideoCapture(self.url)
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        if not cap.isOpened():
            raise RuntimeError(f"Akis acilamadi: {self.url}")

        self.connected = True
        self.last_error = None

        src_fps = self._src_fps(cap)
        skip = max(1, int(round(src_fps / self.target_fps)))
        gercek = src_fps / skip
        print(f"[{self.cam_id}] Baglandi: {self.cam_name}  "
              f"kaynak {src_fps:.0f} fps -> her {skip}. kare islenecek "
              f"({gercek:.2f} fps, saniyede {src_fps - gercek:.0f} kare atlanir)")
        # Sabit nesne bastirmanin sure hesabi bu kameranin GERCEK isleme
        # hizina gore yapilmali; kameralar farkli fps'te calisabiliyor.
        try:
            self.det.set_camera_fps(self.cam_id, gercek)
        except AttributeError:
            pass
        is_file = not str(self.url).lower().startswith("rtsp")

        try:
            while self.running:
                # Atlanan kareleri DECODE ETME. grab() sadece kareyi alir,
                # renk donusumu ve Mat ayirma yapmaz. Olculdu: 1080p60'ta
                # kare basina 125 ms -> 62 ms (2 kat). Canli akista bu fark
                # darbogazi koddan cikarip GPU'ya birakir.
                # SADECE okuma suresini olc. Donma nobetcisi kare ALMA
                # suresine bakmali, dongunun tamamina degil - yoksa cikarim
                # yavasladiginda (cok kamera / yuklu GPU) saglikli baglanti
                # bosuna kopartilir. Testte tam bu oldu: 3 kamera 2 cekirdegi
                # paylasinca dongu 28 sn'ye cikti ve akis "olmus" sayildi.
                okuma_t0 = time.time()
                atla_ok = True
                for _ in range(skip - 1):
                    if not cap.grab():
                        atla_ok = False
                        break
                ret, frame = (False, None)
                if atla_ok:
                    ret, frame = cap.read()
                okuma_suresi = time.time() - okuma_t0

                if not ret:
                    if is_file:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    raise RuntimeError("Akis kesildi")

                # Donma nobetcisi sadece CANLI akisda anlamli: RTSP bazen
                # hata vermeden susar, ret=True gelmeye devam eder ama kare
                # gecikir. Dosyalarda boyle bir durum yok.
                if not is_file and okuma_suresi > self.stall_seconds:
                    raise RuntimeError(
                        f"Kare {okuma_suresi:.0f} sn gecikti (akis takildi)")

                t0 = time.time()
                annotated = self._handle(frame)
                dt = time.time() - t0
                self._times.append(dt)
                avg = sum(self._times) / len(self._times)
                self.stats["fps"] = round(1 / avg, 1) if avg > 0 else 0

                # Canli akista cikarim gercek zamandan yavas kalirsa kareler
                # birikir ve giderek daha eski goruntu islenir. Bunu gorunur yap.
                if not is_file:
                    hedef = 1.0 / max(self.target_fps, 0.1)
                    if dt > hedef * 1.5:
                        self.stats["gecikme"] = round(dt - hedef, 2)
                    else:
                        self.stats["gecikme"] = 0.0

                with self._lock:
                    self.latest_frame = annotated
                    self.latest_frame_raw = frame
                self.stats["frames"] += 1
        finally:
            cap.release()

    def _handle(self, frame):
        # cam_id sart: Detector tum kameralarca paylasiliyor, takip ve baret
        # hafizasi kamera basina ayrilmali.
        detections, violations = self.det.process(
            frame, track=self.tracking, cam_id=self.cam_id)
        img = frame.copy()

        n_h = n_n = n_u = 0
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cls = d["cls"]
            if cls is None:
                cls = CLS_UNKNOWN
            if cls == CLS_HELMET:
                n_h += 1
            elif cls == CLS_HEAD:
                n_n += 1
            else:
                n_u += 1
                # Belirsiz (?) tespitler normalde gorsel gurultu; ama
                # show_unknown acikken cizilir - kacan kisi teshisi icin.
                if not self.show_unknown:
                    continue

            col = COLORS.get(cls, (140, 140, 140))
            thick = 3 if cls == CLS_HEAD else 2
            cv2.rectangle(img, (x1, y1), (x2, y2), col, thick)

            # Kafa kutusunu da goster (ince cizgi) - eslesmeyi gormek icin
            if d.get("head_box"):
                hx1, hy1, hx2, hy2 = d["head_box"]
                cv2.rectangle(img, (hx1, hy1), (hx2, hy2), col, 1)

            lab = NAMES.get(cls, "?")
            if d["confirmed_helmet"]:
                lab += "*"          # baret hafizasindan geliyor
            if d["track_id"] > 0:
                lab += f" #{d['track_id']}"

            (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(img, (x1, y1 - th - 7), (x1 + tw + 6, y1), col, -1)
            cv2.putText(img, lab, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

        for v in violations:
            self._log(frame, img, v)

        self.stats["helmet"] += n_h
        self.stats["no_helmet"] += n_n

        h, w = img.shape[:2]
        bar_h = 62 if self.show_debug else 40
        cv2.rectangle(img, (0, 0), (w, bar_h), (0, 0, 0), -1)
        col = (0, 0, 255) if n_n > 0 else (0, 255, 0)
        cv2.putText(img,
                    f"{self.cam_name}  |  baretli: {n_h}  |  BARETSIZ: {n_n}  |  "
                    f"belirsiz: {n_u}",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        # Tarih/saat kaynak goruntudeki (varsa) kucuk/soluk OSD yazisina
        # bagimli kalmasin diye BURADA, buyuk ve net, sistemin kendi
        # saatiyle ayrica basiliyor - saga hizali, ayni ust cubukta.
        zaman = datetime.now().strftime("%d.%m.%Y  %H:%M:%S")
        (zw, _), _ = cv2.getTextSize(zaman, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.putText(img, zaman, (w - zw - 12, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2)

        if self.show_debug:
            # kafa  = kafa modelinin buldugu kutu (head_first'te = kisi sayisi)
            # kisi  = COCO kisi modelinin buldugu kutu
            # govde = kafasi gercek bir COCO kutusuyla eslesenler
            # kucuk = min_person_h/w filtresine takilip atilanlar
            # kafa_px / yaricap: otomatik kalibrasyonun o kamera icin olctugu
            # olcek. Yeni bir kameraya gecince once bunlara bak - kafa_px
            # 6'nin altindaysa kamera cok uzak, model zorlanir.
            db = getattr(self.det, "last_debug", {}) or {}
            cv2.putText(img,
                        f"kafa:{db.get('kafa', 0)}  "
                        f"kisi:{db.get('kisi', 0)}  "
                        f"govde:{db.get('govde', 0)}  "
                        f"kucuk:{db.get('kucuk', 0)}  |  "
                        f"kafa_px:{db.get('kafa_px', 0)}  "
                        f"yaricap:{db.get('yaricap', 0)}  "
                        f"cakisma:{db.get('cakisma', 0)}  "
                        f"sabit:{db.get('sabit', 0)}  "
                        f"insan:{db.get('destek', 0)}  "
                        f"iyilestir:{db.get('iyilestir', 0)}",
                        (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 220, 255), 1)
        return img

    def _log(self, raw, annotated, v):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = f"{self.cam_id}_{ts}_t{v['track_id']}"
        x1, y1, x2, y2 = v["bbox"]

        snap = crop = None
        if self.snap_mode in ("full", "both"):
            snap = f"{base}_full.jpg"
            cv2.imwrite(str(self.snap_dir / snap), annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 85])
        if self.snap_mode in ("crop", "both"):
            pad = 40
            h, w = raw.shape[:2]
            c = raw[max(0, y1 - pad):min(h, y2 + pad),
                    max(0, x1 - pad):min(w, x2 + pad)]
            if c.size:
                if c.shape[0] < 220:
                    s = 220 / c.shape[0]
                    c = cv2.resize(c, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
                crop = f"{base}_crop.jpg"
                cv2.imwrite(str(self.snap_dir / crop), c,
                            [cv2.IMWRITE_JPEG_QUALITY, 92])

        self.db.add_violation(
            camera_id=self.cam_id, camera_name=self.cam_name,
            track_id=v["track_id"], confidence=round(v["conf"], 3),
            snapshot=snap, crop=crop, bbox=f"{x1},{y1},{x2},{y2}")
        print(f"[{self.cam_id}] IHLAL  track={v['track_id']}  conf={v['conf']:.2f}")

    def get_jpeg(self, quality=70, width=None, ham=False):
        """width verilirse kucultulmus kare doner.

        Panelde kucuk onizlemeler icin: 1920 genislikte bir MJPEG akisi
        kare basina ~200 KB. Dort kamerayi ayni anda tam cozunurlukte
        akitmak saniyede ~1.6 MB eder ve uzaktan izlemede tikanir.
        Onizlemeler 320 px'e inince bu yuk ~%95 duser.

        ham=True: kutu/etiket/ust bilgi cubugu cizilmemis TEMIZ goruntu
        dondurur - panelin 'Kamera' goruntuleme modu icin. Tespit yine de
        arka planda calismaya devam eder, sadece gosterilen goruntu degisir.
        """
        with self._lock:
            kaynak = self.latest_frame_raw if ham else self.latest_frame
            if kaynak is None:
                return None
            f = kaynak.copy()
        if width and f.shape[1] > width:
            oran = width / f.shape[1]
            f = cv2.resize(f, (width, max(1, int(f.shape[0] * oran))),
                           interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return buf.tobytes() if ok else None

    def status(self):
        return {
            "id": self.cam_id, "name": self.cam_name,
            "connected": self.connected, "fps": self.stats["fps"],
            "frames": self.stats["frames"],
            "helmet": self.stats["helmet"], "no_helmet": self.stats["no_helmet"],
            "gecikme": self.stats.get("gecikme", 0.0),
            "error": self.last_error,
            "mesafe": self.mesafe,
            "hassasiyet": round(self._hassasiyet_oku()),
            "show_debug": self.show_debug,
        }

    def _hassasiyet_oku(self):
        """Su an gecerli head_conf esigini 0-100 hassasiyet degerine
        cevirir - panelde kaydiricinin dogru konumda acilmasi icin."""
        from detector import conf_to_hassasiyet
        return conf_to_hassasiyet(self.det.cam_head_conf(self.cam_id))