"""
BARET TESPIT MOTORU  (v4 - KAFA ONCELIKLI)
===========================================

NEDEN DEGISTI (olcumle):
  video_test.mp4 uzerinde 5 kare, ayni sahne:

    kafa modeli (best.pt, imgsz=1920, conf=0.25) ....... 58 tespit
    COCO yolo11m (imgsz=1920, conf=0.05, mukerrer temiz)  44 tespit
    COCO yolo11m (imgsz=1920, conf=0.10) ............... 42 tespit
    SAHI tiling (512/0.25) ............................. daha da az

  Kafa modeli kisileri 0.64-0.89 guvenle buluyor; COCO ayni kisileri
  0.05-0.35 guvenle, ustelik mukerrer kutularla buluyor. 20 karelik
  yanlis-alarm taramasinda kafa modeli tarih damgasina / ekipmana HIC
  kutu koymadi (kare basina ort. 11.3 tespit).

  v3'e kadar akis "once kisi, sonra kafa" idi ve require_person_match
  acikken kafa tespitleri COCO onaylamazsa ATILIYORDU. Yani sistemin
  recall'u zayif olan modele kilitlenmisti - panelde 11 kisi yerine
  4 kisi gorunmesinin sebebi buydu.

v4 AKISI (mode: "head_first"):
  1. Kafa modeli tam karede calisir -> head/helmet kutulari
  2. Her kafa = bir kisi. Kafalar dogrudan takip edilir (IoUTracker).
     Olcum: kafalarin 0.5 sn'deki yer degistirmesi medyan 3 px / p95 31 px,
     ayni karedeki en yakin komsu mesafesi medyan 92 px -> 45 px'lik
     eslestirme yaricapi guvenli.
  3. COCO kisi modeli OPSIYONEL (use_person_boxes): sadece govde kutusu
     saglamak icin. Kapatilirsa GPU yuku yariya iner.
  4. BARET HAFIZASI: baret bir kez dogrulanirsa kalici olur

  mode: "person_first" ile v2/v3 davranisina donulebilir.
"""

import threading
import time
from collections import defaultdict, deque

import cv2
import torch

CLS_HEAD = 0        # baretsiz
CLS_HELMET = 1      # baretli
CLS_UNKNOWN = -1    # kisi var ama kafa siniflandirilamadi


class PersonState:
    __slots__ = ("helmet_confirmed", "head_streak", "helmet_votes",
                 "head_votes", "last_seen", "logged_at")

    def __init__(self):
        self.helmet_confirmed = False
        self.head_streak = 0
        self.helmet_votes = 0
        self.head_votes = 0
        self.last_seen = time.time()
        self.logged_at = 0.0


class CameraProfile:
    """Kameraya ozgu olcekleri CALISIRKEN olcer.

    NEDEN: sistem farkli bakis acilarina sahip kameralara tasinacak.
    Piksel cinsinden sabitlenen her deger (takip yaricapi, govde orani)
    yeni bir acida yanlis olur - kamera yakinsa kafalar 12 px degil 60 px
    olur ve 45 px'lik yaricap yetmez.

    Cozum: her seyi OLCULEN kafa boyuna oranla ifade etmek. Uc videoda
    olculdu, oranlar kamera degisse de tutarli:
        gercek hareket p75      : 0.64 - 1.68 kafa boyu
        iki kisi arasi mesafe   : 8.5 - 16.0 kafa boyu (medyan)
    Aradaki bosluk genis; 3.5 kafa boyu hem hareketi yakalar hem komsuya
    bulasmaz. Ustelik ust sinir da olculuyor: yaricap, o kameradaki
    tipik kisi-arasi mesafenin belli bir oranini asamaz.
    """

    __slots__ = ("head_sizes", "nn_dists", "body_w", "body_h", "frames")

    def __init__(self):
        self.head_sizes = deque(maxlen=600)
        self.nn_dists = deque(maxlen=600)
        self.body_w = deque(maxlen=300)
        self.body_h = deque(maxlen=300)
        self.frames = 0

    def observe(self, heads):
        self.frames += 1
        for hd in heads:
            x1, y1, x2, y2 = hd["bbox"]
            self.head_sizes.append(max(x2 - x1, y2 - y1))
        for i, a in enumerate(heads):
            best = None
            for j, b in enumerate(heads):
                if i == j:
                    continue
                d = ((a["cx"] - b["cx"]) ** 2 + (a["cy"] - b["cy"]) ** 2) ** 0.5
                if best is None or d < best:
                    best = d
            if best is not None:
                self.nn_dists.append(best)

    def observe_body(self, head_box, body_box):
        hw = max(1, head_box[2] - head_box[0])
        hh = max(1, head_box[3] - head_box[1])
        bw = body_box[2] - body_box[0]
        bh = body_box[3] - body_box[1]
        if 1.2 <= bw / hw <= 8 and 2.0 <= bh / hh <= 14:   # sacma oranlari ele
            self.body_w.append(bw / hw)
            self.body_h.append(bh / hh)

    @staticmethod
    def _med(dq):
        if not dq:
            return None
        s = sorted(dq)
        return s[len(s) // 2]

    def head_px(self):
        return self._med(self.head_sizes)

    def nn_px(self):
        return self._med(self.nn_dists)

    def body_mults(self, n_min=40):
        if len(self.body_w) < n_min:
            return None
        return self._med(self.body_w), self._med(self.body_h)

    def reach_px(self, mult, floor, nn_ratio, n_min=60):
        """Takip yaricapi = mult x kafa boyu, ama kisi-arasi mesafeyle sinirli."""
        h = self.head_px()
        if h is None or len(self.head_sizes) < n_min:
            return None
        r = mult * h
        nn = self.nn_px()
        if nn:
            r = min(r, nn_ratio * nn)     # komsu kisiyi kapmasin
        return max(floor, r)

    def summary(self):
        h, nn = self.head_px(), self.nn_px()
        bm = self.body_mults()
        return ("kafa=%s  komsu=%s  govde=%s  ornek=%d" % (
            f"{h:.0f}px" if h else "-",
            f"{nn:.0f}px" if nn else "-",
            ("%.1fx%.1f" % bm) if bm else "-",
            len(self.head_sizes)))


class StaticSuppressor:
    """Sabit nesneleri (lamba, reflektor, asili ekipman) eler.

    SORUN: kafa modeli duvardaki kucuk yuvarlak beyaz lambayi baret saniyor.
    Ust bakisli bir baret de zaten kucuk parlak bir lekedir - goruntude
    ayirt etmek zor.

    COZUM: lamba ASLA kipirdamaz. Olculdu (2 fps, gercek kayitlar):

        lamba                     : doluluk 1.00,  maks sapma 0.5 px
        en hareketsiz gercek isci : doluluk 0.68,  maks sapma 7.0 px
        iskelede duran isci       : doluluk 0.24,  maks sapma 0.7 px

    Iki sart BIRLIKTE arandiginda (yuksek doluluk VE sifira yakin sapma)
    hicbir insan filtreye takilamaz. Ustelik pencere uzadikca kural daha
    guvenli olur: insan er gec oradan ayrilir, lamba ayrilmaz.

    Kasitli olarak yavas: bir noktayi sabit ilan etmek icin en az
    min_seconds boyunca gozlemek gerekir. Yanlis bir bastirma, gercek bir
    ihlali gizleyebilecegi icin acele edilmez.
    """

    def __init__(self, radius_px=14.0, min_frames=1200, min_occupancy=0.97,
                 max_dev_px=2.0, forget_frames=3600, log=None,
                 min_frames_unsupported=None, support_ratio=0.15):
        # Sure KARE cinsinden olculur, duvar saatiyle degil. Cevrimdisi
        # denetimde kareler gercek zamandan yavas islenir; kare sayisi
        # her iki durumda da ayni anlama gelir.
        self.radius = radius_px
        self.min_frames = min_frames
        self.min_occupancy = min_occupancy
        self.max_dev_px = max_dev_px
        self.forget_frames = forget_frames
        self.anchors = []
        self.n = 0
        self.log = log        # bir nokta sabit ilan edilince haber ver
        # KISI KANITI: COCO kisi modeli bu noktayi bir insanin kafasi olarak
        # destekliyorsa nokta ASLA bastirilmaz. Desteklemiyorsa daha kisa
        # surede karar verilebilir - iki bagimsiz kanit birlikte kullanilir.
        # Olculdu (video_test6): lamba %0 COCO destegi, gercek isciler
        # %96-100. Ama COCO'nun kacirdigi gercek isciler de var (%0 destek,
        # 18.7 px hareket) - bu yuzden destek yoklugu TEK BASINA yetmez,
        # hareketsizlikle birlikte aranir.
        self.min_frames_unsupported = (min_frames_unsupported
                                       if min_frames_unsupported is not None
                                       else min_frames)
        self.support_ratio = support_ratio

    def _dev(self, a):
        pts = a["pts"]
        if len(pts) < 3:
            return 0.0
        xs = sorted(p[0] for p in pts)
        ys = sorted(p[1] for p in pts)
        mx, my = xs[len(xs) // 2], ys[len(ys) // 2]
        return max(((p[0] - mx) ** 2 + (p[1] - my) ** 2) ** 0.5 for p in pts)

    def update(self, heads, supported=None):
        """heads: kafa tespitleri, supported: hangilerinin COCO kisi destegi var.
        Doner: bastirilacak tespitlerin indeks kumesi."""
        self.n += 1
        supported = supported or set()
        for a in self.anchors:
            a["frames"] += 1

        suppressed = set()
        # Bir nokta ayni karede EN FAZLA BIR kez sayilir. Coklu olcekli
        # tespitte iki kutu ayni noktaya dusebiliyordu ve "hits" iki artiyordu;
        # bu doluluk oranini 1.0'in uzerine cikariyor (olculdu: 1.25) ve
        # kuralin ERKEN tetiklenmesine yol aciyordu - gercek bir insani
        # bastirabilecek bir hata.
        bu_kare = set()
        for i, hd in enumerate(heads):
            cx, cy = hd["cx"], hd["cy"]
            hit = None
            for a in self.anchors:
                if ((cx - a["x"]) ** 2 + (cy - a["y"]) ** 2) ** 0.5 <= self.radius:
                    hit = a
                    break
            if hit is None:
                self.anchors.append({"x": cx, "y": cy, "hits": 1, "frames": 1,
                                     "first": self.n, "last": self.n,
                                     "pts": deque([(cx, cy)], maxlen=400),
                                     "static": False,
                                     "coco": 1 if i in supported else 0})
                continue
            if id(hit) in bu_kare:
                # ayni karede ayni noktaya ikinci tespit - sayma
                if hit["static"]:
                    suppressed.add(i)
                continue
            bu_kare.add(id(hit))
            hit["hits"] += 1
            hit["last"] = self.n
            if i in supported:
                hit["coco"] += 1
            hit["pts"].append((cx, cy))
            n = len(hit["pts"])
            hit["x"] += (cx - hit["x"]) / min(n, 50)
            hit["y"] += (cy - hit["y"]) / min(n, 50)

            sure = self.n - hit["first"]
            doluluk = hit["hits"] / max(hit["frames"], 1)
            destek = hit["coco"] / max(hit["hits"], 1)
            if destek >= self.support_ratio:
                # COCO bunu insan olarak destekliyor - asla bastirma
                continue
            gerekli = self.min_frames_unsupported
            if (not hit["static"]
                    and sure >= gerekli
                    and doluluk >= self.min_occupancy
                    and self._dev(hit) <= self.max_dev_px):
                hit["static"] = True
                # Sessizce bastirma - bir insani yanlislikla elemis olma
                # ihtimaline karsi her karar gorulebilir olmali.
                if self.log:
                    self.log(f"SABIT NESNE: ({hit['x']:.0f},{hit['y']:.0f}) "
                             f"doluluk={doluluk:.2f} "
                             f"sapma={self._dev(hit):.1f}px "
                             f"COCO_destek={destek:.0%} -> bastiriliyor")
            if hit["static"]:
                suppressed.add(i)

        self.anchors = [a for a in self.anchors
                        if self.n - a["last"] <= self.forget_frames]
        return suppressed

    def static_count(self):
        return sum(1 for a in self.anchors if a["static"])


class _NullLock:
    """inference_lock: false icin - hicbir sey yapmayan kilit."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


class IoUTracker:
    """Kucuk kutular icin hafif takipci (kafa kutulari ~10-15 px).

    Bu olcekte kare arasi IoU cogu zaman 0 cikar, bu yuzden asil is
    merkez mesafesi eslestirmesinde. Yaricap kutu boyundan degil, olculen
    hareketten turetilir:
        yer degistirme (0.5 sn) : medyan 3 px, p90 18 px, p95 31 px
        ayni karede en yakin komsu: medyan 92 px
    -> reach_px=45 gercek hareketin ~%95'ini yakalar, komsuya bulasmaz.
    """

    def __init__(self, iou_thr=0.25, max_age=10, reach_px=45.0,
                 reach_box_mult=3.0, max_gap_grow=1.8):
        self.iou_thr = iou_thr
        self.max_age = max_age              # kac islenmis kare hatirlansin
        self.reach_px = reach_px            # taban eslestirme yaricapi
        self.reach_box_mult = reach_box_mult  # buyuk kutularda yaricap kutuyla olcekenir
        self.max_gap_grow = max_gap_grow    # kayip karede yaricap en fazla kac kat
        self.tracks = {}                    # tid -> {"bbox":(...), "age":int}
        self._next_id = 1

    @staticmethod
    def _center(b):
        return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)

    def _base_reach(self, box):
        return max(self.reach_px,
                   max(box[2] - box[0], box[3] - box[1]) * self.reach_box_mult)

    def update(self, boxes):
        """boxes: [(x1,y1,x2,y2), ...]  ->  ayni sirada [track_id, ...]"""
        assigned = [None] * len(boxes)
        used = set()

        # --- 1. tur: IoU (buyuk / yavas kutularda kesin eslesme) ---
        pairs = []
        for di, box in enumerate(boxes):
            for tid, tr in self.tracks.items():
                v = _iou(box, tr["bbox"])
                if v >= self.iou_thr:
                    pairs.append((v, di, tid))
        pairs.sort(reverse=True)
        for _, di, tid in pairs:
            if assigned[di] is None and tid not in used:
                assigned[di] = tid
                used.add(tid)

        # --- 2. tur: merkez mesafesi (kucuk kutularda asil yontem) ---
        cand = []
        for di, box in enumerate(boxes):
            if assigned[di] is not None:
                continue
            cx, cy = self._center(box)
            base = self._base_reach(box)
            for tid, tr in self.tracks.items():
                if tid in used:
                    continue
                tx, ty = self._center(tr["bbox"])
                d = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                # Kisi N karedir kayipsa daha uzaga gitmis olabilir; yaricapi
                # buyut ama sinirla, yoksa komsu kisiyi kapar.
                grow = min(1.0 + 0.4 * tr["age"], self.max_gap_grow)
                if d <= base * grow:
                    cand.append((d, di, tid))
        cand.sort()
        for _, di, tid in cand:
            if assigned[di] is None and tid not in used:
                assigned[di] = tid
                used.add(tid)

        # --- 3. tur: yeni kimlikler ---
        for di in range(len(boxes)):
            if assigned[di] is None:
                assigned[di] = self._next_id
                self._next_id += 1

        # --- yaslandirma ---
        for tid in list(self.tracks):
            if tid in used:
                self.tracks[tid]["age"] = 0
            else:
                self.tracks[tid]["age"] += 1
                if self.tracks[tid]["age"] > self.max_age:
                    del self.tracks[tid]
        for di, tid in enumerate(assigned):
            self.tracks[tid] = {"bbox": boxes[di], "age": 0}

        return assigned


# Kameraya gore "mesafe" on-ayarlari. Panelden kamera eklerken/duzenlerken
# secilen mesafe, kafa tespitinin agresifligini belirler - AYRI AYRI HER
# KAMERA ICIN, tek bir global degerin tum kameralari yavaslatmasi ya da
# yakin kameralarda gereksiz yanlis alarm uretmesi yerine.
#   yakin  : kafa buyuk gorunur (>15px) - tek olcek yeterli, en hizli.
#   normal : varsayilan - iki olcekli tarama, config.yaml'daki model
#            blogundaki degerlerle ayni (kamera listede yoksa/mesafe
#            belirtilmemisse de bu kullanilir).
#   uzak   : sisli/tozlu/cok uzak kameralar icin (kafa <10px). Uc olcekli
#            tarama + dusuk esik - daha yavas ama kucuk kafalari yakalar.
#            Kamera 3 / video_test7 (Hangar Yol Tarafi) bu profille
#            olculdu ve calisti.
MESAFE_PRESETLERI = {
    "yakin":  {"head_imgsz": [1280],             "head_conf": 0.20, "min_head_px": 6},
    "normal": {"head_imgsz": [1280, 1920],       "head_conf": 0.15, "min_head_px": 5},
    "uzak":   {"head_imgsz": [1280, 1920, 2560], "head_conf": 0.12, "min_head_px": 4},
}


# ---- HASSASIYET KAYDIRICISI (panelde 0-100 goruntulenen slider) ----
# Operator "kamera bu koseyi goremiyor" derse hassasiyeti manuel yukseltip
# head_conf esigini dusurebilsin diye. %0 = en az hassas (sadece cok emin
# tespitler), %100 = en hassas (zayif tespitler bile gosterilir, yanlis
# alarm riski artar). Araligin uclari mesafe on-ayarlarindaki (0.12-0.20)
# degerlerin epey disinda tutuldu ki slider gercekten ise yarasin.
HASSASIYET_MIN_CONF = 0.45   # slider = 0   (en az hassas)
HASSASIYET_MAX_CONF = 0.05   # slider = 100 (en hassas)


def hassasiyet_to_conf(deger):
    """0-100 slider degerini head_conf esigine cevirir."""
    d = max(0.0, min(100.0, float(deger))) / 100.0
    return HASSASIYET_MIN_CONF + d * (HASSASIYET_MAX_CONF - HASSASIYET_MIN_CONF)


def conf_to_hassasiyet(conf):
    """head_conf esigini 0-100 slider degerine cevirir (goruntuleme icin)."""
    if HASSASIYET_MAX_CONF == HASSASIYET_MIN_CONF:
        return 50.0
    d = (float(conf) - HASSASIYET_MIN_CONF) / (HASSASIYET_MAX_CONF - HASSASIYET_MIN_CONF)
    return max(0.0, min(100.0, d * 100.0))


def _cam_override_hesapla(cam_cfg):
    """Bir kamera girdisinden (config.yaml'daki ya da panelden gelen)
    o kameraya ozel tespit ayarlarini hesaplar. Once 'mesafe' on-ayari
    uygulanir, sonra kamerada ELLE verilmis tekil degerler (varsa)
    on-ayarin uzerine yazar - ileri duzey kullanim icin."""
    ov = dict(MESAFE_PRESETLERI.get(cam_cfg.get("mesafe", "normal"),
                                    MESAFE_PRESETLERI["normal"]))
    if cam_cfg.get("head_conf") is not None:
        ov["head_conf"] = float(cam_cfg["head_conf"])
    if cam_cfg.get("head_imgsz"):
        hi = cam_cfg["head_imgsz"]
        ov["head_imgsz"] = [int(hi)] if isinstance(hi, (int, float)) else [int(x) for x in hi]
    if cam_cfg.get("min_head_px") is not None:
        ov["min_head_px"] = int(cam_cfg["min_head_px"])
    return ov


class Detector:
    def __init__(self, cfg):
        m = cfg["model"]
        d = cfg["detection"]

        # Kamera basina mesafe on-ayarlari - bkz. MESAFE_PRESETLERI yukarida.
        self._cam_overrides = {}
        for cam in cfg.get("cameras", []):
            if cam.get("id"):
                self._cam_overrides[cam["id"]] = _cam_override_hesapla(cam)

        from ultralytics import YOLO
        self.head_model = YOLO(m["head_model"])
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.mode = m.get("mode", "head_first")   # head_first | person_first
        self.use_person_boxes = m.get("use_person_boxes", True)

        self.head_conf = m.get("head_conf", 0.15)
        # Tek sayi ya da liste olabilir. Liste verilirse model her olcekte
        # calisir ve sonuclar birlestirilir (cok olcekli cikarim).
        # NEDEN: en iyi olcek sahneye gore degisiyor.
        #   video_test.mp4  -> 1920: 15 tespit, 1280: 14
        #   video_test1.mp4 -> 1280: 11 tespit, 1920:  8  (1920'ninkiler alt kume)
        # Tek sabit deger bir sahnede kisi kaybettiriyor; birlesim ikisini de alir.
        hi = m.get("head_imgsz", [1280, 1920])
        self.head_imgsz = [int(hi)] if isinstance(hi, (int, float)) else [int(x) for x in hi]
        self.head_merge_px = m.get("head_merge_px", 12)

        self.person_conf = m.get("person_conf", 0.10)
        self.person_imgsz = m.get("person_imgsz", 1920)

        # KISI KAPISI: kafa tespitinin gercekten bir insana ait oldugunu
        # bagimsiz olarak dogrular. Lamba insan degildir, COCO onu asla
        # kisi olarak isaretlemez.
        self.person_gate = m.get("person_gate", True)
        self.gate_every = max(1, int(m.get("person_gate_every", 5)))
        self._gate_cache = {}          # cam_id -> (sayac, kisi kutulari)

        self.person_model = None
        if self.use_person_boxes or self.person_gate or self.mode == "person_first":
            self.person_model = YOLO(m.get("coco_model", "yolo11m.pt"))

        # ---- SAHI (tiling) - olcumde bu sahnede FAYDA SAGLAMADI ----
        # 512/0.25 -> 6 kisi, duz tespit -> 7-10 kisi. Kapali tutulmali;
        # secenek, farkli/daha uzak sahneler icin duruyor.
        self.sahi_enabled = m.get("sahi_enabled", False)
        self.sahi_slice = m.get("sahi_slice", 640)
        self.sahi_overlap = m.get("sahi_overlap", 0.2)
        self.sahi_pp_type = m.get("sahi_postprocess", "NMS")
        self.sahi_pp_metric = m.get("sahi_postprocess_metric", "IOU")
        self.sahi_pp_thr = m.get("sahi_postprocess_threshold", 0.45)
        self._sahi_model = None
        if self.sahi_enabled:
            from sahi import AutoDetectionModel
            self._sahi_model = AutoDetectionModel.from_pretrained(
                model_type="ultralytics",
                model_path=m.get("coco_model", "yolo11m.pt"),
                confidence_threshold=self.person_conf,
                device=self.device,
            )

        self.head_zone = d.get("head_zone_ratio", 0.55)
        self.match_pad = d.get("match_pad_ratio", 0.50)
        self.min_person_h = d.get("min_person_h", 14)
        self.min_person_w = d.get("min_person_w", 6)
        self.min_head_px = d.get("min_head_px", 5)
        # GUVENLIK FILTRESI: kafa/baret kutusu icin bir UST sinir da olmali.
        # Bu kamera acisinda gercek bir kafa/baret ASLA cerceve alaninin
        # kucuk bir yuzdesinden fazla olamaz (olculen degerler binde birler
        # mertebesinde). Egitim verisinde (Roboflow export) cerceve alaninin
        # %5-91'ini kaplayan bozuk/hatali etiketler bulundu (bkz. denetim) -
        # boyle bir kutu egitime sizmissa model canli akista da benzer dev
        # kutular uretebilir. Bu filtre onu KAYNAKTA (tespit asamasinda) eler,
        # veri setini temizlemekten bagimsiz ikinci bir savunma katmani.
        self.max_head_frac = d.get("max_head_frac", 0.04)

        # head_first modunda kafadan govde tahmini (snapshot kirpmasi icin)
        self.body_w_mult = d.get("body_w_mult", 2.6)
        self.body_h_mult = d.get("body_h_mult", 6.0)

        self.confirm_frames = d.get("confirm_frames", 3)
        self.helmet_memory = d.get("helmet_memory", True)
        self.helmet_votes_needed = d.get("helmet_votes_needed", 2)
        # Baret hafizasi bu kadar ardisik "baretsiz" gozleminden sonra iptal
        # edilir. confirm_frames'ten buyuk tutulmali ki anlik bir titremede
        # hem hafiza silinip hem ayni anda ihlal yazilmasin.
        self.helmet_revoke_frames = d.get("helmet_revoke_frames", 4)
        self.log_revoke = d.get("log_helmet_revoke", True)
        self.cooldown = d.get("cooldown_seconds", 90)
        self.state_ttl = d.get("state_ttl_seconds", 300)
        self.require_person = d.get("require_person_match", True)
        self.show_unmatched_person = d.get("show_unmatched_person", True)

        # DIKKAT: panel.py tek bir Detector olusturup butun CameraWorker
        # thread'lerine verir. Bu yuzden takipci de, baret hafizasi da
        # KAMERA BASINA ayrilmak zorunda - yoksa KAM-02 acildiginda iki
        # kameranin kisileri ayni kimlik havuzunu paylasir ve bir kameradaki
        # baret onayi otekine sizar.
        self._track_cfg = {
            "iou_thr": d.get("track_iou", 0.25),
            "max_age": d.get("track_max_age", 10),
            "reach_px": d.get("track_reach_px", 45.0),
        }
        self._trackers = {}                      # cam_id -> IoUTracker
        self._states = defaultdict(lambda: defaultdict(PersonState))
        self._lock = threading.Lock()

        # Cikarim kilidi: her kamera ayri thread'de calisiyor ve hepsi ayni
        # model nesnesini paylasiyor. predict() olcumde thread-guvenli cikti
        # (2 thread x 6 cikarim, sonuclar tek thread referansiyla ayni),
        # ama 4 GB VRAM'de iki 1920'lik cikarimin ayni anda bellek ayirmasi
        # OOM riski. GPU zaten isi sirayla yapiyor; kilit sadece tepe
        # kullanimi duzlestirir, toplam is yukunu artirmaz.
        self._infer_lock = threading.Lock() if d.get("inference_lock", True) \
            else _NullLock()
        self._last_cleanup = time.time()

        # ---- OTOMATIK KALIBRASYON ----
        # Sistem farkli bakis acilarindaki kameralara tasinacak. Piksel
        # cinsinden sabit degerler yeni acida yanlis olur; bu yuzden takip
        # yaricapi ve govde orani calisirken olculur.
        # ---- CAKISMA COZUMU ----
        # Comelen/egilen iscide model bareti dogru goruyor AMA ayni kisinin
        # koyu renkli sirtini/ensesini ikinci bir "baretsiz kafa" saniyor.
        # O sahte tespit ayri kimlik aldigi icin dogru kutunun baret hafizasi
        # onu korumuyor, uc ardisik karede birikip SAHTE IHLAL uretiyor.
        # (Gozlendi: tek iscide "baretli* #34" + "baretli #3" + "BARETSIZ #62")
        #
        # Kural yapisal: bir kisi ayni anda hem baretli hem baretsiz olamaz.
        # Baretsiz bir tespit, baretli bir tespitle ayni kisiyi gosteriyorsa
        # ihlal sayilmaz - silinmez, "belirsiz" e dusurulur ki operator yine gorsun.
        # Ihlal KAYDI icin ayri (daha yuksek) esik. Tespit esigi dusuk kalir
        # ki kisi ekranda gorunsun, ama zayif tespitler alarm uretmesin.
        # 0.0 = kapali (tespit esigi ne ise o).
        self.violation_min_conf = d.get("violation_min_conf", 0.0)

        self.conflict_resolve = d.get("conflict_resolve", True)
        self.conflict_iou = d.get("conflict_body_iou", 0.45)
        self.conflict_head_mult = d.get("conflict_head_mult", 1.5)
        self.conflict_person_box = d.get("conflict_person_box", True)

        # ---- SABIT NESNE BASTIRMA ----
        self.static_enabled = d.get("static_suppress", True)
        self._varsayilan_fps = max(0.1, float(d.get("target_fps", 2)))
        self._cam_fps = {}          # cam_id -> o kameranin gercek isleme hizi
        self._static_sec = {
            "unsupported": d.get("static_min_seconds_unsupported", 30.0),
            "default": d.get("static_min_seconds", 600.0),
            "forget": d.get("static_forget_seconds", 1800.0),
        }
        _fps = self._varsayilan_fps
        # Sapma esigi KAFA BOYUNA GORE. Piksel sabiti farkli bakis acisinda
        # yanlis olur: yakin kamerada kafalar 60 px olur ve dedektorun kendi
        # titremesi bile 2 pikseli asar, lamba hic elenmez.
        # Olculdu (kafa 12 px iken):
        #     lamba 0.5 px = 0.04 kafa boyu
        #     en hareketsiz gercek isci 7.0 px = 0.58 kafa boyu
        # 0.18 ikisinin ortasinda genis payla duruyor.
        self._static_dev_heads = d.get("static_max_dev_heads", 0.18)
        self._static_dev_floor = d.get("static_max_dev_floor_px", 1.0)
        self._static_cfg = {
            "min_frames": int(d.get("static_min_seconds", 600.0) * _fps),
            # COCO hic insan demediyse daha kisa surede karar verilebilir.
            "min_frames_unsupported": int(
                d.get("static_min_seconds_unsupported", 90.0) * _fps),
            "support_ratio": d.get("static_support_ratio", 0.15),
            "min_occupancy": d.get("static_min_occupancy", 0.97),
            "forget_frames": int(d.get("static_forget_seconds", 1800.0) * _fps),
        }
        self._statics = {}

        self.autocal = d.get("auto_calibrate", True)
        self.cal_reach_mult = d.get("cal_reach_head_mult", 3.5)
        self.cal_reach_floor = d.get("cal_reach_floor_px", 12.0)
        self.cal_nn_ratio = d.get("cal_reach_max_nn_ratio", 0.40)
        self.cal_min_samples = d.get("cal_min_samples", 60)
        self._profiles = defaultdict(CameraProfile)

        # GORUNURLUK IYILESTIRME: sis, toz veya zayif isik yuzunden
        # kontrasti dusuk karelerde kafa/govde modelleri kisiyi kacirir
        # (bkz. "Hangar Yol Tarafi" kamerasi - sisli havada nerdeyse
        # kimse tespit edilmiyordu). Her karenin kontrasti (gri ton
        # std sapmasi) olculur; esigin altindaysa CLAHE (yerel kontrast
        # esitleme) sadece parlaklik (L) kanaline uygulanir, renk
        # degismez. Bu SADECE modele giden kareye uygulanir - kaydedilen
        # foto ve canli yayin orijinal kalir, kutu konumlari ikisinde de
        # aynidir (CLAHE piksel konumunu degistirmez).
        self.auto_enhance = d.get("auto_enhance", True)
        self.enhance_std_esigi = d.get("enhance_contrast_threshold", 42.0)
        self._clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        self.last_debug = {"kafa": 0, "kisi": 0, "govde": 0, "kucuk": 0,
                           "kafa_px": 0, "yaricap": 0, "cakisma": 0, "iyilestir": 0}
        # Cakisma kurali tarafindan bastirilan baretsiz tespitler.
        # Denetim icin disari verilir: kural yanlislikla gercek bir ihlali
        # susturuyorsa ancak bunlara bakarak anlasilir.
        self.last_suppressed = []

    def kamera_ayarla(self, cam_cfg):
        """Panelden bir kamera eklenirken/duzenlenirken cagrilir (bkz.
        panel.py -> /api/cameras). O kameraya ozel mesafe on-ayarini
        ANINDA devreye sokar - Detector tum kameralarca paylasildigi icin
        programi yeniden baslatmaya gerek kalmaz, bir sonraki karede
        gecerli olur."""
        if cam_cfg.get("id"):
            self._cam_overrides[cam_cfg["id"]] = _cam_override_hesapla(cam_cfg)

    def cam_head_conf(self, cam_id):
        """O kameranin su an gecerli olan head_conf esigini dondurur -
        panelde hassasiyet kaydiricisinin baslangic konumunu dogru
        gostermek icin (bkz. camera_worker.status())."""
        ov = self._cam_overrides.get(cam_id)
        return ov["head_conf"] if ov else self.head_conf

    def cam_head_conf_hassasiyet(self, cam_id):
        """cam_head_conf'u dogrudan 0-100 hassasiyet degerine cevirir."""
        return conf_to_hassasiyet(self.cam_head_conf(cam_id))

    def cam_hassasiyet_ayarla(self, cam_id, deger):
        """Panelden hassasiyet kaydiricisi surklendiginde cagrilir. Akisi
        (stream'i) yeniden baslatmadan, bir sonraki karede gecerli olacak
        sekilde SADECE head_conf esigini gunceller."""
        onceki = self._cam_overrides.get(cam_id) or {}
        # mesafe on-ayarindan gelen imgsz/min_head_px korunsun (onceki
        # override zaten cozulmus halleriyle bunlari icerir), sadece
        # head_conf ELLE verilen degerle ezilsin.
        cam_cfg = {
            "id": cam_id,
            "head_imgsz": onceki.get("head_imgsz"),
            "min_head_px": onceki.get("min_head_px"),
            "head_conf": hassasiyet_to_conf(deger),
        }
        self._cam_overrides[cam_id] = _cam_override_hesapla(cam_cfg)
        return cam_cfg["head_conf"]

    def set_camera_fps(self, cam_id, fps):
        """CameraWorker baglandiginda o kameranin gercek isleme hizini bildirir.

        Sabit nesne bastirma SANIYE cinsinden ayarlanir ama KARE sayarak
        calisir. Kameralar farkli fps'te calisabildigi icin cevrim her kamera
        icin ayri yapilmali - yoksa 4 fps'lik bir kamerada 30 saniyelik
        pencere 15 saniyeye duser.
        """
        with self._lock:
            self._cam_fps[cam_id] = max(0.1, float(fps))
            self._statics.pop(cam_id, None)   # yeni fps ile yeniden kurulsun

    def _static_for(self, cam_id, head_px):
        with self._lock:
            st = self._statics.get(cam_id)
            if st is None:
                # yaricap kafa boyuyla olcekenir - farkli kameralarda calissin
                hp = head_px or 12
                fps = self._cam_fps.get(cam_id, self._varsayilan_fps)
                cfg = dict(self._static_cfg)
                cfg["min_frames"] = int(self._static_sec["default"] * fps)
                cfg["min_frames_unsupported"] = int(
                    self._static_sec["unsupported"] * fps)
                cfg["forget_frames"] = int(self._static_sec["forget"] * fps)
                st = StaticSuppressor(
                    radius_px=max(8.0, hp * 1.2),
                    max_dev_px=max(self._static_dev_floor,
                                   hp * self._static_dev_heads),
                    log=lambda m, c=cam_id: print(f"[{c}] {m}"),
                    **cfg)
                self._statics[cam_id] = st
            return st

    def profile(self, cam_id="_"):
        """Kameranin olculen profili - yeni bir kameraya gecerken bak."""
        return self._profiles[cam_id].summary()

    def _tracker_for(self, cam_id):
        with self._lock:
            t = self._trackers.get(cam_id)
            if t is None:
                t = IoUTracker(**self._track_cfg)
                self._trackers[cam_id] = t
            return t

    def _cleanup(self):
        now = time.time()
        if now - self._last_cleanup < 60:
            return
        self._last_cleanup = now
        for cam, st in self._states.items():
            for k in [k for k, v in st.items()
                      if now - v.last_seen > self.state_ttl]:
                del st[k]

    # ------------------------------------------------------------------
    # GORUNURLUK IYILESTIRME (sis / toz / zayif isik)
    # ------------------------------------------------------------------
    def _kontrast_olc(self, frame):
        """Karenin genel kontrastini ucuza olcer (kucultup gri std sapma)."""
        small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        return float(gray.std())

    def _enhance(self, frame):
        """LAB uzayinda sadece L (parlaklik) kanaline CLAHE uygular.
        Renk kanallari (a,b) degismedigi ve piksel konumlari kaymadigi
        icin buradan cikan kutular orijinal karede de gecerlidir."""
        try:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            l2 = self._clahe.apply(l)
            return cv2.cvtColor(cv2.merge((l2, a, b)), cv2.COLOR_LAB2BGR)
        except Exception:
            return frame

    def _tespit_karesi(self, frame):
        """Modele girecek kareyi hazirlar: dusuk kontrastta iyilestirilmis,
        normal kosulda orijinal kare donuk. (kare, iyilestirildi_mi)"""
        if not self.auto_enhance:
            return frame, False
        try:
            if self._kontrast_olc(frame) < self.enhance_std_esigi:
                return self._enhance(frame), True
        except Exception:
            pass
        return frame, False

    # ------------------------------------------------------------------
    # TESPIT
    # ------------------------------------------------------------------
    def _detect_heads_at(self, frame, imgsz, conf=None, min_px=None):
        conf = self.head_conf if conf is None else conf
        min_px = self.min_head_px if min_px is None else min_px
        with self._infer_lock:
            r = self.head_model.predict(frame, conf=conf,
                                        imgsz=imgsz, verbose=False)[0]
        out = []
        if r.boxes is None:
            return out
        fh, fw = frame.shape[:2]
        frame_area = max(1, fh * fw)
        max_area = self.max_head_frac * frame_area
        for b in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            if (x2 - x1) < min_px or (y2 - y1) < min_px:
                continue
            if (x2 - x1) * (y2 - y1) > max_area:
                # Bir kafa/baret kutusu cerceve alaninin bu kadarini
                # kaplayamaz - muhtemelen bozuk egitim etiketinden gelen
                # sahte genis kutu. Sessizce at (kucuk kutuyu attigimizda
                # oldugu gibi - bu da bir boyut sinir filtresi).
                continue
            out.append({
                "bbox": (x1, y1, x2, y2),
                "cls": int(b.cls[0]),
                "conf": float(b.conf[0]),
                "cx": (x1 + x2) / 2,
                "cy": (y1 + y2) / 2,
            })
        return out

    def _merge_heads(self, groups):
        """Farkli olceklerden gelen kafa tespitlerini birlestir.

        Kafa kutulari 8-20 px oldugu icin IoU guvenilmez (bir piksellik
        kayma IoU'yu ucurur); merkez mesafesi kullanilir. Ayni kafa iki
        olcekte de bulunduysa GUVENI YUKSEK olan tutulur - sinif bilgisi
        (baretli/baretsiz) de ondan gelir.
        """
        merged = []
        for g in groups:
            for hd in sorted(g, key=lambda h: -h["conf"]):
                w = hd["bbox"][2] - hd["bbox"][0]
                h = hd["bbox"][3] - hd["bbox"][1]
                lim = max(self.head_merge_px, 0.8 * max(w, h))
                dup = None
                for other in merged:
                    d = ((hd["cx"] - other["cx"]) ** 2
                         + (hd["cy"] - other["cy"]) ** 2) ** 0.5
                    if d <= lim:
                        dup = other
                        break
                if dup is None:
                    merged.append(hd)
                elif hd["conf"] > dup["conf"]:
                    merged[merged.index(dup)] = hd
        return merged

    def _detect_heads(self, frame, cam_id="_"):
        # Bu kamera icin ozel bir mesafe on-ayari varsa (bkz. kamera_ayarla /
        # MESAFE_PRESETLERI) onu kullan, yoksa config.yaml'daki genel
        # (model:) degerlere don.
        ov = self._cam_overrides.get(cam_id)
        imgsz_list = ov["head_imgsz"] if ov else self.head_imgsz
        conf = ov["head_conf"] if ov else self.head_conf
        min_px = ov["min_head_px"] if ov else self.min_head_px
        if len(imgsz_list) == 1:
            return self._detect_heads_at(frame, imgsz_list[0], conf, min_px)
        return self._merge_heads(
            [self._detect_heads_at(frame, s, conf, min_px) for s in imgsz_list])

    def _detect_persons(self, frame, track=True):
        if self.sahi_enabled:
            from sahi.predict import get_sliced_prediction
            res = get_sliced_prediction(
                frame, self._sahi_model,
                slice_height=self.sahi_slice, slice_width=self.sahi_slice,
                overlap_height_ratio=self.sahi_overlap,
                overlap_width_ratio=self.sahi_overlap,
                perform_standard_pred=True,
                postprocess_type=self.sahi_pp_type,
                postprocess_match_metric=self.sahi_pp_metric,
                postprocess_match_threshold=self.sahi_pp_thr,
                verbose=0,
            )
            return [(int(p.bbox.minx), int(p.bbox.miny),
                     int(p.bbox.maxx), int(p.bbox.maxy), None)
                    for p in res.object_prediction_list
                    if p.category.name == "person"]

        # head_first modunda COCO'nun takip kimlikleri KULLANILMIYOR - sadece
        # govde kutusu aliniyor. track(persist=True) ise Detector tum kameralarca
        # paylasildigi icin ByteTrack durumunu kameralar arasi karistiriyor.
        # OLCULDU: ayni model nesnesine iki farkli sahne donusumlu verildiginde
        # ikinci sahne HIC kimlik alamiyor. Bu yuzden burada predict kullanilir.
        if track and self.mode != "head_first":
            r = self.person_model.track(
                frame, classes=[0], conf=self.person_conf,
                imgsz=self.person_imgsz, persist=True,
                tracker="bytetrack.yaml", verbose=False)[0]
        else:
            r = self.person_model.predict(
                frame, classes=[0], conf=self.person_conf,
                imgsz=self.person_imgsz, verbose=False)[0]
        if r.boxes is None:
            return []
        out = []
        for b in r.boxes:
            x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
            tid = int(b.id[0]) if b.id is not None else None
            out.append((x1, y1, x2, y2, tid))
        return out

    # ------------------------------------------------------------------
    def _body_for_head(self, head_box, persons, prof=None):
        """Kafayi iceren en dar kisi kutusunu bul; yoksa kafadan tahmin et.

        Tahmin oranlari (kafa -> govde) bakis acisina bagli: tepeden bakan
        kamerada govde kisalir, goz hizasindaki kamerada uzar. Bu yuzden
        COCO gercek bir govde kutusu verdiginde oran OGRENILIR ve COCO'nun
        bulamadigi kisilerde o kameranin kendi orani kullanilir.
        """
        hx1, hy1, hx2, hy2 = head_box
        hcx, hcy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        best, best_area = None, None
        for (x1, y1, x2, y2, _tid) in persons:
            # kafa, govdenin ust yarisinda ve yatayda icinde olmali
            if not (x1 - 4 <= hcx <= x2 + 4):
                continue
            if not (y1 - (y2 - y1) * 0.20 <= hcy <= y1 + (y2 - y1) * 0.60):
                continue
            area = (x2 - x1) * (y2 - y1)
            if best is None or area < best_area:
                best, best_area = (x1, y1, x2, y2), area
        if best is not None:
            if prof is not None and self.autocal:
                prof.observe_body(head_box, best)
            return best, True

        wm, hm = self.body_w_mult, self.body_h_mult
        if prof is not None and self.autocal:
            learned = prof.body_mults()
            if learned:
                wm, hm = learned
        hw, hh = hx2 - hx1, hy2 - hy1
        bw = hw * wm
        return (int(hcx - bw / 2), hy1,
                int(hcx + bw / 2), int(hy1 + hh * hm)), False

    @staticmethod
    def _in_head_zone(cx, cy, pbox):
        x1, y1, x2, y2 = pbox
        ph = y2 - y1
        return (x1 - 8 <= cx <= x2 + 8
                and y1 - ph * 0.25 <= cy <= y1 + ph * 0.65)

    def _same_person_has_helmet(self, head_det, others, persons):
        """Ayni COCO KISI KUTUSU icinde hem baret hem acik bas var mi?

        Bu, comelen isci hatasinin kesin testi: model bareti dogru goruyor
        ama ayni kisinin koyu sirtini ikinci bir "acik bas" saniyor. Ikisi de
        TEK bir kisi kutusunun icine duser.

        OLCULDU (video_test3_kesit2, gercek kareler):
          sahte ihlal #12 -> COCO kisi kutusunda 2 baretli + 4 BARETSIZ
          gercek ihlal #17 -> hicbir kisi kutusunda ikisi birden yok
        Mesafe olcutu bu ikisini ayiramiyordu (1.77 ve 2.42 kafa boyu,
        bantlar ic ice); kisi kutusu kesin ayiriyor.
        """
        if not persons:
            return False
        for pbox in persons:
            if not self._in_head_zone(head_det["cx"], head_det["cy"], pbox):
                continue
            for od, _ in others:
                if od["cls"] != CLS_HELMET:
                    continue
                if self._in_head_zone(od["cx"], od["cy"], pbox):
                    return True
        return False

    def _conflicting_helmet(self, head_det, head_body, others, persons=None):
        """Bu baretsiz tespit, baretli bir tespitle AYNI kisiyi mi gosteriyor?

        Iki bagimsiz olcut - biri yeterli:
          1. Govde kutulari IoU >= esik. Yan yana duran iki AYRI kisi bu
             kadar ortusmez; ayni kisiye dusen iki kutu ortusur.
          2. Kafa merkezleri birbirine 1.5 kafa boyundan yakin. Olculdu:
             ayni karede iki AYRI kisi arasi mesafe medyani 8.5-16 kafa boyu,
             yani 1.5 esigi gercek komsulara cok uzak - onlari yanlislikla
             susturmaz.
        """
        # 1) EN GUCLU OLCUT: ayni kisi kutusunda hem baret hem acik bas
        if self.conflict_person_box and self._same_person_has_helmet(
                head_det, others, persons):
            return True

        hx1, hy1, hx2, hy2 = head_det["bbox"]
        hs = max(hx2 - hx1, hy2 - hy1)
        lim = self.conflict_head_mult * max(hs, 1)
        for od, obody in others:
            if od["cls"] != CLS_HELMET:
                continue
            d = ((head_det["cx"] - od["cx"]) ** 2
                 + (head_det["cy"] - od["cy"]) ** 2) ** 0.5
            if d <= lim:
                return True
            if _iou(head_body, obody) >= self.conflict_iou:
                return True
        return False

    def _match_head_to_person(self, person_box, heads):
        """person_first modu - v2/v3 davranisi."""
        x1, y1, x2, y2 = person_box
        h, w = y2 - y1, x2 - x1
        zy2 = y1 + h * self.head_zone
        px = w * self.match_pad
        zx1, zx2 = x1 - px, x2 + px
        zy1 = y1 - h * 0.15
        best, best_score = None, -1
        for hd in heads:
            if not (zx1 <= hd["cx"] <= zx2 and zy1 <= hd["cy"] <= zy2):
                continue
            closeness = 1.0 - abs(hd["cy"] - y1) / max(1.0, h)
            score = hd["conf"] * 0.6 + closeness * 0.4
            if score > best_score:
                best, best_score = hd, score
        return best

    # ------------------------------------------------------------------
    def _register(self, states, tid, cls, conf, bbox, head_box, now,
                  detections, violations):
        st = states[tid]
        st.last_seen = now

        if cls == CLS_HELMET:
            st.helmet_votes += 1
            st.head_streak = 0
            if st.helmet_votes >= self.helmet_votes_needed:
                st.helmet_confirmed = True
        elif cls == CLS_HEAD:
            st.head_votes += 1
            st.head_streak += 1
            # BARET HAFIZASI GERI ALINABILIR OLMALI.
            # Eskiden helmet_confirmed bir kez True olunca asla acilmiyordu:
            # model 11 kare ust uste BARETSIZ dese bile ekranda "baretli"
            # yaziyor ve o kisi icin BIR DAHA HIC ihlal uretilmiyordu.
            # Yani bareti cikaran isci sessizce gorunmez oluyordu - urunun
            # asil islevi devre disi kaliyordu.
            if (st.helmet_confirmed
                    and st.head_streak >= self.helmet_revoke_frames):
                st.helmet_confirmed = False
                st.helmet_votes = 0
                if self.log_revoke:
                    print(f"BARET HAFIZASI IPTAL: track={tid} "
                          f"{st.head_streak} kare ust uste baretsiz")
        else:
            st.head_streak = 0

        if self.helmet_memory and st.helmet_confirmed:
            final_cls = CLS_HELMET
        elif cls is None:
            final_cls = CLS_HELMET if st.helmet_votes else CLS_UNKNOWN
        else:
            final_cls = cls

        detections.append({
            "bbox": bbox,
            "head_box": head_box,
            "track_id": tid,
            "cls": final_cls,
            "conf": conf,
            "confirmed_helmet": st.helmet_confirmed,
        })

        if (final_cls == CLS_HEAD
                and conf >= self.violation_min_conf
                and st.head_streak >= self.confirm_frames
                and not st.helmet_confirmed
                and (now - st.logged_at) > self.cooldown):
            st.logged_at = now
            violations.append({"bbox": bbox, "track_id": tid, "conf": conf})

    def process(self, frame, track=True, cam_id="_"):
        """Doner: (tespitler, ihlaller). cam_id kamera basina takip/hafiza icin."""
        self._cleanup()
        now = time.time()
        detections, violations = [], []
        suppressed = []
        states = self._states[cam_id]

        # Sisli/tozlu/az isikli karelerde modele giden kareyi iyilestir.
        # Orijinal 'frame' degismez - cizim ve kayit hep ondan yapilir.
        det_frame, iyilestirildi = self._tespit_karesi(frame)

        heads = self._detect_heads(det_frame, cam_id)
        persons = []
        if self.person_model is not None and (
                self.use_person_boxes or self.mode == "person_first"):
            persons = self._detect_persons(det_frame, track=track)

        prof = self._profiles[cam_id]
        if self.autocal:
            prof.observe(heads)

        # Sabit nesneleri (lamba, reflektor, asili ekipman) ele. Kasitli olarak
        # yavas: bir nokta ancak dakikalarca hic kipirdamadan gorulurse sabit
        # sayilir. Bkz. StaticSuppressor.
        # Kisi kapisi: her kafa icin "COCO burada bir insan goruyor mu"
        # Maliyeti dusurmek icin COCO her karede degil, gate_every karede bir
        # calisir; sabit nesne kararlari zaten dakikalar sureyle verildigi
        # icin ornekleme yeterli.
        destekli = set()
        if self.person_gate and self.person_model is not None:
            sayac, kisiler = self._gate_cache.get(cam_id, (0, []))
            if sayac % self.gate_every == 0:
                if persons:
                    kisiler = [(p[0], p[1], p[2], p[3]) for p in persons]
                else:
                    kisiler = [(p[0], p[1], p[2], p[3])
                               for p in self._detect_persons(det_frame, track=False)]
            self._gate_cache[cam_id] = (sayac + 1, kisiler)
            kapi_kutulari = kisiler
            for i, hd in enumerate(heads):
                for (px1, py1, px2, py2) in kisiler:
                    ph = py2 - py1
                    if (px1 - 6 <= hd["cx"] <= px2 + 6
                            and py1 - ph * 0.25 <= hd["cy"] <= py1 + ph * 0.60):
                        destekli.add(i)
                        break
            dbg_destek = len(destekli)
        else:
            dbg_destek = 0
            kapi_kutulari = [(p[0], p[1], p[2], p[3]) for p in persons]

        n_static = 0
        if self.static_enabled and heads:
            st = self._static_for(cam_id, prof.head_px())
            bastir = st.update(heads, destekli)
            if bastir:
                heads = [h for i, h in enumerate(heads) if i not in bastir]
                n_static = len(bastir)

        dbg = {"kafa": len(heads), "kisi": len(persons),
               "govde": 0, "kucuk": 0,
               "kafa_px": prof.head_px() or 0, "yaricap": 0, "cakisma": 0,
               "sabit": n_static, "destek": dbg_destek,
               "iyilestir": 1 if iyilestirildi else 0}

        if self.mode == "head_first":
            tracker = self._tracker_for(cam_id)
            # Takip yaricapini o kameranin OLCULEN olcegine gore ayarla.
            # Piksel sabiti farkli bakis acisinda yanlis olurdu.
            if self.autocal:
                r = prof.reach_px(self.cal_reach_mult, self.cal_reach_floor,
                                  self.cal_nn_ratio, self.cal_min_samples)
                if r:
                    tracker.reach_px = r
            dbg["yaricap"] = round(tracker.reach_px)

            # Her kafa bir kisidir. Kafalar dogrudan takip edilir.
            tids = tracker.update([h["bbox"] for h in heads])
            used_person = set()

            # Govde kutularini once hesapla - cakisma cozumu bunlara bakiyor
            bodies = []
            for hd in heads:
                body, real = self._body_for_head(hd["bbox"], persons, prof)
                if real:
                    dbg["govde"] += 1
                    used_person.add(body)
                bodies.append(body)

            for i, (hd, tid) in enumerate(zip(heads, tids)):
                cls = hd["cls"]
                if (self.conflict_resolve and cls == CLS_HEAD):
                    others = [(heads[j], bodies[j])
                              for j in range(len(heads)) if j != i]
                    # Kisi kutulari: kapi zaten hesapladi (her N karede bir
                    # tazelenir). Kisi kutulari buyuk oldugu icin birkac
                    # karelik gecikme sonucu degistirmez.
                    pboxes = kapi_kutulari
                    if self._conflicting_helmet(hd, bodies[i], others, pboxes):
                        # Ayni kisi hem baretli hem baretsiz olamaz -> ihlal degil
                        cls = CLS_UNKNOWN
                        dbg["cakisma"] += 1
                        suppressed.append({"bbox": bodies[i],
                                           "head_box": hd["bbox"],
                                           "conf": hd["conf"],
                                           "track_id": tid})
                self._register(states, tid, cls, hd["conf"], bodies[i],
                               hd["bbox"], now, detections, violations)

            # Kafasi bulunamamis kisi kutulari -> belirsiz (?)
            if self.show_unmatched_person:
                for (x1, y1, x2, y2, ptid) in persons:
                    if (x1, y1, x2, y2) in used_person:
                        continue
                    if (y2 - y1) < self.min_person_h or (x2 - x1) < self.min_person_w:
                        dbg["kucuk"] += 1
                        continue
                    detections.append({
                        "bbox": (x1, y1, x2, y2), "head_box": None,
                        "track_id": -(ptid or 999), "cls": CLS_UNKNOWN,
                        "conf": 0.0, "confirmed_helmet": False,
                    })
        else:
            # ---- person_first: v2/v3 davranisi ----
            for i, (x1, y1, x2, y2, ptid) in enumerate(persons):
                if (y2 - y1) < self.min_person_h or (x2 - x1) < self.min_person_w:
                    dbg["kucuk"] += 1
                    continue
                tid = ptid if ptid is not None else -(i + 1)
                hd = self._match_head_to_person((x1, y1, x2, y2), heads)
                if hd is not None:
                    dbg["govde"] += 1
                    cls, conf, hb = hd["cls"], hd["conf"], hd["bbox"]
                else:
                    cls, conf, hb = None, 0.0, None
                self._register(states, tid, cls, conf, (x1, y1, x2, y2), hb,
                               now, detections, violations)

        self.last_debug = dbg
        self.last_suppressed = suppressed
        return detections, violations
