"""
Ihlal mail bildirimi.

Panelin ana akisini YAVASLATMAMASI icin, mail gonderme islemi HER ZAMAN
ayri bir thread'de (arka planda) yapilir - SMTP sunucusu yavas cevap
verse/cevap vermese bile kamera isleme thread'i beklemez.

SPAM ONLEME - "AYNI KISI Mİ" KARARI NASIL VERILIYOR:
Once izleme numarasina (track_id), sonra kare icindeki konuma gore "ayni
kisi mi" ayrimi denendi - ikisi de kisi YURUYEREK dolastiginda guvenilmez
cikti (ikisi de her saniye degisiyor). Sonra sadece kamera + sabit
bekleme suresi denendi - bu sefer de GERCEKTEN farkli bir kisi girdiginde
mail gitmemesi sorununa yol acti.

SIMDIKI YONTEM: ID Takip ozelligiyle ayni mantik - kisinin KIYAFET/RENK
GORUNUMUNE (HSV renk histogrami) bakiliyor. Bir kameradan mail gittikten
sonra, o kameradaki YENI bir ihlalin kirpilmis fotografi, yakin zamanda
mail gitmis kisilerin fotograflarindan biriyle RENK OLARAK BENZER
CIKARSA "ayni kisi" sayilir ve mail atlanir - izleme numarasi veya konum
ne olursa olsun. Renk olarak BENZEMIYORSA "farkli kisi" sayilir ve mail
hemen gider.

BILINEN SINIRLAMA: Sahada herkes AYNI RENK is kiyafeti/yelegi giyiyorsa,
bu yontem iki farkli kisiyi de "ayni" sanabilir (ikisi de ayni renkte
gorundugu icin). Bu, agir bir yapay-zeka yuz/kisi tanima modeli
kullanilmadan (ki bu GPU/VRAM'i ciddi sekilde zorlar) tamamen cozulmesi
zor bir sinirlama - ama en azindan kiyafet rengi farkli olan cogu
durumda dogru calisir. same_person_similarity esigini (config.yaml)
dusurup yukselterek bu hassasiyeti ayarlayabiliriz.
"""

import smtplib
import ssl
import threading
import time
from email.message import EmailMessage
from pathlib import Path

import cv2

_gecmis_kisiler = {}   # cam_id -> [ (histogram, zaman), ... ]
_lock = threading.Lock()

HIST_BINS = 24


def _histogram(img):
    """Kirpilmis kisi fotografindan (BGR numpy dizisi) bir renk
    'parmak izi' cikarir. img None/bos ise None doner."""
    if img is None or getattr(img, "size", 0) == 0:
        return None
    try:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [HIST_BINS, HIST_BINS],
                             [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist
    except Exception:
        return None


def _ayni_kisi_mi_ve_kaydet(cam_id, img, now, pencere, esik):
    """Cagiran _lock tutuyor olmali. img: yeni ihlalin kirpilmis
    fotografi (numpy dizisi ya da None). Donus: True ise 'ayni kisi,
    mail atla' - False ise 'farkli/bilinmeyen kisi, mail gonder ve bu
    fotografi hafizaya ekle'."""
    kayitlar = [(h, t) for (h, t) in _gecmis_kisiler.get(cam_id, [])
                if now - t < pencere]
    _gecmis_kisiler[cam_id] = kayitlar

    yeni_hist = _histogram(img)
    if yeni_hist is None:
        # Fotograf yoksa/bozuksa kiyaslama yapamayiz - guvenli tarafta
        # kalip mail GONDERILIR (ihlali kacirmamak, "hicbir sey
        # gondermemek"ten daha onemli).
        return False

    for h, _t in kayitlar:
        skor = cv2.compareHist(h, yeni_hist, cv2.HISTCMP_CORREL)
        if skor >= esik:
            return True   # ayni kisi (renk olarak benzer) - mail atlanir

    kayitlar.append((yeni_hist, now))
    _gecmis_kisiler[cam_id] = kayitlar
    return False


def _gonder(cfg_email, konu, govde, kucuk_resim_yolu, buyuk_resim_yolu):
    try:
        msg = EmailMessage()
        msg["Subject"] = konu
        msg["From"] = cfg_email.get("smtp_user", "")
        alicilar = cfg_email.get("to", [])
        if isinstance(alicilar, str):
            alicilar = [alicilar]
        alicilar = [a for a in alicilar if a]
        if not alicilar:
            print("[mail] 'to' alani bos - mail gonderilmedi")
            return
        msg["To"] = ", ".join(alicilar)
        msg.set_content(govde)

        # Iki resim de eklenir (varsa): yakin cekim (kirpilmis) VE genel/
        # buyuk goruntu (tum kare). Boylece yonetici hem kisiyi yakindan
        # hem de olay yerini/baglami genel olarak gorebilir.
        for yol, ek_adi in ((kucuk_resim_yolu, "yakin_cekim.jpg"),
                             (buyuk_resim_yolu, "genel_gorunum.jpg")):
            if yol and Path(yol).exists():
                data = Path(yol).read_bytes()
                msg.add_attachment(data, maintype="image", subtype="jpeg",
                                    filename=ek_adi)

        host = cfg_email.get("smtp_host", "smtp.gmail.com")
        port = int(cfg_email.get("smtp_port", 587))
        user = cfg_email.get("smtp_user", "")
        passwd = cfg_email.get("smtp_pass", "")

        ctx = ssl.create_default_context()
        with smtplib.SMTP(host, port, timeout=15) as s:
            s.starttls(context=ctx)
            s.login(user, passwd)
            s.send_message(msg)
        print(f"[mail] gonderildi -> {msg['To']}")
    except Exception as e:
        # Mail basarisiz olsa bile tespit/kayit sistemi ETKILENMEMELI -
        # sadece log'a yaz, hicbir istisna disariya sizmasin.
        print(f"[mail] HATA: {e}")


def ihlal_maili_gonder(cfg, cam_id, cam_name, track_id, conf, when,
                        kirpilmis_dizi=None,
                        kucuk_resim_yolu=None, buyuk_resim_yolu=None):
    """CameraWorker._log() bunu her ihlalde cagirir. cfg: TUM config.yaml
    (dict). Ayarlar cfg['email'] altinda - 'enabled: false' ise (varsayilan)
    hicbir sey yapmadan aninda geri doner. kirpilmis_dizi: kisinin
    kirpilmis fotografi (numpy/BGR dizisi) - 'ayni kisi mi' renk
    kiyaslamasi icin kullanilir."""
    e = (cfg or {}).get("email", {})
    if not e or not e.get("enabled"):
        return

    # Dusuk guvenli (sinirda) tespitler icin mail atlanir - ihlal panelde
    # yine de "Inceleme Bekleyen"e duser, sadece mail gitmez. Boylece
    # yanlis alarmlar gereksiz yere meshgul etmez.
    min_conf = float(e.get("min_confidence", 0.5))
    if conf < min_conf:
        print(f"[mail] atlandi (guven %{round(conf*100)} < esik %{round(min_conf*100)}) - {cam_name}")
        return

    pencere = float(e.get("min_seconds_between", 600))
    esik = float(e.get("same_person_similarity", 0.6))
    now = time.time()

    with _lock:
        if _ayni_kisi_mi_ve_kaydet(cam_id, kirpilmis_dizi, now, pencere, esik):
            # Kiyafet/renk olarak yakin zamanda mail gitmis biriyle
            # benzer - ayni kisi sayilir, mail atlanir.
            return

    konu = f"⚠ Baret İhlali - {cam_name}"
    govde = (
        f"Baret ihlali tespit edildi.\n\n"
        f"Kamera: {cam_name}\n"
        f"Tarih: {when.strftime('%d.%m.%Y')}\n"
        f"Saat: {when.strftime('%H:%M:%S')}\n"
        f"Takip No: #{track_id}\n"
        f"Güven skoru: %{round(conf * 100)}\n\n"
        f"Görüntüler ektedir (yakın çekim + genel görünüm, varsa).\n"
        f"Bu mail Baret Tespit Sistemi tarafından otomatik gönderilmiştir."
    )
    # SMTP baglantisi/gonderim yavas olabilir - kamera thread'ini asla
    # bloklamasin diye tamamen ayri bir arka plan thread'inde yapilir.
    threading.Thread(target=_gonder,
                      args=(e, konu, govde, kucuk_resim_yolu, buyuk_resim_yolu),
                      daemon=True).start()
