"""
Ihlal mail bildirimi.

Panelin ana akisini YAVASLATMAMASI icin, mail gonderme islemi HER ZAMAN
ayri bir thread'de (arka planda) yapilir - SMTP sunucusu yavas cevap
verse/cevap vermese bile kamera isleme thread'i beklemez.

IKI AYRI "spam onleme" kurali var:

1) AYNI KISI icin tekrar mail (asil istenen kural): Bir kisi baretsiz
   tespit edilip mail gittikten sonra, O KISI (ayni kamera + ayni izleme
   numarasi/track_id) hala baretsiz gorunmeye devam etse bile - mesela
   2-3 dakika boyunca kamerada dursa - tekrar tekrar mail atilmaz. Sadece
   config.yaml -> email -> min_seconds_between_person kadar sure gectikten
   sonra (kisi hala baretsizse) bir mail daha gidebilir.

2) AYNI KAMERA icin genel bir alt limit (guvenlik): Ayni kameradan COK
   KISA arayla (mesela ayni anda 2 farkli kisi baretsiz yakalanirsa) mail
   sel gibi gitmesin diye kucuk bir alt sinir - config.yaml -> email ->
   min_seconds_between.
"""

import smtplib
import ssl
import threading
import time
from email.message import EmailMessage
from pathlib import Path

_son_gonderilen_kisi = {}    # (cam_id, track_id) -> son mail zamani (epoch saniye)
_son_gonderilen_kamera = {}  # cam_id -> son mail zamani (epoch saniye)
_lock = threading.Lock()

# _son_gonderilen_kisi sozlugu zamanla eski (artik kamerada olmayan)
# izleme numaralariyla dolup buyumesin diye, bu sureden (saniye) eski
# kayitlar firsat buldukca temizlenir. Fonksiyonellige etkisi yok, sadece
# bellek temizligi.
_TEMIZLIK_ESIGI_SN = 3600


def _eski_kayitlari_temizle(now):
    eskiler = [k for k, t in _son_gonderilen_kisi.items()
               if now - t > _TEMIZLIK_ESIGI_SN]
    for k in eskiler:
        del _son_gonderilen_kisi[k]


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
                        kucuk_resim_yolu=None, buyuk_resim_yolu=None):
    """CameraWorker._log() bunu her ihlalde cagirir. cfg: TUM config.yaml
    (dict). Ayarlar cfg['email'] altinda - 'enabled: false' ise (varsayilan)
    hicbir sey yapmadan aninda geri doner."""
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

    kisi_bekleme = float(e.get("min_seconds_between_person", 600))
    kamera_bekleme = float(e.get("min_seconds_between", 15))
    now = time.time()

    with _lock:
        _eski_kayitlari_temizle(now)

        kisi_anahtar = (cam_id, track_id)
        son_kisi = _son_gonderilen_kisi.get(kisi_anahtar, 0)
        if now - son_kisi < kisi_bekleme:
            # Ayni kisi (ayni kamera + ayni izleme numarasi) icin, hala
            # baretsiz gorunse bile bu bekleme suresi dolmadan tekrar
            # mail atilmaz - surekli baretsiz durmasi mail trafigini
            # bogmasin diye.
            return

        son_kamera = _son_gonderilen_kamera.get(cam_id, 0)
        if now - son_kamera < kamera_bekleme:
            # Ayni kameradan COK KISA arayla (mesela ayni anda farkli bir
            # kisi de yakalanirsa) art arda mail gitmesin diye kucuk bir
            # genel alt limit.
            return

        _son_gonderilen_kisi[kisi_anahtar] = now
        _son_gonderilen_kamera[cam_id] = now

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
