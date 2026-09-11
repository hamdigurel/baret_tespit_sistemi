"""
Ihlal mail bildirimi.

Panelin ana akisini YAVASLATMAMASI icin, mail gonderme islemi HER ZAMAN
ayri bir thread'de (arka planda) yapilir - SMTP sunucusu yavas cevap
verse/cevap vermese bile kamera isleme thread'i beklemez.

Ayni kameradan arka arkaya cok fazla mail gitmesini (mesela test videosu
oynatilirken 130+ ihlal birikmisken hepsi icin tek tek mail atilmasin diye)
engellemek icin kamera basina basit bir "en az N saniyede bir" kisitlamasi
var - config.yaml -> email -> min_seconds_between.
"""

import smtplib
import ssl
import threading
import time
from email.message import EmailMessage
from pathlib import Path

_son_gonderilen = {}   # cam_id -> son mail zamani (epoch saniye)
_lock = threading.Lock()


def _gonder(cfg_email, konu, govde, resim_yolu):
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

        if resim_yolu and Path(resim_yolu).exists():
            data = Path(resim_yolu).read_bytes()
            msg.add_attachment(data, maintype="image", subtype="jpeg",
                                filename=Path(resim_yolu).name)

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


def ihlal_maili_gonder(cfg, cam_id, cam_name, track_id, conf, when, resim_yolu):
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

    bekleme = float(e.get("min_seconds_between", 60))
    now = time.time()
    with _lock:
        son = _son_gonderilen.get(cam_id, 0)
        if now - son < bekleme:
            return   # bu kamera icin cok yakin zamanda zaten mail gitti
        _son_gonderilen[cam_id] = now

    konu = f"⚠ Baret İhlali - {cam_name}"
    govde = (
        f"Baret ihlali tespit edildi.\n\n"
        f"Kamera: {cam_name}\n"
        f"Tarih: {when.strftime('%d.%m.%Y')}\n"
        f"Saat: {when.strftime('%H:%M:%S')}\n"
        f"Takip No: #{track_id}\n"
        f"Güven skoru: %{round(conf * 100)}\n\n"
        f"Görüntü ektedir (varsa).\n"
        f"Bu mail Baret Tespit Sistemi tarafından otomatik gönderilmiştir."
    )
    # SMTP baglantisi/gonderim yavas olabilir - kamera thread'ini asla
    # bloklamasin diye tamamen ayri bir arka plan thread'inde yapilir.
    threading.Thread(target=_gonder, args=(e, konu, govde, resim_yolu),
                      daemon=True).start()
