# Baret Tespit Sistemi — Proje Özeti

Tersane için geliştirilen, kameralardan canlı görüntü alıp işçilerin baret
takıp takmadığını tespit eden, ihlalleri kaydeden ve bir web panelinden
izlenebilen bir sistem. Bu dosya, projede baştan bugüne ne yapıldığının,
hangi teknoloji/kütüphanelerin kullanıldığının ve sistemin nasıl
çalıştığının bir özetidir.

---

## 1. Sistemin genel amacı ve akışı

- Tersanedeki (şu an 12 adet) kameradan canlı görüntü / kayıtlı video
  alınır.
- Her karede önce **kafa/baret modeli** (özel eğitilmiş) çalışır, kafaları
  bulur ve "baretli" mi "baretsiz" mi olduğuna karar verir.
- **COCO insan modeli** (hazır, genel amaçlı) destek amaçlı çalışır: bir
  kafa tespitinin gerçekten bir insanla eşleşip eşleşmediğini doğrular
  (örneğin tarih damgası, ekipman gibi kafa-benzeri şeyleri elemek için).
- Bir kişi birkaç kare üst üste "baretsiz" görülürse ve bir süre (cooldown)
  içinde tekrar uyarı verilmemişse, bu bir **ihlal** olarak kaydedilir:
  ekran görüntüsü + kırpılmış yakın çekim + veritabanı kaydı.
- İhlaller panelde listelenir, yönetici "İhlal Doğru" / "Yanlış Alarm"
  olarak işaretleyebilir (bu hem gerçek sayıyı verir hem de ileride modeli
  yeniden eğitmek için düzeltici veri biriktirir).

## 2. Yapay zeka / model tarafı

- **Model:** YOLO11 tabanlı iki ayrı model kullanılıyor:
  - `models/best.pt` — Roboflow'dan alınan veriyle **özel olarak
    fine-tune edilmiş** kafa/baret modeli (`head` ve `helmet` sınıfları).
  - `models/yolo11m.pt` — hazır, eğitilmemiş **COCO insan modeli**; sadece
    "burada gerçekten bir insan var mı" diye destek/doğrulama için
    kullanılıyor (kişi kapısı / person-gate).
- **Veri dengesizliği sorunu:** eğitim verisinde `helmet` (baretli) 811
  örnek, `head` (baretsiz) sadece 132 örnekti (6.1:1 dengesizlik), kutular
  da çok küçüktü (~13×13 piksel). Bunu telafi etmek için: eğitimde `head`
  içeren görüntüler 4× çoğaltıldı (oversampling), tespit sırasında model
  tam karede (kırpıntıda değil) çalıştırılıyor, panelde "baret hafızası"
  ile kararsız tespitler stabilize ediliyor.
- **Baret hafızası (helmet memory):** bir kişide baret 2 kere doğrulanınca
  o kişi kalıcı olarak "baretli" sayılır (`baretli*` — yıldız hafızadan
  geldiğini gösterir). Model daha sonra birkaç kare "baretsiz" dese bile
  hemen ihlal üretmez. Ama bu döngüsel değil — kişi 4 kare üst üste
  baretsiz görülürse hafıza **iptal edilir** ve tekrar normal şekilde
  ihlal üretebilir (yani baretini gerçekten çıkaran biri sonsuza kadar
  "baretli" görünmeye devam etmez).
- **Takip (tracking):** `IoUTracker` adında basit, öz-yazılmış bir takip
  algoritması ile her kafaya bir `#track_id` numarası veriliyor, böylece
  aynı kişi kareden kareye takip edilebiliyor. Bu numaralar **her kamera
  için ayrı ve geçicidir** (kalıcı kimlik değildir).
- **Otomatik kalibrasyon:** her kameranın açısına/uzaklığına göre "bir
  kafa kaç piksel" ve "takip için ne kadar hareket normal" gibi değerler
  otomatik ölçülüp ayarlanıyor — kamera başına elle ayar yapmaya gerek
  kalmıyor.
- **Sabit nesne bastırma:** bir lamba, asılı ekipman gibi dakikalarca hiç
  kıpırdamayan "kafa benzeri" nesneler otomatik tespit edilip göz ardı
  ediliyor (yanlış alarmı azaltmak için).
- **Çelişki çözümü:** aynı kişi aynı anda hem "baretli" hem "baretsiz"
  tespit edilirse (örtüşen kutular), sistem bunu ihlal saymıyor,
  belirsiz (`?`) olarak işaretliyor.
- **Sisli/tozlu/karanlık görüntü iyileştirme:** modele giden kare gerekirse
  otomatik kontrast/aydınlatma iyileştirmesinden geçiriliyor (orijinal
  görüntü/kayıt bundan etkilenmiyor, sadece modelin girdisi).
- **Hassasiyet ayarı:** panelden her kamera için ayrı ayrı, canlı olarak
  (yeniden başlatmadan) "ne kadar hassas tespit yapılsın" ayarlanabiliyor
  — düşükte sadece emin olduğu kişileri gösterir, yükseğe çekilince
  zayıf/uzak tespitleri de yakalar ama yanlış alarm riski artar.
- **Mesafe ön ayarları:** her kamera "yakın / normal / uzak" olarak
  etiketlenip, buna göre makul varsayılan hassasiyet/ölçek değerleri
  otomatik uygulanıyor.
- **Kameralar arası ID Takip (deneysel, isteğe bağlı):** açıldığında bir
  kişinin renk/görünüm bilgisine (HSV renk histogramı) bakarak başka bir
  kameraya geçtiğinde aynı "global kimliği" (`[G..]`) taşımaya çalışıyor.
  Kapalıyken sisteme hiçbir ek yük getirmiyor; açıkken de tersanede
  herkesin benzer renk iş kıyafeti giymesi yüzünden %100 güvenilir değil
  — bu yüzden bilerek panelden açılıp kapatılabilen, varsayılan **kapalı**
  bir özellik olarak eklendi.
- **Çoklu kamera performansı:** `BatchInferenceEngine` adlı bir mekanizma,
  birden fazla kameranın karelerini kısa bir bekleme penceresinde tek bir
  GPU çağrısında birleştiriyor — kamera sayısı arttıkça gecikmenin
  doğrusal artmasını engelliyor.

## 3. Panel (web arayüzü) tarafında yapılanlar

- **Canlı görüntü:** her kameranın işlenmiş (kutulu/etiketli) görüntüsü
  tarayıcıda MJPEG akışı olarak izlenebiliyor; "Teknik Görünüm" açılırsa
  ekranın üstünde kafa/kişi/kalibrasyon sayaçları da gösteriliyor.
- **İhlal listesi:** sağ tarafta ihlaller anlık düşüyor, tıklayınca büyük
  görüntü + "İhlal Doğru" / "Yanlış Alarm" onay/red butonları çıkıyor.
  Toplu temizleme ve "geri al" desteği var.
- **Aynı anda çalışacak kamera sayısı sınırı:** düşük VRAM'li test
  makinesinde (4GB) tüm kameraları birden çalıştırmak güvenli olmadığı
  için, panelden bir üst sınır (`max_aktif_kamera`) belirlenebiliyor.
  Sınırı aşan kameralar otomatik olarak "sırada" durumuna geçiyor.
- **Kamera durumları ve manuel kontrol:** her kamera üç halden birinde
  olabiliyor — **aktif** (çalışıyor), **sırada** (kapasite doluluğu
  yüzünden bekliyor), **durduruldu** (elle kapatılmış). Panelden her
  kamera için ayrı ayrı Durdur/Başlat butonu, sırada bekleyen bir kamerayı
  "Aktif Yap" ile öne alma (kapasiteyi aşıyorsa hangi kameranın
  durdurulacağını önceden sorup onay alan bir mekanizma ile) eklendi.
  Ayrıca o an izlenen kameranın üstünde/yanında her zaman görünen tek bir
  güç butonu var, hangi kamerayı izliyorsan ona göre Durdur/Başlat/Aktif
  Yap gösteriyor.
- **Kamera büyütme:** izlenen kameranın görüntüsünü tek butonla (sadece o
  videoyu, tüm paneli değil) tam ekran büyütüp küçültme eklendi.
- **Sistem/GPU izleme:** panelden GPU/VRAM/CPU/RAM kullanımı anlık
  izlenebiliyor (`nvidia-smi` üzerinden okunuyor).
- **Bildirimler:** tarayıcı bildirimleri (Notification API) ile yeni
  ihlal geldiğinde uyarı verilebiliyor.
- **Kamera ekle/sil/link yönetimi:** panelden yeni kamera eklenip
  silinebiliyor, RTSP/dosya bağlantıları yönetilebiliyor.

## 4. Depolama

- **Veritabanı:** SQLite (`ihlaller.db`) — her ihlalin kamerası, zamanı,
  takip numarası, güven skoru, onay durumu vb. tutuluyor.
- **Görseller:** her ihlal için tam kare ekran görüntüsü ve kırpılmış
  yakın çekim `ihlaller/` klasörüne kaydediliyor.
- Kayıt tutma süresi (`retention_days`) config'den ayarlanabiliyor.

## 5. Kullanılan kütüphaneler / teknolojiler

**Yapay zeka / görüntü işleme**
- `ultralytics` — YOLO11 modellerini çalıştırmak için (hem özel baret
  modeli hem hazır COCO insan modeli).
- `torch` (PyTorch) — modellerin altyapısı; Docker imajında GPU'ya göre
  (CUDA 12.1) hazır geliyor.
- `opencv-python` (cv2) — video okuma, görüntü işleme, çizim, renk
  histogramı (ID Takip özelliği için) vb.
- `lapx` — YOLO'nun kendi kişi takip özelliği (`model.track`) için gerekli
  eşleştirme kütüphanesi.
- `sahi` — büyük/kalabalık sahnelerde parça parça (tiling) tespit için
  kuruldu, şu an config'de kapalı (`sahi_enabled: false`).

**Sunucu / web**
- `fastapi` — panelin arka uç (backend) API'si.
- `uvicorn` — FastAPI'yi çalıştıran web sunucusu.
- Vanilla **HTML/CSS/JavaScript** — panelin ön yüzü (herhangi bir React/
  Vue gibi çatı kullanılmadı, tek sayfa, sade ve bağımsız).
- Fullscreen API, Notification API — tarayıcı yerleşik özellikleri.

**Yardımcı**
- `pyyaml` — tüm ayarların tutulduğu `config.yaml` dosyasını okumak için.
- `psutil` — CPU/RAM kullanımını okumak için (opsiyonel).
- `sqlite3` — Python'un kendi standart kütüphanesi, veritabanı için.

**Dağıtım / çalıştırma**
- **Docker + docker-compose** — sistemi bir konteyner içinde, GPU
  destekli (`pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime` taban imajı)
  paketlemek için hazırlandı. Docker Desktop'ta tekrarlayan bir motor
  hatası (`_ping` / 500 Internal Server Error) yüzünden şu an **beraber
  duraklatıldı**, sistem geçici olarak doğrudan `python panel.py` ile
  Windows üzerinde çalıştırılıyor. Docker dosyaları (Dockerfile,
  docker-compose.yml, .dockerignore) hazır ve doğrulanmış durumda, motor
  sorunu çözülünce (ya da başka bir makinede) tekrar kullanılabilir.
- `.bat` betikleri (`Paneli_Baslat.bat`, `Kisayol_Olustur.bat`,
  `_tarayici_ac.bat`) — panelin tek tıkla başlatılması ve masaüstü
  kısayolu oluşturulması için.

## 6. Eğitim (fine-tuning) tarafı

- Kafa/baret modeli, Roboflow'dan alınan gerçek tersane görüntüleriyle
  fine-tune edildi (`models/best.pt`).
- Veri az ve dengesiz olduğu için (`head` sınıfı sadece 132 örnek) oran
  düzeltme (oversampling) ve iki-aşamalı tespit mimarisiyle bu eksiklik
  telafi edilmeye çalışıldı.
- Eğitim sırasında segment/kutu (box) etiket formatı uyuşmazlığı yüzünden
  bir miktar görüntü (~60+9 adet) düşmüştü — bu, ileride verinin gözden
  geçirilip modelin yeniden eğitilmesi gerektiğinde hatırlanması gereken
  bir not olarak duruyor.

## 7. Şu an bilinen, ileride bakılabilecek noktalar

- Docker Desktop motor kararsızlığı (`_ping` hatası) tam çözülmedi;
  `görüntüler` klasör adındaki Türkçe karakterlerin (`ö`, `ü`) Docker
  mount'unda sorun çıkardığı da tespit edildi — ASCII isimli bir klasöre
  (`goruntuler`) taşınması öneriliyor.
- Nadiren aynı kişi için iki farklı takip numarasıyla çakışan kutu
  görülebiliyor (takip algoritmasının küçük bir yan etkisi) — sık
  tekrarlarsa `track_reach_px` / `track_max_age` ayarlarıyla
  iyileştirilebilir.
- Panelde henüz kullanıcı girişi / şifre koruması yok — birden fazla
  kişinin erişeceği bir üretim makinesine taşınmadan önce eklenmesi
  önerildi.
- Kullanılmayan eski dosyalar (`test_videolari/`, `runs/`, `__pycache__/`,
  `yolo11n.pt`) hâlâ diskte duruyor, temizlenmesi önerildi ama henüz
  yapılmadı.

---

*Bu belge projenin genel bir özetidir; ayrıntılı kurulum ve ayar rehberi
için aynı klasördeki `OKUBENI.md` dosyasına bakabilirsin.*
