# Baret Tespit Sistemi

İki aşamalı: **COCO kişiyi bulur → kafa modeli baş bölgesine bakar**

---

## Verinin durumu

Roboflow export'undan ölçülen gerçek değerler:

| Sınıf | Kutu sayısı | Ortalama boyut |
|---|---|---|
| `head` (baretsiz) | 132 | 13.6 × 14.3 px |
| `helmet` (baretli) | 811 | 13.1 × 12.0 px |

Dengesizlik **6.1 : 1**, kutular **13×13 piksel**. İkisi de zor koşul —
sistem buna göre üç yerde telafi ediyor:

1. **Eğitimde:** `head` içeren görüntüler 4× çoğaltılıyor (oversampling)
2. **Tespitte:** kafa modeli tüm sahneyi değil, kırpılmış baş bölgesini
   tarıyor — 13px'lik kutu kırpıntı içinde orantılı olarak çok daha büyük
3. **Panelde:** baret hafızası, kararsız `head` tespitlerini stabilize ediyor

---

## Neden iki aşamalı

| Sorun | Nasıl çözülüyor |
|---|---|
| Tarih damgası insan sanılıyordu | COCO yazının etrafında kişi bulmaz |
| 13px kutu tespit edilemiyordu | Model kırpıntıda çalışıyor, kutu orantılı büyük |
| Elde baret "takılı" sanılıyordu | Baret bel hizasında, baş kırpıntısına girmiyor |
| `head` verisi az, tespit kararsız | Baret hafızası + N kare onayı |

---

## Baret hafızası (senin istediğin davranış)

> "eğer varsa hep var olarak kalsın"

Bir kişide baret **2 kez** doğrulanınca, o kişi (track ID) kalıcı olarak
"baretli" sayılır. Sonraki karelerde model yanlışlıkla "baretsiz" dese
bile uyarı üretilmez.

Panelde bu kişiler `baretli*` olarak gösterilir (yıldız = hafızadan).

Kişi kadrajdan çıkıp 5 dakika görünmezse hafızası silinir.

`config.yaml` → `helmet_memory: false` ile kapatılabilir.

---

## Kurulum

```bash
pip install -r requirements.txt
```

### 1 — Eğitim

Roboflow zip'ini `dataset` klasörüne aç, sonra:

```bash
python 1_egit.py
```

4GB VRAM'de bellek hatası alırsan sırayla:
```bash
python 1_egit.py --imgsz 1280
python 1_egit.py --imgsz 1280 --model yolo11s.pt
```

Eğitim sonunda `head` sınıfının **recall** değerine bak. Bu sayı
"132 örnek yeterli miydi" sorusunun cevabı.

### 2 — Modeli yerleştir

```
runs/detect/kafa_v1/weights/best.pt  →  models/best.pt
```

### 3 — Kameraları tanımla

`config.yaml` → `cameras` bölümü. Test için video dosyası da verilebilir.

### 4 — Paneli başlat

```bash
python 2_panel.py
```

Panel: **http://localhost:8000**

---

## Panel

**Sol:** canlı kamera görüntüleri + özet
**Sağ:** ihlal kayıtları (yeni gelen kırmızı yanıp söner)

Bir kayda tıkla → büyük görüntü → **"İhlal Doğru"** / **"Yanlış Alarm"**

Bu işaretleme iki işe yarar: yöneticiye gerçek sayıyı verir, ve bir
sonraki eğitim turu için düzeltici veri toplar.

---

## Ayar rehberi

| Sorun | Ayar |
|---|---|
| Çok yanlış alarm | `confirm_frames` ↑ (6), `head_conf` ↑ (0.4) |
| İhlal kaçırılıyor | `confirm_frames` ↓ (2), `head_conf` ↓ (0.15), `helmet_memory: false` |
| GPU zorlanıyor | `target_fps` ↓ (2), `person_imgsz` ↓ (1280) |
| Panel çok doluyor | `cooldown_seconds` ↑ (180) |

### İlk hafta kalibrasyonu

1. `head_conf: 0.15` ile başla (daha çok yakalasın)
2. Panelde her ihlali işaretle
3. Bir hafta sonra yanlış alarm oranına bak, `head_conf`'u kademeli yükselt

---

## Gerçekçi beklenti

132 `head` örneği bu zorlukta bir görev için hâlâ az. Eğitim sonrası
`head` recall'u %40-60 civarı çıkarsa normal karşıla — sistem tasarımı
(kırpıntı + hafıza + N kare onayı) bunu telafi edecek şekilde kuruldu.

Recall'u yükseltmenin tek yolu daha fazla `head` örneği. Hedef: **250-300**.
Panelde "Yanlış Alarm" işaretlediğin kareler bu birikimin bir parçası olur.

---

## Dosyalar

```
baret_sistem/
├── 1_egit.py          eğitim (polygon→bbox + oversampling dahil)
├── 2_panel.py         panel sunucusu
├── detector.py        iki aşamalı motor + baret hafızası
├── camera_worker.py   kamera thread'i
├── database.py        SQLite
├── config.yaml        TÜM AYARLAR
├── static/index.html  panel arayüzü
├── models/best.pt     ← eğitilen model buraya
├── ihlaller/          (otomatik) ihlal görüntüleri
└── ihlaller.db        (otomatik) veritabanı
```

---

## v2 düzeltmesi — önemli

İlk sürümde kafa modeli **kırpıntıda** çalıştırılıyordu. Bu yanlıştı:
model tam karede, 1280px'de, 13 piksellik kutularla eğitilmişti — kırpıntıda
(160px'e büyütülmüş) o ölçekte kafa görmeyi hiç öğrenmediği için hiçbir şey
bulamıyordu. Buna *domain mismatch* denir.

**Düzeltilmiş akış:**

```
Kafa modeli TAM KAREDE çalışır (eğitildiği gibi)
        +
COCO kişileri bulur
        ↓
Kafa tespiti bir kişinin üst %45'inde mi?
   Evet → geçerli
   Hayır → AT   ← tarih damgası, yazı, ekipman böyle elenir
```

`head_imgsz` mutlaka eğitimdeki değerle **aynı** olmalı (1280).

### Panelde göreceğin etiketler

| Etiket | Anlamı |
|---|---|
| `baretli` | kafa modeli helmet dedi |
| `baretli*` | baret hafızasından geliyor |
| `BARETSIZ` | kafa modeli head dedi |
| `?` | COCO kişi buldu ama kafa sınıflandırılamadı |

Çok fazla `?` görüyorsan `head_conf` değerini düşür (0.15).

### Hata ayıklama

Kafa modelinin ne bulduğunu görmek için:
```yaml
require_person_match: false
```
Bu ayarla kişiyle eşleşmeyen tespitler de çizilir — tarih damgası üzerinde
kutu çıkarsa filtrenin gerçekten çalıştığını doğrulamış olursun.
