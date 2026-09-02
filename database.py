"""
Veritabani islemleri - SQLite
Ihlal kayitlarini saklar ve sorgular.
"""

import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path


class Database:
    def __init__(self, db_path="ihlaller.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock, self._conn() as c:
            c.execute("""
                CREATE TABLE IF NOT EXISTS violations (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id   TEXT NOT NULL,
                    camera_name TEXT,
                    track_id    INTEGER,
                    confidence  REAL,
                    timestamp   TEXT NOT NULL,
                    snapshot    TEXT,
                    crop        TEXT,
                    bbox        TEXT,
                    reviewed    INTEGER DEFAULT 0,
                    valid       INTEGER DEFAULT NULL,
                    note        TEXT
                )
            """)
            c.execute("CREATE INDEX IF NOT EXISTS idx_ts ON violations(timestamp DESC)")
            c.execute("CREATE INDEX IF NOT EXISTS idx_cam ON violations(camera_id)")

            # Kamera istatistikleri (saatlik ozet)
            c.execute("""
                CREATE TABLE IF NOT EXISTS stats (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id   TEXT NOT NULL,
                    hour        TEXT NOT NULL,
                    helmet      INTEGER DEFAULT 0,
                    no_helmet   INTEGER DEFAULT 0,
                    UNIQUE(camera_id, hour)
                )
            """)

    # ---------- Ihlal kayitlari ----------

    def add_violation(self, camera_id, camera_name, track_id, confidence,
                      snapshot=None, crop=None, bbox=None):
        ts = datetime.now().isoformat(timespec="seconds")
        with self._lock, self._conn() as c:
            cur = c.execute("""
                INSERT INTO violations
                (camera_id, camera_name, track_id, confidence, timestamp, snapshot, crop, bbox)
                VALUES (?,?,?,?,?,?,?,?)
            """, (camera_id, camera_name, track_id, confidence, ts, snapshot, crop, bbox))
            return cur.lastrowid

    def get_violations(self, limit=100, offset=0, camera_id=None,
                       since=None, only_unreviewed=False, durum=None):
        """durum: 'bekleyen' | 'onayli' | 'reddedilen' | None (hepsi)

        Inceleme kuyrugu bu uc durum uzerinden calisir. NOT: "Reddet"
        basildiginda kayit burada birikmez - panel.py'deki /review
        endpoint'i valid=False geldiginde kaydi (ve goruntusunu) aninda
        SILER (bkz. reddi_sil). Operator sadece ONAYLANAN (gercek)
        ihlallerin ihlaller/ klasorunde birikmesini istedi - "reddedilen"
        durumu bu yuzden pratikte hep bos gorunur, sorgu yine de eksiksiz
        kalsin diye burada duruyor.
        """
        q = "SELECT * FROM violations WHERE 1=1"
        params = []
        if camera_id:
            q += " AND camera_id=?"
            params.append(camera_id)
        if since:
            q += " AND timestamp>=?"
            params.append(since)
        if only_unreviewed or durum == "bekleyen":
            q += " AND reviewed=0"
        elif durum == "onayli":
            q += " AND reviewed=1 AND valid=1"
        elif durum == "reddedilen":
            q += " AND reviewed=1 AND valid=0"
        # Not: "bekleyen" kuyrugu eskiden bilerek ESKIDEN-YENIYE (ASC)
        # siralaniyordu (kuyrukta unutulan olmasin diye ilk giren ilk
        # incelensin). Panelde en son olayin en ustte gorunmesi istendigi
        # icin artik butun sekmeler AYNI SEKILDE en yeniden en eskiye
        # (DESC) siraliyor - EN ESKI BEKLEYEN KAYITLARIN LISTENIN
        # SONUNA DUSTUGUNU unutma, cok kalabalik bir kuyrukta operator
        # asagi kaydirmazsa eski kayitlar gozden kacabilir.
        q += " ORDER BY timestamp DESC"
        q += " LIMIT ? OFFSET ?"
        params += [limit, offset]

        with self._lock, self._conn() as c:
            return [dict(r) for r in c.execute(q, params).fetchall()]

    def bildirim_adaylari(self, since, pencere_dk=15):
        """Operator icin AKILLI bildirim listesi.

        SORUN: ayni kisi bareti takana kadar kamerada durursa, her
        tespit ediminde ayri bir bildirim gonderirsek operator dakikada
        bir uyari alir ve sistemi kapatir/yok sayar. COZUM: her kamera+
        kisi (track_id) icin ilk gorulen andan itibaren 'pencere_dk'
        dakikalik bir sessizlik penceresi acilir - o pencere icinde ayni
        kisi tekrar tekrar tespit edilse bile SADECE ILK SEFERINDE
        bildirim uretilir. Kayit (kuyruga dusme) bundan ETKILENMEZ -
        hepsi normal sekilde loglanmaya devam eder, sadece "birine haber
        ver" sinyali seyreltilir.

        `since`: bu zamandan SONRAKI bildirim-adaylarini dondurur (panel
        en son buraya kadar baktiginda neyi zaten gordugunu bilir).
        """
        pencere_baslangic = (datetime.now()
                             - timedelta(minutes=pencere_dk * 4)).isoformat(timespec="seconds")
        with self._lock, self._conn() as c:
            rows = [dict(r) for r in c.execute("""
                SELECT * FROM violations
                WHERE timestamp >= ? AND reviewed = 0
                ORDER BY timestamp ASC
            """, (pencere_baslangic,)).fetchall()]

        gorulen = {}   # (camera_id, track_id) -> bu pencerenin baslangic zamani
        adaylar = []
        for r in rows:
            key = (r["camera_id"], r["track_id"])
            ts = datetime.fromisoformat(r["timestamp"])
            pencere_ts = gorulen.get(key)
            if pencere_ts is None or (ts - pencere_ts) >= timedelta(minutes=pencere_dk):
                gorulen[key] = ts   # yeni pencere basladi - bu kisi/kamera icin "ilk"
                if r["timestamp"] > since:
                    adaylar.append(r)
            # pencere icindeyse (ayni kisi hala baretsiz) - gorulen GUNCELLENMEZ,
            # yani bildirim uretilmez, ama kayit yine ust tarafta zaten alinmisti.
        return adaylar

    def get_violation(self, violation_id):
        """Tek kayit - tutanak icin."""
        with self._lock, self._conn() as c:
            r = c.execute("SELECT * FROM violations WHERE id=?",
                          (violation_id,)).fetchone()
            return dict(r) if r else None

    def durum_sayilari(self):
        """Kuyruk sekmelerindeki sayilar."""
        with self._lock, self._conn() as c:
            row = c.execute("""
                SELECT
                  SUM(CASE WHEN reviewed=0 THEN 1 ELSE 0 END) bekleyen,
                  SUM(CASE WHEN reviewed=1 AND valid=1 THEN 1 ELSE 0 END) onayli,
                  SUM(CASE WHEN reviewed=1 AND valid=0 THEN 1 ELSE 0 END) reddedilen
                FROM violations
            """).fetchone()
            return {k: (row[k] or 0) for k in ("bekleyen", "onayli", "reddedilen")}

    def mark_reviewed(self, violation_id, valid=True, note=None):
        """Operator onayi: ihlal gercek mi (valid=1) yoksa yanlis alarm mi (valid=0)"""
        with self._lock, self._conn() as c:
            c.execute("""
                UPDATE violations SET reviewed=1, valid=?, note=? WHERE id=?
            """, (1 if valid else 0, note, violation_id))

    def reddi_sil(self, violation_id, snapshots_dir):
        """Operator bir kaydi 'Reddet' ile isaretledigi anda cagrilir.

        Sadece durumu guncellemekle kalmaz, kaydi VE goruntusunu (full +
        crop) TAMAMEN siler. Boylece ihlaller/ klasorunde ve veritabaninda
        sadece gercekten ONAYLANAN ihlaller birikir - operatorun acikca
        istedigi budur. Yanlis alarmlarin egitim verisi olarak saklanmasi
        (eski davranis) artik BILEREK devre disi - istenmedi."""
        with self._lock, self._conn() as c:
            row = c.execute(
                "SELECT snapshot, crop FROM violations WHERE id=?", (violation_id,)
            ).fetchone()
            if row:
                for f in (row["snapshot"], row["crop"]):
                    if f:
                        p = Path(snapshots_dir) / Path(f).name
                        p.unlink(missing_ok=True)
            c.execute("DELETE FROM violations WHERE id=?", (violation_id,))

    def reddedilenleri_toplu_sil(self, snapshots_dir):
        """BIR KERELIK gecmis temizligi: su an reddedilmis durumda duran
        eski kayitlarin hepsini (goruntuleriyle birlikte) siler. Yeni
        kural (reddi_sil) devreye girmeden ONCE reddedilmis kayitlar
        icin - reddedilenleri_temizle.py scripti bunu cagirir."""
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT snapshot, crop FROM violations WHERE reviewed=1 AND valid=0"
            ).fetchall()
            for r in rows:
                for f in (r["snapshot"], r["crop"]):
                    if f:
                        p = Path(snapshots_dir) / Path(f).name
                        p.unlink(missing_ok=True)
            c.execute("DELETE FROM violations WHERE reviewed=1 AND valid=0")
        return len(rows)

    def geri_al(self, violation_id):
        """Karari geri al - kayit tekrar bekleyen kuyruguna doner.

        Yanlis tusa basmak kacinilmaz; geri alma olmadan operator kendi
        hatasini duzeltemez ve kayitlara guveni azalir.
        """
        with self._lock, self._conn() as c:
            c.execute("""
                UPDATE violations SET reviewed=0, valid=NULL, note=NULL
                WHERE id=?
            """, (violation_id,))

    def get_summary(self, days=7):
        since = (datetime.now() - timedelta(days=days)).isoformat()
        with self._lock, self._conn() as c:
            total = c.execute(
                "SELECT COUNT(*) n FROM violations WHERE timestamp>=?", (since,)
            ).fetchone()["n"]

            today = datetime.now().strftime("%Y-%m-%d")
            today_n = c.execute(
                "SELECT COUNT(*) n FROM violations WHERE timestamp LIKE ?", (today + "%",)
            ).fetchone()["n"]

            by_cam = [dict(r) for r in c.execute("""
                SELECT camera_id, camera_name, COUNT(*) n
                FROM violations WHERE timestamp>=?
                GROUP BY camera_id ORDER BY n DESC
            """, (since,)).fetchall()]

            pending = c.execute(
                "SELECT COUNT(*) n FROM violations WHERE reviewed=0"
            ).fetchone()["n"]

            # Yanlis alarm orani (operator onayindan)
            fp = c.execute("""
                SELECT
                  SUM(CASE WHEN valid=0 THEN 1 ELSE 0 END) yanlis,
                  SUM(CASE WHEN valid=1 THEN 1 ELSE 0 END) dogru
                FROM violations WHERE reviewed=1
            """).fetchone()

        return {
            "total": total,
            "today": today_n,
            "pending_review": pending,
            "by_camera": by_cam,
            "false_positives": fp["yanlis"] or 0,
            "true_positives": fp["dogru"] or 0,
        }

    # ---------- Istatistik ----------

    def update_stats(self, camera_id, helmet=0, no_helmet=0):
        hour = datetime.now().strftime("%Y-%m-%d %H:00")
        with self._lock, self._conn() as c:
            c.execute("""
                INSERT INTO stats (camera_id, hour, helmet, no_helmet)
                VALUES (?,?,?,?)
                ON CONFLICT(camera_id, hour) DO UPDATE SET
                    helmet = helmet + excluded.helmet,
                    no_helmet = no_helmet + excluded.no_helmet
            """, (camera_id, hour, helmet, no_helmet))

    def get_hourly_stats(self, hours=24):
        since = (datetime.now() - timedelta(hours=hours)).strftime("%Y-%m-%d %H:00")
        with self._lock, self._conn() as c:
            return [dict(r) for r in c.execute("""
                SELECT hour, SUM(helmet) helmet, SUM(no_helmet) no_helmet
                FROM stats WHERE hour>=? GROUP BY hour ORDER BY hour
            """, (since,)).fetchall()]

    def tumunu_temizle(self, snapshots_dir):
        """TUM kayitlari (bekleyen+onaylanan+reddedilen) ve goruntuleri siler.

        Panelden 'Tumunu Temizle' butonuyla cagrilir - canliya gecerken
        test videolarindan biriken gecmisi sifirlamak icin. Reddedilen
        kayitlarin normalde hic silinmemesi kurali burada BILEREK
        gecersiz kilinir; bu operatorun acikca istedigi, tek seferlik bir
        sifirlama islemidir, otomatik bakim (cleanup) ile karistirilmamali.
        """
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT snapshot, crop FROM violations").fetchall()
            for r in rows:
                for f in (r["snapshot"], r["crop"]):
                    if f:
                        p = Path(snapshots_dir) / Path(f).name
                        p.unlink(missing_ok=True)
            c.execute("DELETE FROM violations")
            c.execute("DELETE FROM stats")
        return len(rows)

    # ---------- Bakim ----------

    def cleanup(self, retention_days, snapshots_dir):
        """Eski kayitlari ve goruntuleri sil"""
        if retention_days <= 0:
            return 0
        cutoff = (datetime.now() - timedelta(days=retention_days)).isoformat()
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT snapshot, crop FROM violations WHERE timestamp<?", (cutoff,)
            ).fetchall()
            for r in rows:
                for f in (r["snapshot"], r["crop"]):
                    if f:
                        p = Path(snapshots_dir) / Path(f).name
                        p.unlink(missing_ok=True)
            c.execute("DELETE FROM violations WHERE timestamp<?", (cutoff,))
            c.execute("DELETE FROM stats WHERE hour<?", (cutoff[:13] + ":00",))
        return len(rows)
