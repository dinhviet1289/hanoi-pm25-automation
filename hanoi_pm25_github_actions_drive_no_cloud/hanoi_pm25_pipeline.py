"""
Converted from the team's Hanoi PM2.5 Colab notebook for scheduled execution.
Do not store API keys or service-account JSON in this file.
The workflow sets HANOI_PM25_DATA_ROOT and OPENAQ_API_KEY.
"""


# ===== Original notebook code cell 2 =====

import subprocess, sys, importlib


def _ensure(pip_name, module_name):
    try:
        importlib.import_module(module_name)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pip_name], check=False)


for _pip, _mod in [("pyspark", "pyspark"), ("xgboost", "xgboost"), ("scikit-learn", "sklearn"),
                   ("pyarrow", "pyarrow"), ("seaborn", "seaborn"), ("matplotlib", "matplotlib"),
                   ("requests", "requests"), ("joblib", "joblib"),
                   ("google-cloud-storage", "google.cloud.storage"),
                   ("google-cloud-secret-manager", "google.cloud.secretmanager")]:
    _ensure(_pip, _mod)

import os
import json
import time
import shutil
import tempfile
import warnings
from collections import deque
from datetime import date, timedelta, datetime, timezone

import numpy as np
import pandas as pd
import requests
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ===== Original notebook code cell 4 =====

# OpenAQ API key is read from Google Colab Secrets (key name: OPENAQ_API_KEY).
# Do not hard-code API keys in this notebook.
import os


# ===== Original notebook code cell 6 =====

# ==============================================================================
# CẤU HÌNH
# ==============================================================================
# Lấy API key từ Colab Secrets hoặc biến môi trường. Không ghi key trực tiếp vào notebook.
OPENAQ_API_KEY = os.environ.get("OPENAQ_API_KEY", "").strip()
# Scheduled Colab Enterprise runs can retrieve the API key from Secret Manager.
if not OPENAQ_API_KEY and os.environ.get("OPENAQ_SECRET_ID"):
    try:
        from google.cloud import secretmanager
        _project_id = (os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID") or "").strip()
        if _project_id:
            _secret_name = f"projects/{_project_id}/secrets/{os.environ['OPENAQ_SECRET_ID']}/versions/latest"
            OPENAQ_API_KEY = secretmanager.SecretManagerServiceClient().access_secret_version(
                request={"name": _secret_name}
            ).payload.data.decode("utf-8").strip()
    except Exception as _secret_exc:
        print(f"[!] Không đọc được Secret Manager; sẽ thử Colab Secrets. Chi tiết: {_secret_exc}")
if not OPENAQ_API_KEY:
    try:
        from google.colab import userdata
        OPENAQ_API_KEY = (userdata.get("OPENAQ_API_KEY") or "").strip()
    except Exception:
        pass
if not OPENAQ_API_KEY:
    # Không dừng cell cấu hình tại đây: để notebook định nghĩa đầy đủ hàm.
    # Kiểm tra bắt buộc sẽ chạy ở đầu main() với thông báo rõ ràng.
    print("[!] Chưa tìm thấy OPENAQ_API_KEY. Hãy cấu hình Colab Secrets trước khi chạy main().")

HANOI_LAT, HANOI_LON = 21.0285, 105.8542
RADIUS_M = 25000
START_DATE = "2019-01-01"

# Chỉ dự báo từ giờ UTC đã hoàn tất gần nhất, tránh lấy một giờ chưa đủ dữ liệu.
NOW_UTC = datetime.now(timezone.utc)
LATEST_HOUR_UTC = NOW_UTC.replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
TODAY = LATEST_HOUR_UTC.date()
END_DATE = TODAY.isoformat()
WEATHER_ARCHIVE_END = (TODAY - timedelta(days=7)).isoformat()  # dùng Archive đến khoảng 7 ngày trước
UPDATE_WINDOW_DAYS = 14  # cửa sổ cập nhật để nhận dữ liệu đến trễ/sửa đổi
ACTIVE_SENSOR_LOOKBACK_DAYS = 30  # sensor có datetimeLast cũ hơn mốc này không cần hỏi ở mọi lần chạy
STALE_SENSOR_RECHECK_HOURS = 24  # sensor metadata cũ được hỏi lại tối đa mỗi 24h
UNCHANGED_SENSOR_RECHECK_HOURS = 6  # nếu timestamp metadata không đổi, tránh hỏi lặp trong 6h
UPDATE_START_DATE = max(START_DATE, (TODAY - timedelta(days=UPDATE_WINDOW_DAYS)).isoformat())
START_YEAR = int(START_DATE[:4])
END_YEAR = TODAY.year

TARGET_PARAMS = ["pm25", "pm10", "no2", "co", "so2", "o3"]
POLLUTANT_COLS = ["pm25", "pm10", "co", "no2", "so2", "o3"]

HORIZONS = [1, 3, 6, 12, 24]
TRAIN_Q, VAL_Q = 0.70, 0.85
MIN_FEATURE_COVERAGE = 0.30
RANDOM_STATE = 42
RF_PARAMS = dict(n_estimators=100, max_depth=12, min_samples_leaf=10, max_samples=0.5,
                 n_jobs=-1, random_state=RANDOM_STATE)
XGB_PARAMS = dict(n_estimators=1000, learning_rate=0.05, max_depth=6, subsample=0.8,
                  colsample_bytree=0.8, tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE)
EARLY_STOPPING_ROUNDS = 40
# Khi cập nhật định kỳ, dùng lại model đã lưu để giảm thời gian chạy.
# Đặt True trong một lần chạy có chủ đích nếu muốn huấn luyện lại toàn bộ mô hình.
RETRAIN_MODELS_EVERY_RUN = False
REFIT_XGB_FOR_LATEST = False
PIPELINE_VERSION = 4  # phiên bản pipeline/đầu ra; bổ sung audit theo trạm và thời gian
MODEL_SCHEMA_VERSION = 3  # feature schema chưa đổi; không retrain chỉ vì thay đổi bảng đầu ra

# Quy tắc vận hành cho dashboard: không xem dự báo từ quan trắc cũ hơn 6 giờ là "hiện tại".
# Đây là ngưỡng dự án, KHÔNG phải ngưỡng chính thức của OpenAQ; điều chỉnh sau khi đo độ trễ nguồn.
MAX_FORECAST_AGE_HOURS = 6
MIN_DAILY_VALID_HOURS = 18  # ngưỡng kiểm soát chất lượng do nhóm chọn, không phải trích nguyên văn QĐ 1459

# Tuỳ chọn để chạy lịch tự động trên Colab Enterprise / môi trường có quyền Cloud Storage.
# Tạo bucket, cấp quyền Storage Object Viewer/Creator cho danh tính thực thi, rồi đặt env vars.
GCS_BUCKET = os.environ.get("HANOI_PM25_GCS_BUCKET", "").strip()
GCS_PREFIX = os.environ.get("HANOI_PM25_GCS_PREFIX", "hanoi_pm25").strip("/")
GCP_PROJECT_ID = (os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT_ID") or "").strip()
AUTO_DOWNLOAD_EXPORT = os.environ.get("COLAB_SCHEDULED_RUN", "false").lower() != "true"

# False = dùng cache lịch sử và cập nhật gần đây; True = tải lại toàn bộ lịch sử (tốn thời gian/API).
FORCE_REFRESH_RAW = False
# --- GitHub Actions / local runner: Drive folder is synced by run_pipeline.py ---
DATA_ROOT = os.path.abspath(os.environ.get("HANOI_PM25_DATA_ROOT", "./runtime_data"))
os.makedirs(DATA_ROOT, exist_ok=True)

RAW_EXPECTED = [
    "openaq_hanoi_2019_2026_raw.csv",
    "open_meteo_hanoi_2019_2026_raw.csv",
]
RAW_ALIASES = {
    RAW_EXPECTED[0]: [
        "openaq_hanoi_2019_2026_raw (1).csv",
        "openaq_hanoi_2019_2026_raw(1).csv",
    ],
    RAW_EXPECTED[1]: [
        "open_meteo_hanoi_2019_2026_raw (1).csv",
        "open_meteo_hanoi_2019_2026_raw(1).csv",
    ],
}

def _normalize_raw_names(folder):
    names = set(os.listdir(folder))
    for expected, aliases in RAW_ALIASES.items():
        if expected in names:
            continue
        for alias in aliases:
            if alias in names:
                os.rename(os.path.join(folder, alias), os.path.join(folder, expected))
                print(f"-> Đã chuẩn hóa tên file: {alias} → {expected}")
                names.remove(alias)
                names.add(expected)
                break
    return all(name in names for name in RAW_EXPECTED)

if not _normalize_raw_names(DATA_ROOT):
    raise FileNotFoundError(
        "Thiếu file RAW sau khi đồng bộ Google Drive. Cần có: "
        + ", ".join(RAW_EXPECTED)
        + ". Kiểm tra GOOGLE_DRIVE_FOLDER_ID và quyền Service Account."
    )

print(f"✓ DATA_ROOT đang dùng: {DATA_ROOT}")
print("✓ Hai file RAW:", [os.path.exists(os.path.join(DATA_ROOT, f)) for f in RAW_EXPECTED])
OUTPUT_DIR = os.path.join(DATA_ROOT, "output_hanoi_pm25")
CACHE_DIR = os.path.join(DATA_ROOT, "cache_openaq")
SENSOR_POLL_STATE_FILE = os.path.join(DATA_ROOT, "sensor_poll_state.csv")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
AQ_FILE = os.path.join(DATA_ROOT, RAW_EXPECTED[0])
WEATHER_FILE = os.path.join(DATA_ROOT, RAW_EXPECTED[1])
PARQUET_PATH = os.path.join(tempfile.gettempdir(), "spark_warehouse_hanoi_pm25.parquet")

LOOKER_MAX_MB = 95
BQ_PROJECT_ID = ""
BQ_DATASET = "hanoi_pm25"

HEADERS = {"X-API-Key": OPENAQ_API_KEY, "Accept": "application/json",
           "User-Agent": "HanoiPM25-SparkPipeline/3.1-incremental"}
OPENAQ_BASE = "https://api.openaq.org/v3"
METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_VARS = ["temperature_2m", "relative_humidity_2m", "surface_pressure",
                "wind_speed_10m", "wind_direction_10m", "precipitation"]

# Ngưỡng VN_AQI theo Bảng 2, Quyết định 1459/QĐ-TCMT (đơn vị µg/m³).
# Với từng chất, nội suy tuyến tính giữa các BP và các mốc AQI 0, 50, 100, 150, 200, 300, 400, 500.
VN_AQI_LEVELS = np.array([0, 50, 100, 150, 200, 300, 400, 500], dtype=float)
VN_PM25_BP = np.array([0, 25, 50, 80, 150, 250, 350, 500], dtype=float)
VN_AQI_LV = VN_AQI_LEVELS.copy()
VN_BREAKPOINTS = {
    "pm25": np.array([0, 25, 50, 80, 150, 250, 350, 500], dtype=float),
    "pm10": np.array([0, 50, 150, 250, 350, 420, 500, 600], dtype=float),
    "co":   np.array([0, 10000, 30000, 45000, 60000, 90000, 120000, 150000], dtype=float),
    "so2":  np.array([0, 125, 350, 550, 800, 1600, 2100, 2630], dtype=float),
    "no2":  np.array([0, 100, 200, 700, 1200, 2350, 3100, 3850], dtype=float),
    "o3_1h": np.array([0, 160, 200, 300, 400, 800, 1000, 1200], dtype=float),
    # Hướng dẫn không tính chỉ số O3-8h nếu O3-8h lớn nhất trong ngày > 400 µg/m³.
    "o3_8h": np.array([0, 100, 120, 170, 210, 400], dtype=float),
}
AQI_LABELS = ["1. Tốt", "2. Trung bình", "3. Kém", "4. Xấu", "5. Rất xấu", "6. Nguy hại"]
MOLAR_MASS = {"co": 28.01, "no2": 46.0055, "so2": 64.066, "o3": 48.0}


# ===== Original notebook code cell 8 =====

# ==============================================================================
# HTTP có kiểm soát rate-limit (KHÔNG nuốt lỗi âm thầm)
# ==============================================================================
class AuthError(RuntimeError):
    pass


class FetchError(RuntimeError):
    pass


class RateLimiter:
    """Giữ số request dưới 55/phút và 1.900/giờ (OpenAQ: 60/phút, 2.000/giờ)."""

    def __init__(self, per_minute=55, per_hour=1900):
        self.per_minute, self.per_hour = per_minute, per_hour
        self.stamps = deque()

    def wait(self):
        while True:
            now = time.time()
            while self.stamps and now - self.stamps[0] > 3600:
                self.stamps.popleft()
            in_minute = [t for t in self.stamps if now - t <= 60]
            if len(self.stamps) >= self.per_hour:
                pause = 3600 - (now - self.stamps[0]) + 1
                print(f"   [rate-limit] chạm ngưỡng 1 giờ, nghỉ {pause / 60:.1f} phút...")
                time.sleep(max(pause, 1))
                continue
            if len(in_minute) >= self.per_minute:
                time.sleep(max(60 - (now - in_minute[0]) + 0.5, 0.5))
                continue
            self.stamps.append(time.time())
            return


def http_get(url, headers=None, params=None, limiter=None, max_retries=6):
    last_err = None
    for attempt in range(max_retries):
        if limiter:
            limiter.wait()
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=60)
        except requests.exceptions.RequestException as exc:
            last_err = repr(exc)
            time.sleep(5 * (attempt + 1))
            continue
        code = resp.status_code
        if code == 200:
            return resp
        if code in (401, 403):
            raise AuthError(f"HTTP {code}: API key sai/hết hạn/không đủ quyền (độ dài key hiện tại: {len(OPENAQ_API_KEY)}).")
        if code == 429:
            try:
                wait = int(float(resp.headers.get("x-ratelimit-reset", 60))) + 2
            except ValueError:
                wait = 62
            print(f"   [429] Bị giới hạn tốc độ, chờ {wait}s...")
            time.sleep(wait)
            last_err = "HTTP 429"
            continue
        if code in (408, 500, 502, 503, 504):
            last_err = f"HTTP {code}"
            time.sleep(5 * (attempt + 1))
            continue
        raise FetchError(f"HTTP {code}: {resp.text[:200]} | {url} {params}")
    raise FetchError(f"Hết {max_retries} lần thử ({last_err}): {url} {params}")


def to_ugm3(value, parameter, units):
    """Chuẩn hoá đơn vị khí (ppm/ppb/mg/m³) về µg/m³ để các trạm so sánh được với nhau."""
    if value is None or parameter in ("pm25", "pm10"):
        return value
    u = (units or "").lower().replace("µ", "u").replace("μ", "u")
    if u == "ppm":
        return value * 1000.0 * MOLAR_MASS[parameter] / 24.45
    if u == "ppb":
        return value * MOLAR_MASS[parameter] / 24.45
    if u.startswith("mg"):
        return value * 1000.0
    return value                                   # đã là µg/m³


# ===== Original notebook code cell 10 =====

# ==============================================================================
# BƯỚC 1: THU THẬP DỮ LIỆU (OpenAQ v3 + Open-Meteo) → CSV
# Chế độ mặc định: giữ cache lịch sử, cập nhật lại cửa sổ 14 ngày gần nhất mỗi lần chạy.
# ==============================================================================
def list_locations(limiter):
    locations, page = [], 1
    while True:
        resp = http_get(f"{OPENAQ_BASE}/locations", headers=HEADERS, limiter=limiter,
                        params={"coordinates": f"{HANOI_LAT},{HANOI_LON}", "radius": RADIUS_M,
                                "limit": 100, "page": page})
        results = resp.json().get("results", [])
        locations.extend(results)
        if len(results) < 100:
            break
        page += 1
    return locations


def _utc_iso(dt):
    """Format datetime có timezone thành chuỗi UTC ISO-8601 cho OpenAQ."""
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_sensor_range(sensor_id, start_utc, end_utc, limiter):
    """Lấy các giá trị trung bình theo giờ trong một khoảng UTC từ OpenAQ v3."""
    rows, page = [], 1
    while True:
        resp = http_get(
            f"{OPENAQ_BASE}/sensors/{sensor_id}/hours",
            headers=HEADERS,
            params={"datetime_from": start_utc, "datetime_to": end_utc,
                    "limit": 1000, "page": page},
            limiter=limiter,
        )
        results = resp.json().get("results", [])
        for r in results:
            ts = ((r.get("period") or {}).get("datetimeFrom") or {}).get("utc")
            val = r.get("value")
            if ts is not None and val is not None:
                rows.append((ts, val))
        if len(results) < 1000:
            break
        page += 1
    return rows


def fetch_sensor_year(sensor_id, year, limiter):
    """Tải cả năm cho lần khởi tạo cache hoặc khi FORCE_REFRESH_RAW=True."""
    start = datetime(year, 1, 1, tzinfo=timezone.utc)
    end = (datetime(year + 1, 1, 1, tzinfo=timezone.utc) - timedelta(seconds=1)
           if year < END_YEAR else LATEST_HOUR_UTC)
    return fetch_sensor_range(sensor_id, _utc_iso(start), _utc_iso(end), limiter)


def _fetch_rows_to_frame(rows):
    return pd.DataFrame(rows, columns=["ts", "value"])


def ingest_openaq():
    os.makedirs(CACHE_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    limiter = RateLimiter()
    print("=== [1/6] OpenAQ v3: dùng cache lịch sử + cập nhật cửa sổ gần nhất ===")
    locations = list_locations(limiter)
    print(f"-> Tìm thấy {len(locations)} trạm trong bán kính {RADIUS_M // 1000} km quanh Hà Nội.")
    print(f"-> Cập nhật từ {UPDATE_START_DATE} đến giờ UTC đã hoàn tất gần nhất: {_utc_iso(LATEST_HOUR_UTC)}")

    names = [str(l.get("name") or "Unknown") for l in locations]
    dup = {n for n in names if names.count(n) > 1}
    jobs = []
    for loc, name in zip(locations, names):
        disp = f"{name} ({loc.get('id')})" if name in dup else name
        coords = loc.get("coordinates") or {}
        lat, lon = coords.get("latitude", HANOI_LAT), coords.get("longitude", HANOI_LON)
        first_utc = (loc.get("datetimeFirst") or {}).get("utc") or f"{START_YEAR}-01-01T00:00:00Z"
        last_utc = (loc.get("datetimeLast") or {}).get("utc") or f"{END_YEAR}-12-31T23:59:59Z"
        y0, y1 = max(START_YEAR, int(str(first_utc)[:4])), min(END_YEAR, int(str(last_utc)[:4]))
        for sensor in loc.get("sensors", []):
            p = sensor.get("parameter") or {}
            sensor_last = sensor.get("datetimeLast") or {}
            if isinstance(sensor_last, dict):
                sensor_last_utc = sensor_last.get("utc") or sensor_last.get("local")
            else:
                sensor_last_utc = sensor_last if isinstance(sensor_last, str) else None
            if p.get("name") in TARGET_PARAMS and y0 <= y1:
                for yr in range(y0, y1 + 1):
                    jobs.append(dict(sensor_id=sensor.get("id"), station_id=loc.get("id"), station=disp,
                                     lat=lat, lon=lon, parameter=p["name"], units=p.get("units"), year=yr,
                                     sensor_last_utc=sensor_last_utc))
    print(f"-> Tìm thấy {len(jobs)} cặp (sensor, năm); ở chế độ cập nhật sẽ gộp còn 1 truy vấn cho mỗi sensor cần kiểm tra.")

    # ------------------------------------------------------------------
    # FAST INCREMENTAL: khi đã có RAW, không đọc/ghép lại hàng trăm cache sensor-năm.
    # Chỉ hỏi các sensor đang hoạt động; sensor có metadata cũ được kiểm tra lại hằng ngày.
    # ------------------------------------------------------------------
    if os.path.exists(AQ_FILE) and not FORCE_REFRESH_RAW:
        print("[FAST INCREMENTAL] Đã có RAW: giữ lịch sử, chỉ cập nhật sensor gần đây.")
        print("-> Bỏ qua việc đọc/ghép lại cache lịch sử sensor-năm; upsert vào RAW hiện có.")
        old = pd.read_csv(AQ_FILE)
        required = {"datetime", "station_id", "station", "latitude", "longitude"}
        if not required.issubset(old.columns):
            raise RuntimeError(
                f"RAW hiện có thiếu cột {sorted(required - set(old.columns))}. "
                "Hãy kiểm tra file RAW hoặc bật FORCE_REFRESH_RAW=True trong một lần chạy có chủ đích."
            )
        old["datetime"] = pd.to_datetime(old["datetime"], errors="coerce", utc=True).dt.floor("h").dt.tz_localize(None)
        old["station_id"] = old["station_id"].astype(str)
        for c in POLLUTANT_COLS:
            if c not in old.columns:
                old[c] = np.nan
            old[c] = pd.to_numeric(old[c], errors="coerce")
        old = old.dropna(subset=["datetime", "station_id"])
        key_cols = ["datetime", "station_id", "station", "latitude", "longitude"]
        old = old[key_cols + POLLUTANT_COLS]

        start_dt = max(
            datetime(START_YEAR, 1, 1, tzinfo=timezone.utc),
            datetime.fromisoformat(UPDATE_START_DATE).replace(tzinfo=timezone.utc),
        )
        end_iso = _utc_iso(LATEST_HOUR_UTC)
        # Một sensor xuất hiện trong nhiều năm của jobs; chỉ giữ một job metadata/sensor.
        recent_jobs = {}
        for j in jobs:
            if j.get("sensor_id") is not None:
                sid = str(j["sensor_id"])
                if sid not in recent_jobs or int(j.get("year", 0)) > int(recent_jobs[sid].get("year", 0)):
                    recent_jobs[sid] = j

        # Lưu trạng thái poll để sensor cũ không bị hỏi lại ở mỗi lần chạy (ví dụ lịch 6 giờ).
        poll_records = {}
        if os.path.exists(SENSOR_POLL_STATE_FILE):
            try:
                _ps = pd.read_csv(SENSOR_POLL_STATE_FILE, dtype={"sensor_id": str})
                if {"sensor_id", "last_checked_utc"}.issubset(_ps.columns):
                    for _, _r in _ps.iterrows():
                        poll_records[str(_r["sensor_id"])] = _r.to_dict()
            except Exception as _exc:
                print(f"[!] Không đọc được sensor poll-state; sẽ tiếp tục cập nhật an toàn: {_exc}")

        recent_frames, failed_recent, request_rows = [], [], 0
        requests_made, skipped_stale = 0, 0
        now_poll = pd.Timestamp.now(tz="UTC")
        ordered_jobs = list(recent_jobs.values())
        for i, j in enumerate(ordered_jobs, 1):
            sid = str(j["sensor_id"])
            meta_last = pd.to_datetime(j.get("sensor_last_utc"), errors="coerce", utc=True)
            previous = poll_records.get(sid, {})
            last_check = pd.to_datetime(previous.get("last_checked_utc"), errors="coerce", utc=True)
            previous_meta_last = pd.to_datetime(previous.get("sensor_datetime_last_utc"), errors="coerce", utc=True)
            metadata_is_stale = pd.notna(meta_last) and (now_poll - meta_last) > pd.Timedelta(days=ACTIVE_SENSOR_LOOKBACK_DAYS)
            checked_recently = pd.notna(last_check) and (now_poll - last_check) < pd.Timedelta(hours=STALE_SENSOR_RECHECK_HOURS)
            metadata_unchanged = (pd.notna(meta_last) and pd.notna(previous_meta_last)
                                  and meta_last <= previous_meta_last)
            unchanged_checked_recently = (pd.notna(last_check) and
                (now_poll - last_check) < pd.Timedelta(hours=UNCHANGED_SENSOR_RECHECK_HOURS))
            if metadata_is_stale and checked_recently:
                skipped_stale += 1
                continue
            if (not metadata_is_stale) and metadata_unchanged and unchanged_checked_recently:
                skipped_stale += 1
                continue
            try:
                rows = fetch_sensor_range(j["sensor_id"], _utc_iso(start_dt), end_iso, limiter)
            except AuthError:
                raise
            except FetchError as exc:
                failed_recent.append({**j, "error": str(exc)[:300]})
                print(f"   [!] Cập nhật lỗi {j['station']} | {j['parameter']}: {exc}")
                continue
            requests_made += 1
            request_rows += len(rows)
            # Ghi poll-state cả khi API trả 0 dòng: tránh truy vấn lại sensor cũ mỗi 6 giờ.
            poll_records[sid] = {
                "sensor_id": sid,
                "last_checked_utc": pd.Timestamp.now(tz="UTC").isoformat(),
                "sensor_datetime_last_utc": j.get("sensor_last_utc") or "",
                "last_returned_rows": int(len(rows)),
                "last_status": "ok",
            }
            if rows:
                ts = pd.to_datetime([r[0] for r in rows], errors="coerce", utc=True).floor("h").tz_localize(None)
                vals = [to_ugm3(float(v), j["parameter"], j["units"]) for _, v in rows]
                part = pd.DataFrame({
                    "datetime": ts,
                    "station_id": str(j["station_id"]),
                    "station": j["station"],
                    "latitude": j["lat"],
                    "longitude": j["lon"],
                    "parameter": j["parameter"],
                    "value": vals,
                })
                recent_frames.append(part)
            if requests_made % 25 == 0:
                pd.DataFrame(list(poll_records.values())).to_csv(SENSOR_POLL_STATE_FILE, index=False)
                print(f"   -> Đã gọi {requests_made} sensor; bỏ qua {skipped_stale} sensor không đổi/cũ đang trong cooldown; API trả {request_rows:,} bản ghi.")
        if poll_records:
            pd.DataFrame(list(poll_records.values())).to_csv(SENSOR_POLL_STATE_FILE, index=False)
        print(f"-> Kết thúc cập nhật: {requests_made} API request; bỏ qua {skipped_stale} sensor không đổi/cũ; {request_rows:,} bản ghi trả về.")

        if failed_recent:
            pd.DataFrame(failed_recent).to_csv(
                os.path.join(OUTPUT_DIR, "openaq_failed_requests.csv"), index=False
            )
        if recent_frames:
            recent_long = pd.concat(recent_frames, ignore_index=True)
            recent = recent_long.pivot_table(
                index=key_cols, columns="parameter", values="value", aggfunc="mean"
            ).reset_index()
            recent.columns.name = None
            for c in POLLUTANT_COLS:
                if c not in recent.columns:
                    recent[c] = np.nan
            recent = recent[key_cols + POLLUTANT_COLS]
            combined = pd.concat([old, recent], ignore_index=True, sort=False)
            combined["_order"] = np.arange(len(combined))
            combined = (combined.sort_values(["datetime", "_order"], kind="mergesort")
                                .groupby(["datetime", "station_id"], as_index=False, sort=False)
                                .last())
            combined = combined.drop(columns=["_order"], errors="ignore")
            combined = combined[key_cols + POLLUTANT_COLS].sort_values(["datetime", "station_id"]).reset_index(drop=True)
            combined.to_csv(AQ_FILE, index=False)
            with open(AQ_FILE + ".ok", "w", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
            print(f"-> API trả {request_rows:,} bản ghi gần đây; RAW sau upsert: {len(combined):,} dòng.")
            print("-> Timestamp mới nhất trong RAW:", combined["datetime"].max())
        else:
            print("[!] API không trả phép đo mới trong cửa sổ truy vấn. Giữ nguyên RAW lịch sử; kiểm tra độ trễ/nguồn tại các trạm.")
        return AQ_FILE

    frames, failed = [], []
    for i, j in enumerate(jobs, 1):
        cache_path = os.path.join(CACHE_DIR, f"{j['sensor_id']}_{j['year']}.csv")
        part = None
        if os.path.exists(cache_path) and not FORCE_REFRESH_RAW:
            try:
                cached = pd.read_csv(cache_path)
                if {"ts", "value"}.issubset(cached.columns):
                    part = cached.dropna(subset=["ts", "value"])
            except Exception:
                print(f"   [!] Cache lỗi, sẽ tải lại: {cache_path}")
                part = None

        # Nếu có cache cho năm hiện tại: gọi API lại cho cửa sổ 14 ngày gần nhất,
        # ghép vào cache và ghi đè giá trị cũ khi cùng timestamp (để nhận bản cập nhật từ nguồn).
        if j["year"] == END_YEAR and part is not None and not FORCE_REFRESH_RAW:
            update_start_dt = max(
                datetime(END_YEAR, 1, 1, tzinfo=timezone.utc),
                datetime.fromisoformat(UPDATE_START_DATE).replace(tzinfo=timezone.utc),
            )
            try:
                fresh_rows = fetch_sensor_range(
                    j["sensor_id"], _utc_iso(update_start_dt), _utc_iso(LATEST_HOUR_UTC), limiter
                )
                if fresh_rows:
                    recent = _fetch_rows_to_frame(fresh_rows)
                    part = pd.concat([part, recent], ignore_index=True)
                    part = (part.dropna(subset=["ts", "value"])
                                .drop_duplicates(subset=["ts"], keep="last")
                                .sort_values("ts"))
                    part.to_csv(cache_path, index=False)
                print(f"   [{i}/{len(jobs)}] cập nhật gần đây: {j['station']} | {j['parameter'].upper()} | {len(fresh_rows):,} bản ghi API")
            except AuthError:
                raise
            except FetchError as exc:
                failed.append({**j, "error": str(exc)[:300]})
                print(f"   [!] Cập nhật API thất bại; giữ cache cũ cho sensor {j['sensor_id']}: {exc}")

        # Cache lịch sử: giữ nguyên nếu không phải full refresh.
        elif part is None or FORCE_REFRESH_RAW:
            try:
                rows = fetch_sensor_year(j["sensor_id"], j["year"], limiter)
            except AuthError:
                raise
            except FetchError as exc:
                print(f"   [!] Tải lỗi sensor {j['sensor_id']} năm {j['year']}: {exc}")
                failed.append({**j, "error": str(exc)[:300]})
                continue
            part = _fetch_rows_to_frame(rows)
            part.to_csv(cache_path, index=False)
            if i % 20 == 0 or len(part) > 0:
                print(f"   [{i}/{len(jobs)}] tải cache: {j['station']} | {j['parameter'].upper()} | {j['year']}: {len(part):,} giờ")

        if part is None or part.empty:
            continue
        part = part.assign(
            datetime=part["ts"], station_id=str(j["station_id"]), station=j["station"],
            latitude=j["lat"], longitude=j["lon"], parameter=j["parameter"],
            value=[to_ugm3(float(v), j["parameter"], j["units"]) for v in part["value"]]
        )
        frames.append(part[["datetime", "station_id", "station", "latitude", "longitude", "parameter", "value"]])

    if failed:
        pd.DataFrame(failed).to_csv(os.path.join(OUTPUT_DIR, "openaq_failed_requests.csv"), index=False)
        print(f"\n[CẢNH BÁO] {len(failed)} request lỗi; các cache cũ có sẵn được giữ lại nếu có.")
    if not frames:
        raise RuntimeError("Không có dữ liệu OpenAQ. Kiểm tra API key, cache và kết nối mạng.")

    df_raw = pd.concat(frames, ignore_index=True)
    df_raw["datetime"] = pd.to_datetime(df_raw["datetime"], utc=True).dt.floor("h").dt.tz_localize(None)
    df_piv = df_raw.pivot_table(
        index=["datetime", "station_id", "station", "latitude", "longitude"],
        columns="parameter", values="value", aggfunc="mean"
    ).reset_index()
    df_piv.columns.name = None
    for c in POLLUTANT_COLS:
        if c not in df_piv.columns:
            df_piv[c] = np.nan
    df_piv.to_csv(AQ_FILE, index=False)
    with open(AQ_FILE + ".ok", "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"-> Đã lưu {AQ_FILE}: {len(df_piv):,} dòng, {df_piv['station'].nunique()} trạm.")
    print("-> Timestamp mới nhất trong OpenAQ:", df_piv["datetime"].max())


def _meteo_frame(hourly):
    return pd.DataFrame({
        "datetime": pd.to_datetime(hourly["time"]),
        "temperature": hourly["temperature_2m"], "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"], "wind_speed": hourly["wind_speed_10m"],
        "direction": hourly["wind_direction_10m"], "precipitation": hourly["precipitation"]})


def _fetch_weather_archive(start_date, end_date, base):
    if start_date > end_date:
        return pd.DataFrame(columns=["datetime", "temperature", "humidity", "pressure", "wind_speed", "direction", "precipitation"])
    resp = http_get(METEO_ARCHIVE_URL, params={**base, "start_date": start_date, "end_date": end_date})
    return _meteo_frame(resp.json()["hourly"])


def _fetch_recent_weather_forecast(base):
    gap_days = max(1, (TODAY - date.fromisoformat(WEATHER_ARCHIVE_END)).days + 1)
    resp = http_get(METEO_FORECAST_URL, params={**base, "past_days": min(92, gap_days), "forecast_days": 1})
    wx = _meteo_frame(resp.json()["hourly"])
    # Archive dùng đến hết ngày WEATHER_ARCHIVE_END; forecast chỉ lấp sau mốc đó.
    cutover = pd.Timestamp(WEATHER_ARCHIVE_END) + pd.Timedelta(days=1)
    latest_naive = pd.Timestamp(LATEST_HOUR_UTC.replace(tzinfo=None))
    return wx[(wx["datetime"] >= cutover) & (wx["datetime"] <= latest_naive)].copy()


def ingest_weather():
    print("=== [2/6] Open-Meteo: giữ lịch sử + cập nhật gần đây ===")
    base = {"latitude": HANOI_LAT, "longitude": HANOI_LON, "hourly": ",".join(WEATHER_VARS),
            "wind_speed_unit": "ms", "timezone": "UTC"}
    weather_cols = ["datetime", "temperature", "humidity", "pressure", "wind_speed", "direction", "precipitation"]
    old = None
    if os.path.exists(WEATHER_FILE) and not FORCE_REFRESH_RAW:
        try:
            old = pd.read_csv(WEATHER_FILE, parse_dates=["datetime"])
            old = old[[c for c in weather_cols + ["city"] if c in old.columns]]
            print(f"-> Giữ lại {len(old):,} dòng thời tiết lịch sử đã lưu.")
        except Exception as exc:
            print(f"[!] Không đọc được weather cache; sẽ tải lịch sử lại: {exc}")
            old = None

    frames = [old] if old is not None else []
    try:
        if old is None or FORCE_REFRESH_RAW:
            # Lần đầu (hoặc full refresh): tải Archive theo từng năm.
            for yr in range(START_YEAR, END_YEAR + 1):
                s = max(START_DATE, f"{yr}-01-01")
                e = min(f"{yr}-12-31", WEATHER_ARCHIVE_END)
                if s > e:
                    continue
                part = _fetch_weather_archive(s, e, base)
                frames.append(part)
                print(f"   + Archive {yr}: {len(part):,} giờ")
        else:
            # Các lần chạy sau: chỉ tải lại 14 ngày gần nhất đến mốc Archive khả dụng.
            if UPDATE_START_DATE <= WEATHER_ARCHIVE_END:
                part = _fetch_weather_archive(UPDATE_START_DATE, WEATHER_ARCHIVE_END, base)
                frames.append(part)
                print(f"   + Cập nhật Archive {UPDATE_START_DATE} → {WEATHER_ARCHIVE_END}: {len(part):,} giờ")
    except FetchError as exc:
        if old is None or FORCE_REFRESH_RAW:
            raise RuntimeError(
                f"Không tải đủ dữ liệu thời tiết lịch sử từ Open-Meteo ({exc}). "
                "Dừng để tránh vô tình ghi đè bằng bộ dữ liệu chỉ có vài ngày gần nhất."
            ) from exc
        print(f"[!] Không cập nhật được một phần dữ liệu Archive: {exc}. Giữ dữ liệu cũ nếu có.")

    try:
        recent_forecast = _fetch_recent_weather_forecast(base)
        if not recent_forecast.empty:
            frames.append(recent_forecast)
        print(f"   + Forecast API: {len(recent_forecast):,} giờ gần đây/đã hoàn tất")
    except FetchError as exc:
        print(f"[!] Không cập nhật được phần thời tiết gần đây: {exc}. Giữ dữ liệu cũ nếu có.")

    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise RuntimeError("Không có dữ liệu thời tiết đọc được hoặc tải được từ Open-Meteo.")
    df_w = pd.concat(frames, ignore_index=True, sort=False)
    df_w["datetime"] = pd.to_datetime(df_w["datetime"], errors="coerce")
    latest_naive = pd.Timestamp(LATEST_HOUR_UTC.replace(tzinfo=None))
    df_w = df_w[df_w["datetime"].notna() & (df_w["datetime"] <= latest_naive)]
    df_w = (df_w.drop_duplicates("datetime", keep="last")
                .sort_values("datetime")
                .reset_index(drop=True))
    df_w["city"] = "Ha Noi"
    df_w.to_csv(WEATHER_FILE, index=False)
    with open(WEATHER_FILE + ".ok", "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"-> Đã lưu {WEATHER_FILE}: {len(df_w):,} dòng.")
    print("-> Timestamp mới nhất trong thời tiết:", df_w["datetime"].max())


def ingest_raw_data():
    """Luôn gọi API: dùng cache năm cũ và cập nhật cửa sổ gần đây ở mỗi lần chạy."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)
    if FORCE_REFRESH_RAW:
        print("[CHẾ ĐỘ FULL REFRESH] Sẽ tải lại toàn bộ các năm; có thể mất nhiều thời gian và request API.")
    else:
        print(f"[CHẾ ĐỘ INCREMENTAL] Giữ lịch sử/cache, cập nhật {UPDATE_WINDOW_DAYS} ngày gần nhất từ API.")
    ingest_openaq()
    ingest_weather()
    return AQ_FILE, WEATHER_FILE


# ===== Original notebook code cell 12 =====

# ==============================================================================
# BƯỚC 2: PYSPARK + SPARK SQL (làm sạch, join, lag/rolling/target theo THỜI GIAN THẬT)
# ==============================================================================
SQL_JOIN = """
WITH j AS (
    SELECT
        CAST(a.datetime AS TIMESTAMP)                                          AS datetime,
        FROM_UTC_TIMESTAMP(CAST(a.datetime AS TIMESTAMP), 'Asia/Ho_Chi_Minh')  AS dt_local,
        CAST(a.station_id AS STRING)                                           AS station_id,
        TRIM(CAST(a.station AS STRING))                                        AS station,
        CAST(a.latitude  AS DOUBLE) AS latitude,
        CAST(a.longitude AS DOUBLE) AS longitude,
        CAST(a.pm25 AS DOUBLE) AS pm25,
        CAST(a.pm10 AS DOUBLE) AS pm10,
        CAST(a.co   AS DOUBLE) AS co,
        CAST(a.no2  AS DOUBLE) AS no2,
        CAST(a.so2  AS DOUBLE) AS so2,
        CAST(a.o3   AS DOUBLE) AS o3,
        CAST(w.temperature   AS DOUBLE) AS temperature,
        CAST(w.humidity      AS DOUBLE) AS humidity,
        CAST(w.pressure      AS DOUBLE) AS pressure,
        CAST(w.wind_speed    AS DOUBLE) AS wind_speed,
        CAST(w.direction     AS DOUBLE) AS direction,
        CAST(w.precipitation AS DOUBLE) AS precipitation,
        w.temp_diff_3h, w.pressure_diff_3h, w.precip_sum_24h
    FROM openaq_table a
    INNER JOIN weather_table w
        ON CAST(a.datetime AS TIMESTAMP) = w.datetime
    WHERE CAST(a.pm25 AS DOUBLE) IS NOT NULL
      AND CAST(a.pm25 AS DOUBLE) > 0
      AND CAST(a.pm25 AS DOUBLE) < 1000
)
SELECT
    datetime,
    UNIX_TIMESTAMP(datetime)        AS ts_sec,
    YEAR(dt_local)                  AS year,
    MONTH(dt_local)                 AS month,
    HOUR(dt_local)                  AS hour,
    (DAYOFWEEK(dt_local) + 5) % 7   AS day_of_week,
    station_id, station, latitude, longitude,
    pm25, pm10, co, no2, so2, o3,
    temperature, humidity, pressure, wind_speed, direction, precipitation,
    temp_diff_3h, pressure_diff_3h, precip_sum_24h
FROM j
"""


def spark_bigdata_pipeline(aq_file, weather_file):
    """Trả về (df_features_pandas, df_all_hours_pandas)."""
    from pyspark.sql import SparkSession, functions as F
    from pyspark.sql.window import Window

    print("\n=== [3/6] Apache Spark & Spark SQL... ===")
    spark = (SparkSession.builder.appName("Hanoi_PM25_Spark_Pipeline")
             .config("spark.sql.session.timeZone", "UTC")             # tránh Spark tự đổi giờ khi toPandas()
             .config("spark.driver.memory", "4g")
             .config("spark.sql.shuffle.partitions", "16")
             .config("spark.sql.execution.arrow.pyspark.enabled", "true")
             .getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")

    # ---- Dữ liệu khí tượng: tính đặc trưng trên chuỗi LIÊN TỤC theo giờ ----
    wx = spark.read.csv(os.path.abspath(weather_file), header=True, inferSchema=True)
    wx = (wx.withColumn("datetime", F.col("datetime").cast("timestamp"))
            .withColumn("ts_sec", F.unix_timestamp("datetime")))
    ww = Window.orderBy("ts_sec")
    wx = (wx.withColumn("temp_diff_3h", F.col("temperature") - F.first("temperature").over(ww.rangeBetween(-3 * 3600, -3 * 3600)))
             .withColumn("pressure_diff_3h", F.col("pressure") - F.first("pressure").over(ww.rangeBetween(-3 * 3600, -3 * 3600)))
             .withColumn("precip_sum_24h", F.sum("precipitation").over(ww.rangeBetween(-23 * 3600, 0))))
    wx.createOrReplaceTempView("weather_table")

    # ---- Dữ liệu OpenAQ ----
    aq = spark.read.csv(os.path.abspath(aq_file), header=True, inferSchema=True)
    if "station_id" not in aq.columns:                                 # file raw kiểu cũ
        aq = aq.withColumn("station_id", F.col("station"))
    for c in POLLUTANT_COLS:
        if c not in aq.columns:
            aq = aq.withColumn(c, F.lit(None).cast("double"))
    aq.createOrReplaceTempView("openaq_table")

    # ---- Spark SQL: làm sạch + join (giữ NULL, KHÔNG đổi thành 0) ----
    df_joined = spark.sql(SQL_JOIN).dropDuplicates(["station_id", "ts_sec"]).cache()

    # ---- Lag / rolling / target theo KHOẢNG CÁCH THỜI GIAN THẬT (giây) ----
    w = Window.partitionBy("station_id").orderBy("ts_sec")

    def exact(col, hours):                       # giá trị đúng tại thời điểm (t + hours); thiếu giờ đó → NULL
        return F.first(col).over(w.rangeBetween(hours * 3600, hours * 3600))

    def mean_rng(col, h_from, h_to):
        return F.avg(col).over(w.rangeBetween(h_from * 3600, h_to * 3600))

    df = (df_joined
          .withColumn("pm25_lag_1h", exact("pm25", -1))
          .withColumn("pm25_lag_3h", exact("pm25", -3))
          .withColumn("pm25_lag_6h", exact("pm25", -6))
          .withColumn("pm25_lag_24h", exact("pm25", -24))
          .withColumn("pm25_roll_mean_3h", mean_rng("pm25", -2, 0))
          .withColumn("pm25_roll_mean_12h", mean_rng("pm25", -11, 0))
          .withColumn("pm25_roll_mean_24h", mean_rng("pm25", -23, 0))
          .withColumn("pm25_roll_std_6h", F.coalesce(F.stddev("pm25").over(w.rangeBetween(-5 * 3600, 0)), F.lit(0.0))))
    for h in HORIZONS:
        df = df.withColumn(f"target_pm25_{h}h", exact("pm25", h))

    # Trạm hụt đúng 1 giờ lag → điền bằng trung bình trượt 24h (đã ghi rõ trong tài liệu)
    for c in ["pm25_lag_3h", "pm25_lag_6h", "pm25_lag_24h"]:
        df = df.withColumn(c, F.coalesce(F.col(c), F.col("pm25_roll_mean_24h")))

    # Chỉ bắt buộc có lag 1h. KHÔNG bắt buộc có target → giữ lại các giờ mới nhất để dự báo "hiện tại"
    df = df.filter(F.col("pm25_lag_1h").isNotNull())
    df.write.mode("overwrite").parquet(PARQUET_PATH)
    print(f"-> Đã lưu Parquet: '{PARQUET_PATH}'")

    df_feat = spark.read.parquet(PARQUET_PATH).orderBy("datetime", "station_id").toPandas()
    # AQI cần dùng dữ liệu OpenAQ độc lập với bộ feature ML đã lọc pm25>0.
    # Giữ cả các giờ có PM10/khí nhưng PM2.5 thiếu để tuân thủ điều kiện PM2.5 HOẶC PM10.
    df_all = aq.select(
        "station_id", "station", "latitude", "longitude", "datetime",
        "pm25", "pm10", "co", "no2", "so2", "o3"
    ).toPandas()
    spark.stop()
    print(f"-> Bản ghi có đủ đặc trưng: {len(df_feat):,} | tổng số giờ PM2.5 hợp lệ: {len(df_all):,}")
    if df_feat.empty:
        raise RuntimeError("Không có bản ghi nào sau khi join OpenAQ với Open-Meteo. Kiểm tra lại dữ liệu raw.")
    return df_feat, df_all


# ===== Original notebook code cell 14 =====

# ==============================================================================
# VN_AQI (QĐ 1459/QĐ-TCMT): AQI GIỜ + AQI NGÀY; TÁCH RIÊNG CHỈ SỐ PM2.5 DỰ BÁO
# ==============================================================================
def _round_aqi(values):
    """Làm tròn số AQI về số nguyên gần nhất (nồng độ âm/thiếu trả về NaN)."""
    arr = np.asarray(values, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    valid = np.isfinite(arr)
    out[valid] = np.floor(arr[valid] + 0.5)
    return out


def concentration_to_vn_subindex(conc, pollutant):
    """Nội suy AQIx từ nồng độ µg/m³ theo Bảng 2 QĐ 1459. Đây là chỉ số của MỘT thông số."""
    arr = np.asarray(conc, dtype=float)
    out = np.full(arr.shape, np.nan, dtype=float)
    bp = VN_BREAKPOINTS[pollutant]
    levels = VN_AQI_LEVELS[:len(bp)] if pollutant != "o3_8h" else np.array([0, 50, 100, 150, 200, 300], dtype=float)
    valid = np.isfinite(arr) & (arr >= 0)
    if valid.any():
        # Trên BP cao nhất, AQI bị chặn ở mốc cao nhất của bảng (500 với các chất thông thường).
        out[valid] = np.interp(arr[valid], bp, levels)
    return _round_aqi(out)


def pm25_to_vn_aqi(conc):
    """Chỉ số thông số PM2.5 từ một nồng độ. Với AQI giờ chính thức phải đưa NowCast vào hàm này."""
    return concentration_to_vn_subindex(conc, "pm25")


def aqi_category(aqi):
    """Phân hạng AQI theo QĐ 1459: 0–50, 51–100, 101–150, 151–200, 201–300, 301–500."""
    a = np.asarray(aqi, dtype=float)
    idx = np.digitize(a, [50, 100, 150, 200, 300], right=True)
    idx = np.clip(idx, 0, len(AQI_LABELS) - 1)
    out = np.array(AQI_LABELS, dtype=object)[idx]
    out[~np.isfinite(a)] = "Không có dữ liệu"
    return out


def pm25_equivalent_band_idx(conc):
    """Phân hạng dựa trên một điểm nồng độ PM2.5, CHỈ LÀ PROXY; không phải độ chính xác VN_AQI tổng hợp."""
    return np.digitize(pm25_to_vn_aqi(conc), [50, 100, 150, 200, 300], right=True)


def nowcast_pm(values):
    """NowCast 12 giờ theo QĐ 1459: w=max(Cmin/Cmax,0.5), yêu cầu ≥2/3 giờ gần nhất có số liệu."""
    vals = np.asarray(values, dtype=float).copy()
    vals[(~np.isfinite(vals)) | (vals < 0)] = np.nan
    padded = np.concatenate([np.full(11, np.nan), vals])
    win = sliding_window_view(padded, 12)[:, ::-1]  # cột 0 = giờ hiện tại
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        cmax = np.nanmax(win, axis=1)
        cmin = np.nanmin(win, axis=1)
        w_star = np.divide(cmin, cmax, out=np.ones_like(cmax), where=np.isfinite(cmax) & (cmax > 0))
        w = np.maximum(w_star, 0.5)
        weights = w[:, None] ** np.arange(12)[None, :]
        valid = np.isfinite(win)
        numerator = np.nansum(win * weights, axis=1)
        denominator = np.sum(weights * valid, axis=1)
        nc = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    nc[valid[:, :3].sum(axis=1) < 2] = np.nan
    return nc


def _max_available_indices(frame, columns):
    """Max các AQIx sẵn có nhưng chỉ trả AQI tổng hợp khi PM2.5 hoặc PM10 có chỉ số hợp lệ."""
    pm_required = frame[["aqi_pm25", "aqi_pm10"]].notna().any(axis=1)
    max_idx = frame[columns].max(axis=1, skipna=True)
    max_idx.loc[~pm_required] = np.nan
    return _round_aqi(max_idx.to_numpy())


def compute_nowcast_table(df_all):
    """Tính AQI giờ tổng hợp cho từng trạm; PM2.5/PM10 dùng NowCast, khí dùng trung bình 1 giờ."""
    d = df_all.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce", utc=True).dt.tz_localize(None).dt.floor("h")
    for col in ["pm25", "pm10", "co", "no2", "so2", "o3"]:
        if col not in d.columns:
            d[col] = np.nan
        d[col] = pd.to_numeric(d[col], errors="coerce")
        d.loc[d[col] < 0, col] = np.nan
    d = d.dropna(subset=["station_id", "datetime"]).drop_duplicates(["station_id", "datetime"]).copy()
    parts = []
    for sid, g in d.groupby("station_id", sort=False):
        g = g.sort_values("datetime").drop_duplicates("datetime").set_index("datetime")
        idx = pd.date_range(g.index.min(), g.index.max(), freq="h")
        p25 = g["pm25"].reindex(idx).to_numpy(dtype=float)
        p10 = g["pm10"].reindex(idx).to_numpy(dtype=float)
        parts.append(pd.DataFrame({
            "station_id": str(sid), "datetime": idx,
            "pm25_nowcast": nowcast_pm(p25), "pm10_nowcast": nowcast_pm(p10),
        }))
    if not parts:
        return pd.DataFrame(columns=["station_id", "datetime", "pm25_nowcast", "pm10_nowcast",
                                     "aqi_pm25", "aqi_pm10", "aqi_co", "aqi_no2", "aqi_so2", "aqi_o3",
                                     "aqi_hourly_vn", "aqi_nowcast", "aqi_category"])
    nc = pd.concat(parts, ignore_index=True)
    d["station_id"] = d["station_id"].astype(str)
    out = d.merge(nc, on=["station_id", "datetime"], how="left", validate="one_to_one")
    out["aqi_pm25"] = concentration_to_vn_subindex(out["pm25_nowcast"].to_numpy(), "pm25")
    out["aqi_pm10"] = concentration_to_vn_subindex(out["pm10_nowcast"].to_numpy(), "pm10")
    out["aqi_co"] = concentration_to_vn_subindex(out["co"].to_numpy(), "co")
    out["aqi_no2"] = concentration_to_vn_subindex(out["no2"].to_numpy(), "no2")
    out["aqi_so2"] = concentration_to_vn_subindex(out["so2"].to_numpy(), "so2")
    out["aqi_o3"] = concentration_to_vn_subindex(out["o3"].to_numpy(), "o3_1h")
    index_cols = ["aqi_pm25", "aqi_pm10", "aqi_co", "aqi_no2", "aqi_so2", "aqi_o3"]
    out["aqi_hourly_vn"] = _max_available_indices(out, index_cols)
    # Keep legacy column for downstream tables, but it now holds composite hourly VN_AQI, not PM2.5-only AQI.
    out["aqi_nowcast"] = out["aqi_hourly_vn"]
    out["aqi_category"] = aqi_category(out["aqi_hourly_vn"].to_numpy())
    return out[["station_id", "datetime", "pm25_nowcast", "pm10_nowcast", *index_cols,
                "aqi_hourly_vn", "aqi_nowcast", "aqi_category"]]


def prepare_features(df_feat, df_all):
    """Thêm đặc trưng thời gian/gió và ghép AQI giờ tổng hợp theo QĐ 1459."""
    df = df_feat.copy()
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce").astype("datetime64[ns]")
    df["datetime_local"] = df["datetime"] + pd.Timedelta(hours=7)
    rad = np.deg2rad(pd.to_numeric(df["direction"], errors="coerce"))
    df["wind_u"] = -df["wind_speed"] * np.sin(rad)
    df["wind_v"] = -df["wind_speed"] * np.cos(rad)
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df["hour"] / 24), np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"], df["month_cos"] = np.sin(2 * np.pi * df["month"] / 12), np.cos(2 * np.pi * df["month"] / 12)
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["pm25_diff_1h"] = df["pm25"] - df["pm25_lag_1h"]
    nc = compute_nowcast_table(df_all)
    df = df.merge(nc, on=["station_id", "datetime"], how="left", validate="many_to_one")
    df["aqi_category"] = df["aqi_category"].fillna("Không có dữ liệu")
    return df.sort_values(["datetime", "station_id"]).reset_index(drop=True)


# ===== Original notebook code cell 16 =====

# ==============================================================================
# BƯỚC 4: CHIA TRAIN / VAL / TEST THEO THỜI GIAN + MÔ HÌNH 1-24 GIỜ + BASELINE
# ==============================================================================
FEATURE_COLS = [
    "pm25", "pm10", "co", "no2", "so2", "o3",
    "temperature", "humidity", "pressure", "wind_speed", "wind_u", "wind_v", "precipitation",
    "temp_diff_3h", "pressure_diff_3h", "precip_sum_24h",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "day_of_week", "is_weekend",
    "latitude", "longitude",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_6h", "pm25_lag_24h",
    "pm25_roll_mean_3h", "pm25_roll_mean_12h", "pm25_roll_mean_24h", "pm25_roll_std_6h", "pm25_diff_1h",
]   # không dùng biến phân giai đoạn; chỉ giữ các đặc trưng có sẵn tại thời điểm dự báo


def regression_metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return {"MAE": round(float(mean_absolute_error(y, p)), 3),
            "RMSE": round(float(np.sqrt(mean_squared_error(y, p))), 3),
            "R2": round(float(r2_score(y, p)), 4),
            "PM25_Equivalent_Band_Acc_pct": round(float(np.mean(pm25_equivalent_band_idx(y) == pm25_equivalent_band_idx(p)) * 100), 2)}


def fit_xgb(X_tr, y_tr, X_va, y_va):
    from xgboost import XGBRegressor
    try:                                                     # xgboost >= 2.0
        model = XGBRegressor(**XGB_PARAMS, early_stopping_rounds=EARLY_STOPPING_ROUNDS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    except TypeError:                                        # xgboost 1.x
        model = XGBRegressor(**XGB_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], early_stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)
    best = getattr(model, "best_iteration", None)
    n_best = int(best) + 1 if best is not None else XGB_PARAMS["n_estimators"]
    return model, n_best


def train_multi_horizon(df):
    print("\n=== [4/6] Chia Train / Val / Test theo mốc thời gian & huấn luyện 1-24 giờ... ===")
    t1, t2 = df["datetime"].quantile(TRAIN_Q), df["datetime"].quantile(VAL_Q)
    df["split"] = np.where(df["datetime"] <= t1, "train", np.where(df["datetime"] <= t2, "val", "test"))
    print(f" * Train: đến {t1} | Val: đến {t2} | Test: sau {t2}")
    for s in ["train", "val", "test"]:
        print(f"   - {s:<5}: {(df['split'] == s).sum():>9,} dòng")

    is_train = df["split"] == "train"
    feats = [c for c in FEATURE_COLS if df.loc[is_train, c].notna().mean() >= MIN_FEATURE_COVERAGE]
    dropped = [c for c in FEATURE_COLS if c not in feats]
    print(f" * Dùng {len(feats)} đặc trưng. Bỏ (quá ít dữ liệu thật trong tập train): {dropped or 'không có'}")

    metrics, fact_parts, xgb_models, xgb_best_n = [], [], {}, {}
    for h in HORIZONS:
        tcol = f"target_pm25_{h}h"
        tgt_time = df["datetime"] + pd.Timedelta(hours=h)
        has_y = df[tcol].notna()
        # Purge: nhãn của dòng train/val không được "chạm" sang giai đoạn kế tiếp
        m_tr = has_y & (tgt_time <= t1)
        m_va = has_y & (df["datetime"] > t1) & (tgt_time <= t2)
        m_te = has_y & (df["datetime"] > t2)
        if min(m_tr.sum(), m_va.sum(), m_te.sum()) < 100:
            print(f"   [!] +{h}h: quá ít mẫu (train={m_tr.sum()}, val={m_va.sum()}, test={m_te.sum()}) → bỏ qua.")
            continue
        print(f"   → Khung +{h}h: train={m_tr.sum():,} val={m_va.sum():,} test={m_te.sum():,}")

        X_tr, X_va, X_te = df.loc[m_tr, feats], df.loc[m_va, feats], df.loc[m_te, feats]
        y_tr, y_va, y_te = df.loc[m_tr, tcol], df.loc[m_va, tcol], df.loc[m_te, tcol]

        # Random Forest: cần điền khuyết bằng median của tập train
        med = X_tr.median().fillna(0.0)
        rf = RandomForestRegressor(**RF_PARAMS).fit(X_tr.fillna(med), y_tr)
        rf_va, rf_te = np.clip(rf.predict(X_va.fillna(med)), 0, None), np.clip(rf.predict(X_te.fillna(med)), 0, None)

        # XGBoost: xử lý NaN gốc + early stopping trên tập validation
        xgb, n_best = fit_xgb(X_tr, y_tr, X_va, y_va)
        xgb_va, xgb_te = np.clip(xgb.predict(X_va), 0, None), np.clip(xgb.predict(X_te), 0, None)
        xgb_models[h], xgb_best_n[h] = xgb, n_best

        # Baseline: Persistence (PM2.5 hiện tại) và Trung bình 24h gần nhất
        cands = {
            "Persistence (baseline)": (df.loc[m_va, "pm25"].to_numpy(), df.loc[m_te, "pm25"].to_numpy()),
            "Trung bình 24h (baseline)": (df.loc[m_va, "pm25_roll_mean_24h"].to_numpy(), df.loc[m_te, "pm25_roll_mean_24h"].to_numpy()),
            "Random Forest": (rf_va, rf_te),
            "XGBoost": (xgb_va, xgb_te),
        }
        rmse_persist = regression_metrics(y_te, cands["Persistence (baseline)"][1])["RMSE"]
        for name, (p_va, p_te) in cands.items():
            mv, mt = regression_metrics(y_va, p_va), regression_metrics(y_te, p_te)
            skill = round((1 - mt["RMSE"] / rmse_persist) * 100, 2) if rmse_persist > 0 else np.nan
            metrics.append({"Horizon_h": h, "Horizon_label": f"+{h:02d}h", "Model": name,
                            "Val_MAE": mv["MAE"], "Val_RMSE": mv["RMSE"], "Val_R2": mv["R2"],
                            "Test_MAE": mt["MAE"], "Test_RMSE": mt["RMSE"], "Test_R2": mt["R2"],
                            "Test_PM25_Equivalent_Band_Acc_pct": mt["PM25_Equivalent_Band_Acc_pct"], "Skill_vs_Persistence_pct": skill,
                            "N_Test": int(m_te.sum())})
            sub = df.loc[m_te, ["station", "datetime_local"]]
            fact_parts.append(pd.DataFrame({
                "station": sub["station"].to_numpy(),
                "issue_time_local": sub["datetime_local"].to_numpy(),
                "target_time_local": (sub["datetime_local"] + pd.Timedelta(hours=h)).to_numpy(),
                "horizon_h": h, "horizon_label": f"+{h:02d}h", "model": name,
                "y_true": y_te.to_numpy(), "y_pred": np.asarray(p_te, dtype=float)}))

    if not metrics:
        raise RuntimeError("Không huấn luyện được khung dự báo nào (quá ít dữ liệu).")
    df_metrics = pd.DataFrame(metrics)
    cols = ["Horizon_label", "Model", "Test_MAE", "Test_RMSE", "Test_R2", "Skill_vs_Persistence_pct"]
    print("\n================ KẾT QUẢ TRÊN TẬP TEST ================")
    print(df_metrics[cols].to_string(index=False))

    fact = pd.concat(fact_parts, ignore_index=True)
    fact["abs_error"] = (fact["y_true"] - fact["y_pred"]).abs()
    # Đây là phân hạng tương đương theo một nồng độ PM2.5 đơn lẻ, KHÔNG phải độ chính xác VN_AQI tổng hợp.
    fact["pm25_equiv_index_actual"] = pm25_to_vn_aqi(fact["y_true"])
    fact["pm25_equiv_index_pred"] = pm25_to_vn_aqi(fact["y_pred"])
    fact["pm25_equiv_category_actual"] = aqi_category(fact["pm25_equiv_index_actual"].to_numpy())
    fact["pm25_equiv_category_pred"] = aqi_category(fact["pm25_equiv_index_pred"].to_numpy())
    fact["pm25_equiv_category_match"] = (
        fact["pm25_equiv_category_actual"] == fact["pm25_equiv_category_pred"]
    ).astype(int)
    fact["aqi_forecast_scope"] = "PM2.5-equivalent proxy; không phải VN_AQI tổng hợp chính thức"
    fact["date_key"] = fact["target_time_local"].dt.strftime("%Y%m%d")
    fact["datetime_hour_key"] = fact["target_time_local"].dt.strftime("%Y%m%d%H")
    fact[["y_true", "y_pred", "abs_error"]] = fact[["y_true", "y_pred", "abs_error"]].round(1)
    return df, feats, df_metrics, fact, xgb_models, xgb_best_n


def save_models(feats, xgb_models):
    os.makedirs(MODEL_DIR, exist_ok=True)
    for h, m in xgb_models.items():
        try:
            m.save_model(os.path.join(MODEL_DIR, f"xgb_pm25_{h}h.json"))
        except Exception:
            joblib.dump(m, os.path.join(MODEL_DIR, f"xgb_pm25_{h}h.joblib"), compress=3)
    with open(os.path.join(MODEL_DIR, "features.json"), "w", encoding="utf-8") as f:
        json.dump({"features": feats, "horizons": sorted(xgb_models), "pipeline_version": PIPELINE_VERSION},
                  f, ensure_ascii=False, indent=2)
    metadata = {
        "pipeline_version": PIPELINE_VERSION,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "horizons": sorted(int(h) for h in xgb_models),
        "features": feats,
        "model_scope": "XGBoost model trained on the temporal training split; validation is used for early stopping.",
    }
    with open(os.path.join(MODEL_DIR, "training_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"-> Đã lưu mô hình XGBoost + metadata tại '{MODEL_DIR}'.")


def load_saved_state():
    """Tải mô hình/kết quả đã huấn luyện để lần cập nhật định kỳ không phải train lại mỗi giờ."""
    metadata_path = os.path.join(MODEL_DIR, "training_metadata.json")
    features_path = os.path.join(MODEL_DIR, "features.json")
    metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
    fact_path = os.path.join(OUTPUT_DIR, "fact_forecast_long.csv")
    station_metrics_path = os.path.join(OUTPUT_DIR, "model_metrics_by_station.csv")
    quarter_metrics_path = os.path.join(OUTPUT_DIR, "model_metrics_by_quarter.csv")
    # Thiếu breakdown đầy đủ theo trạm/quý thì huấn luyện lại một lần để tái tạo fact test đầy đủ.
    if RETRAIN_MODELS_EVERY_RUN or not all(os.path.exists(p) for p in [metadata_path, features_path, metrics_path, fact_path, station_metrics_path, quarter_metrics_path]):
        return None
    try:
        metadata = json.load(open(metadata_path, encoding="utf-8"))
        fcfg = json.load(open(features_path, encoding="utf-8"))
        if int(metadata.get("model_schema_version", metadata.get("pipeline_version", 0))) != MODEL_SCHEMA_VERSION:
            print("-> Feature schema của model cũ không tương thích; sẽ huấn luyện lại.")
            return None
        metrics = pd.read_csv(metrics_path)
        if "Test_PM25_Equivalent_Band_Acc_pct" not in metrics.columns:
            print("-> Metrics chưa có schema mới; sẽ huấn luyện lại một lần.")
            return None
        from xgboost import XGBRegressor
        models = {}
        for h in fcfg.get("horizons", []):
            model_path = os.path.join(MODEL_DIR, f"xgb_pm25_{int(h)}h.json")
            if not os.path.exists(model_path):
                return None
            model = XGBRegressor()
            model.load_model(model_path)
            models[int(h)] = model
        fact = pd.read_csv(fact_path, parse_dates=["issue_time_local", "target_time_local"])
        print(f"-> Tái sử dụng model đã huấn luyện lúc {metadata.get('trained_at_utc', 'không rõ')}; chỉ cập nhật dữ liệu và dự báo mới.")
        return fcfg["features"], metrics, fact, models, {h: 0 for h in models}, metadata
    except Exception as exc:
        print(f"[!] Không tải được model/evaluation đã lưu; sẽ huấn luyện lại: {exc}")
        return None


def build_latest_forecast(df, feats, xgb_models, xgb_best_n):
    """Dự báo PM2.5; đánh dấu độ mới và loại các mốc đích đã qua khỏi bảng dùng cho dashboard hiện tại."""
    last = df.sort_values("datetime").groupby("station_id", as_index=False).tail(1).reset_index(drop=True)
    now_utc = pd.Timestamp.now(tz="UTC").tz_localize(None)
    generated_at_local = pd.Timestamp.now(tz="Asia/Ho_Chi_Minh").tz_localize(None)
    rows = []
    for h, model in sorted(xgb_models.items()):
        pred = np.clip(model.predict(last[feats]), 0, None)
        for i, r in last.iterrows():
            rows.append({
                "station_id": r["station_id"], "station": r["station"],
                "latitude": r["latitude"], "longitude": r["longitude"],
                "lat_lon": f"{r['latitude']:.5f},{r['longitude']:.5f}",
                "issue_time_local": r["datetime_local"],
                "generated_at_local": generated_at_local,
                "last_pm25": round(float(r["pm25"]), 1),
                "last_vn_aqi_hourly": r.get("aqi_hourly_vn", np.nan),
                "horizon_h": int(h), "horizon_label": f"+{int(h):02d}h",
                "target_time_local": r["datetime_local"] + pd.Timedelta(hours=int(h)),
                "pred_pm25": round(float(pred[i]), 1),
                "hours_since_last_obs": round(float((now_utc - r["datetime"]) / pd.Timedelta(hours=1)), 1),
            })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["pred_pm25_index_estimate"] = pm25_to_vn_aqi(out["pred_pm25"].to_numpy())
    out["pred_pm25_index_category_estimate"] = aqi_category(out["pred_pm25_index_estimate"].to_numpy())
    out["aqi_forecast_scope"] = "Ước tính chỉ số PM2.5 từ nồng độ dự báo; không phải VN_AQI tổng hợp chính thức"
    age = pd.to_numeric(out["hours_since_last_obs"], errors="coerce")
    target = pd.to_datetime(out["target_time_local"], errors="coerce")
    generated = pd.to_datetime(out["generated_at_local"], errors="coerce")
    out["is_stale"] = (age.isna() | (age > MAX_FORECAST_AGE_HOURS)).astype(int)
    out["is_stale_48h"] = (age.isna() | (age > 48)).astype(int)
    out["is_future_target"] = (target > generated).astype(int)
    out["is_current_forecast"] = ((out["is_stale"] == 0) & (out["is_future_target"] == 1)).astype(int)
    out["freshness_status"] = np.select(
        [age.isna(), age <= 3, age <= MAX_FORECAST_AGE_HOURS, age <= 24, age <= 48],
        ["Không có PM2.5 hợp lệ", "Mới (<=3h)", f"Còn mới (3–{MAX_FORECAST_AGE_HOURS}h)",
         f"Trễ ({MAX_FORECAST_AGE_HOURS}–24h)", "Cũ (24–48h)"],
        default="Rất cũ (>48h)"
    )
    out["alert_required"] = (out["is_current_forecast"] == 0).astype(int)
    out["forecast_status"] = np.select(
        [age.isna(), age > MAX_FORECAST_AGE_HOURS, out["is_future_target"] == 0],
        ["Không có PM2.5 hợp lệ", f"Quan trắc cũ hơn {MAX_FORECAST_AGE_HOURS} giờ", "Thời điểm dự báo đã qua"],
        default="Đủ điều kiện hiển thị (không phải xác nhận cảnh báo sức khỏe)"
    )
    out["alert_message"] = np.where(
        out["is_current_forecast"] == 1,
        "Dự báo còn ở tương lai và quan trắc trong ngưỡng độ mới của dự án; không thay thế cảnh báo chính thức.",
        "Không dùng như dự báo hiện tại: quan trắc quá cũ hoặc thời điểm đích đã qua."
    )
    return out


# ===== Original notebook code cell 18 =====

# ==============================================================================
# BƯỚC 5: CÁC BẢNG CHO LOOKER STUDIO
# ==============================================================================
def add_date_keys(df, col, prefix=""):
    """Looker Studio nhận dạng: YYYYMMDD = Date, YYYYMMDDHH = Date Hour, YYYYMM = Year Month."""
    d = df[col]
    df[f"{prefix}date_key"] = d.dt.strftime("%Y%m%d")
    df[f"{prefix}datetime_hour_key"] = d.dt.strftime("%Y%m%d%H")
    df[f"{prefix}year_month"] = d.dt.strftime("%Y%m")
    return df


def build_observed_hourly(df):
    cols = ["station", "datetime_local", "year", "month", "hour", "split",
            "pm25", "pm10", "no2", "co", "so2", "o3", "temperature", "humidity", "wind_speed",
            "precipitation", "pm25_nowcast", "pm10_nowcast", "aqi_pm25", "aqi_pm10",
            "aqi_co", "aqi_no2", "aqi_so2", "aqi_o3", "aqi_hourly_vn", "aqi_nowcast", "aqi_category"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan if c != "aqi_category" else "Không có dữ liệu"
    o = df[cols].copy()
    o = add_date_keys(o, "datetime_local")
    p25 = pd.to_numeric(o["pm25"], errors="coerce")
    p10 = pd.to_numeric(o["pm10"], errors="coerce")
    flag_high = p25 > 500
    flag_order = p25.notna() & p10.notna() & (p25 > p10)
    o["aqi_qc_review_flag"] = (flag_high | flag_order).astype(int)
    o["aqi_qc_review_reason"] = np.select(
        [flag_high & flag_order, flag_high, flag_order],
        ["PM2.5 > 500 và PM2.5 > PM10: cần kiểm tra sensor/đơn vị",
         "PM2.5 > 500 µg/m³: cần kiểm tra sensor/nguồn đo",
         "PM2.5 > PM10: kiểm tra cùng thời điểm/đơn vị/sensor"],
        default="Không gắn cờ bởi hai quy tắc rà soát sơ bộ"
    )
    return o.round({"pm25": 1, "pm10": 1, "no2": 1, "co": 0, "so2": 1, "o3": 1,
                    "temperature": 1, "humidity": 0, "wind_speed": 1,
                    "precipitation": 1, "pm25_nowcast": 1, "pm10_nowcast": 1})


def _daily_o3_rolling8h(df_all_local):
    """Tạo trung bình O3 của 8 giờ LIÊN TIẾP; giờ khuyết không được thay bằng 0."""
    parts = []
    for sid, g in df_all_local.groupby("station_id", sort=False):
        g = g.dropna(subset=["datetime_local"]).drop_duplicates("datetime_local").sort_values("datetime_local")
        if g.empty:
            continue
        s = g.set_index("datetime_local")["o3"].astype(float)
        idx = pd.date_range(s.index.min().floor("h"), s.index.max().floor("h"), freq="h")
        s = s.reindex(idx)
        roll = s.rolling(window=8, min_periods=8).mean()
        parts.append(pd.DataFrame({"station_id": str(sid), "datetime_local": idx, "o3_8h_mean": roll.to_numpy()}))
    if not parts:
        return pd.DataFrame(columns=["station_id", "datetime_local", "o3_8h_mean"])
    return pd.concat(parts, ignore_index=True)


def build_daily_summary(df, df_all):
    """Tính AQI ngày theo các thông số trong QĐ 1459. Ngưỡng ≥18/24 giờ là QC nội bộ của nhóm."""
    d = df_all.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce", utc=True).dt.tz_localize(None).dt.floor("h")
    d["station_id"] = d["station_id"].astype(str)
    d["datetime_local"] = d["datetime"] + pd.Timedelta(hours=7)
    for c in ["pm25", "pm10", "co", "no2", "so2", "o3"]:
        if c not in d.columns:
            d[c] = np.nan
        d[c] = pd.to_numeric(d[c], errors="coerce")
        d.loc[d[c] < 0, c] = np.nan
    # QĐ 1459 define ngày dữ liệu từ 01:00 đến 00:00 hôm sau; shift 1h gom đúng cửa sổ.
    d["date"] = (d["datetime_local"] - pd.Timedelta(hours=1)).dt.floor("D")
    p25_raw = pd.to_numeric(d["pm25"], errors="coerce")
    p10_raw = pd.to_numeric(d["pm10"], errors="coerce")
    d["aqi_qc_review_flag"] = ((p25_raw > 500) | (p25_raw.notna() & p10_raw.notna() & (p25_raw > p10_raw))).astype(int)
    d["year"] = d["date"].dt.year
    d["month"] = d["date"].dt.month
    o3r = _daily_o3_rolling8h(d)
    d = d.merge(o3r, on=["station_id", "datetime_local"], how="left", validate="one_to_one")
    g = d.groupby(["station", "date"], as_index=False).agg(
        year=("year", "first"), month=("month", "first"),
        n_hours=("pm25", "count"), n_pm10_hours=("pm10", "count"),
        qc_review_hours=("aqi_qc_review_flag", "sum"),
        pm25_mean=("pm25", "mean"), pm25_max=("pm25", "max"),
        pm10_mean=("pm10", "mean"),
        co_max=("co", "max"), no2_max=("no2", "max"), so2_max=("so2", "max"),
        o3_1h_max=("o3", "max"), o3_8h_max=("o3_8h_mean", "max")
    )
    g["n_hours_24h_window"] = g["n_hours"]
    g["aqi_pm25_24h"] = np.nan
    valid25 = g["n_hours"] >= MIN_DAILY_VALID_HOURS
    g.loc[valid25, "aqi_pm25_24h"] = pm25_to_vn_aqi(g.loc[valid25, "pm25_mean"].to_numpy())
    g["aqi_pm10_24h"] = np.nan
    valid10 = g["n_pm10_hours"] >= MIN_DAILY_VALID_HOURS
    g.loc[valid10, "aqi_pm10_24h"] = concentration_to_vn_subindex(g.loc[valid10, "pm10_mean"].to_numpy(), "pm10")
    g["aqi_co_1h_max"] = concentration_to_vn_subindex(g["co_max"].to_numpy(), "co")
    g["aqi_no2_1h_max"] = concentration_to_vn_subindex(g["no2_max"].to_numpy(), "no2")
    g["aqi_so2_1h_max"] = concentration_to_vn_subindex(g["so2_max"].to_numpy(), "so2")
    g["aqi_o3_1h_max"] = concentration_to_vn_subindex(g["o3_1h_max"].to_numpy(), "o3_1h")
    g["aqi_o3_8h_max"] = np.nan
    valid_o3_8h = g["o3_8h_max"].notna() & (g["o3_8h_max"] <= 400)
    g.loc[valid_o3_8h, "aqi_o3_8h_max"] = concentration_to_vn_subindex(g.loc[valid_o3_8h, "o3_8h_max"].to_numpy(), "o3_8h")
    g["aqi_o3_daily"] = g[["aqi_o3_1h_max", "aqi_o3_8h_max"]].max(axis=1, skipna=True)
    aqi_cols = ["aqi_pm25_24h", "aqi_pm10_24h", "aqi_co_1h_max", "aqi_no2_1h_max", "aqi_so2_1h_max", "aqi_o3_daily"]
    eligible = g[["aqi_pm25_24h", "aqi_pm10_24h"]].notna().any(axis=1)
    g["aqi_24h"] = g[aqi_cols].max(axis=1, skipna=True)
    g.loc[~eligible, "aqi_24h"] = np.nan
    g["aqi_24h"] = _round_aqi(g["aqi_24h"].to_numpy())
    g["aqi_24h_category"] = aqi_category(g["aqi_24h"].to_numpy())
    g["aqi_method_note"] = "VN_AQI ngày theo QĐ 1459; yêu cầu ít nhất PM2.5 hoặc PM10; ngưỡng >=18/24 giờ là QC nội bộ"
    g = add_date_keys(g, "date")
    return g.round({"pm25_mean": 1, "pm25_max": 1, "pm10_mean": 1,
                    "co_max": 1, "no2_max": 1, "so2_max": 1, "o3_1h_max": 1, "o3_8h_max": 1})


def build_station_freshness(df_all):
    """Bảng độ mới theo trạm dựa trên PM2.5; ngưỡng 6h là quy tắc dự án, không phải chuẩn OpenAQ."""
    d = df_all.copy()
    d["datetime"] = pd.to_datetime(d["datetime"], errors="coerce", utc=True).dt.tz_localize(None)
    d["station_id"] = d["station_id"].astype(str)
    d["pm25"] = pd.to_numeric(d.get("pm25"), errors="coerce")
    for c in ["station", "latitude", "longitude"]:
        if c not in d.columns:
            d[c] = np.nan
    stations = (d.sort_values("datetime")
                  .groupby("station_id", as_index=False)
                  .agg(station=("station", "last"), latitude=("latitude", "last"), longitude=("longitude", "last")))
    valid = d.dropna(subset=["datetime", "station_id", "pm25"])
    last = (valid.sort_values("datetime")
                .groupby("station_id", as_index=False)
                .tail(1)[["station_id", "datetime", "pm25"]]
                .rename(columns={"datetime": "last_observation_utc", "pm25": "observed_pm25"}))
    out = stations.merge(last, on="station_id", how="left", validate="one_to_one")
    now_utc = pd.Timestamp.now("UTC").tz_localize(None)
    out["last_observation_local"] = out["last_observation_utc"] + pd.Timedelta(hours=7)
    out["hours_since_last_obs"] = ((now_utc - out["last_observation_utc"]) / pd.Timedelta(hours=1)).round(1)
    age = out["hours_since_last_obs"]
    out["freshness_status"] = np.select(
        [age.isna(), age <= 3, age <= MAX_FORECAST_AGE_HOURS, age <= 24, age <= 48],
        ["Không có PM2.5 hợp lệ", "Mới (<=3h)", f"Còn mới (3–{MAX_FORECAST_AGE_HOURS}h)",
         f"Trễ ({MAX_FORECAST_AGE_HOURS}–24h)", "Cũ (24–48h)"],
        default="Rất cũ (>48h)"
    )
    out["alert_required"] = (age.isna() | (age > MAX_FORECAST_AGE_HOURS)).astype(int)
    out["alert_message"] = np.where(
        age.isna(), "CẢNH BÁO: trạm chưa có PM2.5 hợp lệ trong RAW.",
        np.where(out["alert_required"] == 1,
                 "CẢNH BÁO DỮ LIỆU CŨ: không trình bày như AQI/dự báo hiện tại.",
                 "Trong ngưỡng độ mới dự án; vẫn cần kiểm tra nguồn/sensor."))
    return out[["station_id", "station", "latitude", "longitude", "last_observation_utc",
                "last_observation_local", "observed_pm25", "hours_since_last_obs",
                "freshness_status", "alert_required", "alert_message"]].sort_values(
                    "hours_since_last_obs", ascending=False, na_position="first").reset_index(drop=True)

def build_station_tables(df, df_all):
    base = df.groupby("station", as_index=False).agg(latitude=("latitude", "first"), longitude=("longitude", "first"),
                                                     first_obs_local=("datetime_local", "min"),
                                                     last_obs_local=("datetime_local", "max"),
                                                     n_hours=("pm25", "count"), pm25_mean=("pm25", "mean"))
    base["lat_lon"] = base["latitude"].round(5).astype(str) + "," + base["longitude"].round(5).astype(str)
    base["pm25_mean"] = base["pm25_mean"].round(1)
    cov = df_all.copy()
    cov["year"] = (pd.to_datetime(cov["datetime"]) + pd.Timedelta(hours=7)).dt.year
    cov = cov.groupby(["station", "year"], as_index=False).agg(n_hours=("pm25", "count"))
    _year_start = pd.to_datetime(cov["year"].astype(str) + "-01-01")
    _next_year = pd.to_datetime((cov["year"] + 1).astype(str) + "-01-01")
    cov["hours_in_year"] = ((_next_year - _year_start).dt.total_seconds() / 3600).astype(int)
    cov["pct_of_year"] = (cov["n_hours"] / cov["hours_in_year"] * 100).round(1)

    # Thêm mẫu số YTD cho năm dữ liệu mới nhất để so sánh với thời gian đã trôi qua.
    _utc = pd.to_datetime(df_all["datetime"], errors="coerce", utc=True)
    _latest_local = _utc.dt.tz_convert("Asia/Ho_Chi_Minh").dt.tz_localize(None).max()
    if pd.notna(_latest_local):
        _latest_local = _latest_local.floor("h")
        _elapsed_end = _latest_local + pd.Timedelta(hours=1)  # tính cả giờ quan trắc cuối
        _elapsed_hours = ((_elapsed_end - _year_start).dt.total_seconds() / 3600).clip(lower=1)
        cov["hours_elapsed_in_period"] = np.where(cov["year"] == _latest_local.year, _elapsed_hours, cov["hours_in_year"]).astype(int)
        cov["pct_of_elapsed_period"] = (cov["n_hours"] / cov["hours_elapsed_in_period"] * 100).round(1)
        cov["coverage_basis"] = np.where(cov["year"] == _latest_local.year,
                                          f"YTD đến {_latest_local:%Y-%m-%d %H:%M} giờ địa phương",
                                          "Năm lịch hoàn chỉnh")
    else:
        cov["hours_elapsed_in_period"] = cov["hours_in_year"]
        cov["pct_of_elapsed_period"] = cov["pct_of_year"]
        cov["coverage_basis"] = "Mẫu số năm lịch; không xác định được timestamp cuối"
    return base, cov


def save_preview_png(fact, metrics, daily, observed):
    fig, axes = plt.subplots(2, 2, figsize=(18, 11))
    fig.suptitle("DASHBOARD DỰ BÁO Ô NHIỄM KHÔNG KHÍ HÀ NỘI (OpenAQ + Open-Meteo | PySpark)", fontsize=15, fontweight="bold")
    f1 = fact[(fact["horizon_h"] == fact["horizon_h"].min())]
    st = f1["station"].value_counts().idxmax()                                   # 1 trạm có nhiều dữ liệu test nhất
    f1 = f1[f1["station"] == st].sort_values("target_time_local")
    last_t = f1["target_time_local"].max()
    f1 = f1[f1["target_time_local"] >= last_t - pd.Timedelta(hours=168)]
    ax = axes[0, 0]
    ax.plot(f1.loc[f1["model"] == "XGBoost", "target_time_local"], f1.loc[f1["model"] == "XGBoost", "y_true"], color="black", lw=2, label="Thực tế")
    for m, c, ls in [("Random Forest", "#1f77b4", "--"), ("XGBoost", "#ff7f0e", "-."), ("Persistence (baseline)", "#999999", ":")]:
        sub = f1[f1["model"] == m]
        ax.plot(sub["target_time_local"], sub["y_pred"], color=c, ls=ls, label=m)
    ax.set_title(f"1. Thực tế vs Dự báo (+{f1['horizon_h'].iloc[0]}h) - trạm {st} - 7 ngày cuối tập Test", fontweight="bold")
    ax.set_ylabel("PM2.5 (µg/m³)"); ax.legend(); ax.grid(alpha=0.3); ax.tick_params(axis="x", rotation=25)

    sns.barplot(data=metrics, x="Horizon_label", y="Test_RMSE", hue="Model", ax=axes[0, 1], palette="Set2")
    axes[0, 1].set_title("2. Sai số RMSE theo khung giờ (thấp hơn Persistence = mô hình có giá trị)", fontweight="bold")
    axes[0, 1].set_xlabel("Khung dự báo"); axes[0, 1].set_ylabel("Test RMSE (µg/m³)"); axes[0, 1].grid(axis="y", alpha=0.3)

    mm = daily.groupby(["year", "month"])["pm25_mean"].mean().unstack(0)
    mm.plot(ax=axes[1, 0], marker="o")
    axes[1, 0].set_title("3. PM2.5 trung bình theo tháng và năm (phân tích mùa vụ)", fontweight="bold")
    axes[1, 0].set_xlabel("Tháng"); axes[1, 0].set_ylabel("µg/m³"); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].text(0.01, -0.22, "Lưu ý: độ phủ và số trạm thay đổi theo năm; xem như mô tả, không phải so sánh hoàn toàn đồng nhất.",
                    transform=axes[1, 0].transAxes, fontsize=8)

    cnt = observed["aqi_category"].value_counts().sort_index()
    sns.barplot(x=cnt.values, y=cnt.index, ax=axes[1, 1], hue=cnt.index, palette="Spectral_r", legend=False)
    axes[1, 1].set_title("4. Phân bố mức VN_AQI (NowCast, QĐ 1459/QĐ-TCMT)", fontweight="bold")
    axes[1, 1].set_xlabel("Số bản ghi trạm–giờ"); axes[1, 1].set_ylabel("")
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    path = os.path.join(OUTPUT_DIR, "Dashboard_Preview.png")
    plt.savefig(path, dpi=150)
    try:
        plt.show()
    except Exception:
        pass
    plt.close(fig)
    return path


# ===== Original notebook code cell 20 =====

# ==============================================================================
# BƯỚC 6: XUẤT FILE (CSV ≤ 100MB cho Looker Studio, tuỳ chọn BigQuery)
# ==============================================================================
def write_csv_limited(df, name, time_col=None, max_mb=None):
    """Ghi CSV UTF-8. Nếu > max_mb thì giữ phần DỮ LIỆU MỚI NHẤT (Looker Studio giới hạn 100MB/file)."""
    if max_mb is None:
        max_mb = LOOKER_MAX_MB
    path = os.path.join(OUTPUT_DIR, name)
    df.to_csv(path, index=False, encoding="utf-8")
    size = os.path.getsize(path) / 1e6
    if size > max_mb:
        if time_col is not None and pd.api.types.is_datetime64_any_dtype(df[time_col]):
            keep = max_mb * 0.95 / size
            cutoff = df[time_col].quantile(1 - keep)
            df = df[df[time_col] >= cutoff]
            df.to_csv(path, index=False, encoding="utf-8")
            print(f"   [!] {name}: {size:.0f}MB > {max_mb}MB → chỉ giữ dữ liệu từ {cutoff} trở đi "
                  f"({os.path.getsize(path) / 1e6:.0f}MB). Muốn đủ toàn bộ: dùng BigQuery (đặt BQ_PROJECT_ID).")
        else:
            print(f"   [!] {name} nặng {size:.0f}MB (> {max_mb}MB) và không thể tự cắt - hãy dùng BigQuery.")
    else:
        print(f"   ✓ {name}: {len(df):,} dòng, {size:.1f}MB")
    return path


def upload_to_bigquery(tables):
    if not BQ_PROJECT_ID:
        return
    print(f"\n-> Đẩy {len(tables)} bảng lên BigQuery ({BQ_PROJECT_ID}.{BQ_DATASET})...")
    try:
        from google.colab import auth
        auth.authenticate_user()
        _ensure("pandas-gbq", "pandas_gbq")
        import pandas_gbq
        from google.cloud import bigquery
        bigquery.Client(project=BQ_PROJECT_ID).create_dataset(f"{BQ_PROJECT_ID}.{BQ_DATASET}", exists_ok=True)
        for name, tb in tables.items():
            pandas_gbq.to_gbq(tb, f"{BQ_DATASET}.{name}", project_id=BQ_PROJECT_ID, if_exists="replace")
            print(f"   ✓ {BQ_DATASET}.{name} ({len(tb):,} dòng)")
    except Exception as exc:
        print(f"   [!] Không đẩy được lên BigQuery: {exc}")


def _metric_group_row(g, model_name, group_values, baseline_rmse, date_col="issue_time_local"):
    y = pd.to_numeric(g["y_true"], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(g["y_pred"], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(y) & np.isfinite(p)
    y, p = y[ok], p[ok]
    if len(y) == 0:
        return None
    mae = float(np.mean(np.abs(y - p)))
    rmse = float(np.sqrt(np.mean((y - p) ** 2)))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1 - np.sum((y - p) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    row = dict(group_values)
    row.update({"model": model_name, "n_samples": int(len(y)), "MAE": round(mae, 3),
                "RMSE": round(rmse, 3), "R2": round(r2, 4) if np.isfinite(r2) else np.nan,
                "baseline_persistence_RMSE": round(float(baseline_rmse), 3) if np.isfinite(baseline_rmse) else np.nan,
                "skill_vs_persistence_pct": round((1 - rmse / baseline_rmse) * 100, 2) if np.isfinite(baseline_rmse) and baseline_rmse > 0 else np.nan})
    if date_col in g.columns and len(g):
        dates = pd.to_datetime(g[date_col], errors="coerce")
        row["issue_time_min"] = dates.min()
        row["issue_time_max"] = dates.max()
    return row


def build_forecast_metric_breakdowns(fact):
    """Metrics theo trạm/quý từ dự báo test. Gọi khi fact test đầy đủ, trước khi cắt CSV dashboard."""
    if fact is None or fact.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = fact.copy()
    for c in ["issue_time_local", "target_time_local"]:
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")
    if "horizon_label" not in d.columns:
        d["horizon_label"] = d["horizon_h"].map(lambda h: f"+{int(h):02d}h")

    def make_summary(group_cols, period_col=None):
        baseline = {}
        for key, g in d[d["model"] == "Persistence (baseline)"].groupby(group_cols, dropna=False):
            if not isinstance(key, tuple): key = (key,)
            y = pd.to_numeric(g["y_true"], errors="coerce").to_numpy(float)
            p = pd.to_numeric(g["y_pred"], errors="coerce").to_numpy(float)
            ok = np.isfinite(y) & np.isfinite(p)
            baseline[key] = float(np.sqrt(np.mean((y[ok] - p[ok]) ** 2))) if ok.any() else np.nan
        rows = []
        for key, g in d.groupby(group_cols + ["model"], dropna=False):
            if not isinstance(key, tuple): key = (key,)
            group_key, model_name = key[:-1], key[-1]
            vals = dict(zip(group_cols, group_key))
            row = _metric_group_row(g, model_name, vals, baseline.get(tuple(group_key), np.nan))
            if row:
                rows.append(row)
        out = pd.DataFrame(rows)
        if period_col and not out.empty:
            out["period_type"] = "quarter"
            out["period_note"] = "Các quý đầu/cuối có thể chưa đầy đủ; xem khoảng issue_time_min/max."
        return out

    by_station = make_summary(["station", "horizon_h", "horizon_label"])
    d["quarter"] = d["issue_time_local"].dt.to_period("Q").astype(str)
    by_quarter = make_summary(["quarter", "horizon_h", "horizon_label"], period_col="quarter")
    if not by_quarter.empty:
        by_quarter["quarter_calendar_start"] = pd.PeriodIndex(by_quarter["quarter"], freq="Q").start_time
        by_quarter["quarter_calendar_end"] = pd.PeriodIndex(by_quarter["quarter"], freq="Q").end_time.floor("h")
        by_quarter["period_is_partial_in_export"] = ((pd.to_datetime(by_quarter["issue_time_min"]) > pd.to_datetime(by_quarter["quarter_calendar_start"])) |
                                                        (pd.to_datetime(by_quarter["issue_time_max"]) < pd.to_datetime(by_quarter["quarter_calendar_end"]))).astype(int)
    return by_station, by_quarter


def build_quality_review_tables(observed):
    """Tạo bảng gắn cờ chất lượng dữ liệu; không tự động xóa số đo bị nghi ngờ."""
    if observed is None or observed.empty:
        return pd.DataFrame(), pd.DataFrame()
    d = observed.copy()
    flagged = pd.to_numeric(d.get("aqi_qc_review_flag", 0), errors="coerce").fillna(0).astype(int) == 1
    review = d[flagged].copy()
    counts = d.groupby("station", as_index=False).agg(n_observed_rows=("pm25", "size"))
    if review.empty:
        summary = counts.copy()
        summary["n_review_rows"] = 0
        summary["pct_review_rows"] = 0.0
        return review, summary
    summary = review.groupby("station", as_index=False).agg(
        n_review_rows=("aqi_qc_review_flag", "size"), pm25_max_in_review=("pm25", "max"),
        first_review_time=("datetime_local", "min"), last_review_time=("datetime_local", "max"))
    summary = summary.merge(counts, on="station", how="left")
    summary["pct_review_rows"] = (summary["n_review_rows"] / summary["n_observed_rows"] * 100).round(2)
    summary["review_note"] = "Cờ rà soát, không tự động xóa; xác minh timestamp, sensor và đơn vị trước khi xử lý."
    return review, summary.sort_values("n_review_rows", ascending=False)


def export_dashboard_tables(df, df_all, df_metrics, fact, latest):
    print("\n=== [5/6] Dựng các bảng cho dashboard... ===")
    observed = build_observed_hourly(df)
    daily = build_daily_summary(df, df_all)
    dim_station, coverage = build_station_tables(df, df_all)
    freshness = build_station_freshness(df_all)
    quality_rows, quality_by_station = build_quality_review_tables(observed)
    png = save_preview_png(fact, df_metrics, daily, observed)

    # Nếu fact test đầy đủ, tính breakdown trước khi write_csv_limited cắt bảng lớn cho dashboard.
    station_metric_path = os.path.join(OUTPUT_DIR, "model_metrics_by_station.csv")
    quarter_metric_path = os.path.join(OUTPUT_DIR, "model_metrics_by_quarter.csv")
    n_models = int(df_metrics["Model"].nunique()) if "Model" in df_metrics.columns else 0
    expected_fact_rows = int(df_metrics.drop_duplicates("Horizon_h")["N_Test"].sum() * n_models) if {"Horizon_h", "N_Test"}.issubset(df_metrics.columns) else 0
    fact_is_full = expected_fact_rows > 0 and len(fact) >= expected_fact_rows
    if fact_is_full:
        metrics_by_station, metrics_by_quarter = build_forecast_metric_breakdowns(fact)
        metrics_by_station.to_csv(station_metric_path, index=False, encoding="utf-8")
        metrics_by_quarter.to_csv(quarter_metric_path, index=False, encoding="utf-8")
        print("-> Đã tạo metrics theo trạm/quý từ fact test đầy đủ.")
    elif os.path.exists(station_metric_path) and os.path.exists(quarter_metric_path):
        metrics_by_station = pd.read_csv(station_metric_path)
        metrics_by_quarter = pd.read_csv(quarter_metric_path)
        print("-> Giữ metrics theo trạm/quý từ lần huấn luyện đầy đủ gần nhất (fact dashboard có thể bị giới hạn kích thước).")
    else:
        metrics_by_station, metrics_by_quarter = build_forecast_metric_breakdowns(fact)
        if not metrics_by_station.empty:
            metrics_by_station["metric_scope_note"] = "Tạm tính từ fact hiện có; xác minh phạm vi thời gian vì fact có thể bị giới hạn kích thước."
        if not metrics_by_quarter.empty:
            metrics_by_quarter["metric_scope_note"] = "Tạm tính từ fact hiện có; xác minh phạm vi thời gian vì fact có thể bị giới hạn kích thước."
        metrics_by_station.to_csv(station_metric_path, index=False, encoding="utf-8")
        metrics_by_quarter.to_csv(quarter_metric_path, index=False, encoding="utf-8")

    quality_rows.to_csv(os.path.join(OUTPUT_DIR, "quality_review_rows.csv"), index=False, encoding="utf-8")
    quality_by_station.to_csv(os.path.join(OUTPUT_DIR, "quality_review_summary_by_station.csv"), index=False, encoding="utf-8")

    # Dọn file cũ từ phiên bản trước để ZIP không còn bảng COVID.
    obsolete_covid = os.path.join(OUTPUT_DIR, "covid_compare.csv")
    if os.path.exists(obsolete_covid):
        os.remove(obsolete_covid)
        print("-> Đã bỏ covid_compare.csv theo phạm vi mới của đề tài.")

    # Chỉ dùng các dự báo còn hiệu lực (quan trắc đủ mới + thời điểm đích chưa qua) cho trang hiện tại.
    if "is_current_forecast" in latest.columns:
        latest_current = latest.loc[latest["is_current_forecast"] == 1].copy()
    else:
        latest_current = latest.iloc[0:0].copy()
    print(f"-> Dự báo đủ điều kiện hiển thị hiện tại: {len(latest_current):,}/{len(latest):,} dòng.")

    print("\n=== [6/6] Xuất CSV UTF-8 cho dashboard... ===")
    write_csv_limited(fact, "fact_forecast_long.csv", "issue_time_local")
    write_csv_limited(observed, "observed_hourly.csv", "datetime_local")
    write_csv_limited(daily, "daily_summary.csv", "date")
    write_csv_limited(latest, "latest_forecast.csv")
    write_csv_limited(latest_current, "latest_forecast_current.csv", "target_time_local")
    write_csv_limited(df_metrics, "model_metrics.csv")
    write_csv_limited(dim_station, "dim_station.csv")
    write_csv_limited(coverage, "data_coverage.csv")
    write_csv_limited(freshness, "station_freshness.csv")
    write_csv_limited(metrics_by_station, "model_metrics_by_station.csv")
    write_csv_limited(metrics_by_quarter, "model_metrics_by_quarter.csv")
    write_csv_limited(quality_rows, "quality_review_rows.csv")
    write_csv_limited(quality_by_station, "quality_review_summary_by_station.csv")

    latest_dt = pd.to_datetime(df_all["datetime"], errors="coerce", utc=True).max()
    freshness_utc = pd.to_numeric(freshness["hours_since_last_obs"], errors="coerce")
    metadata_path = os.path.join(MODEL_DIR, "training_metadata.json")
    trained_at = ""
    if os.path.exists(metadata_path):
        try:
            trained_at = json.load(open(metadata_path, encoding="utf-8")).get("trained_at_utc", "")
        except Exception:
            pass
    status = pd.DataFrame([{
        "run_generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "latest_openAQ_timestamp_utc": latest_dt.isoformat() if pd.notna(latest_dt) else "",
        "stations_in_raw": int(freshness["station_id"].nunique()),
        "stations_with_pm25_observation_le_threshold": int((freshness_utc <= MAX_FORECAST_AGE_HOURS).sum()),
        "stations_with_pm25_observation_gt_threshold": int((freshness_utc > MAX_FORECAST_AGE_HOURS).sum()),
        "stations_without_valid_pm25": int(freshness_utc.isna().sum()),
        "freshness_threshold_hours_project": int(MAX_FORECAST_AGE_HOURS),
        "models_last_trained_at_utc": trained_at,
        "aqi_method": "VN_AQI formula per QD 1459; input sensor QC remains necessary; forecast PM2.5 index is proxy only",
        "pipeline_version": PIPELINE_VERSION,
        "automatic_alert_delivery": "No; alert_required is a dashboard data/freshness flag, not email/SMS delivery",
        "forecast_rows_total": int(len(latest)),
        "forecast_rows_usable_current": int(latest["is_current_forecast"].sum()) if "is_current_forecast" in latest else 0,
        "forecast_rows_stale_or_expired": int(latest["alert_required"].sum()) if "alert_required" in latest else int(len(latest)),
        "data_update_strategy": "Incremental; active sensors queried each run; metadata-stale sensors rechecked no more often than daily; recent window upserted"
    }])
    write_csv_limited(status, "run_status.csv")

    upload_to_bigquery({"fact_forecast_long": fact, "observed_hourly": observed, "daily_summary": daily,
                        "latest_forecast": latest, "latest_forecast_current": latest_current,
                        "model_metrics": df_metrics, "model_metrics_by_station": metrics_by_station,
                        "model_metrics_by_quarter": metrics_by_quarter, "quality_review_rows": quality_rows,
                        "quality_review_summary_by_station": quality_by_station, "dim_station": dim_station,
                        "data_coverage": coverage, "station_freshness": freshness, "run_status": status})

    zip_base = os.path.join(DATA_ROOT, "looker_studio_export")
    zip_path = shutil.make_archive(zip_base, "zip", OUTPUT_DIR)
    print(f"\n-> HOÀN TẤT. Thư mục: '{OUTPUT_DIR}' | Gói tải về: '{zip_path}' | Ảnh xem trước: '{png}'")
    if AUTO_DOWNLOAD_EXPORT:
        try:
            from google.colab import files
            files.download(zip_path)
        except Exception:
            pass


def _gcs_bucket_client():
    if not GCS_BUCKET:
        return None
    from google.cloud import storage
    return storage.Client(project=GCP_PROJECT_ID or None).bucket(GCS_BUCKET)


def sync_inputs_from_gcs():
    """Khôi phục RAW và mô hình/kết quả trước lần chạy theo lịch; không có bucket thì chạy cục bộ."""
    bucket = _gcs_bucket_client()
    if bucket is None:
        print("-> GCS persistence chưa bật; dữ liệu lưu tại DATA_ROOT của runtime hiện tại.")
        return
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for path in [AQ_FILE, WEATHER_FILE, SENSOR_POLL_STATE_FILE]:
        remote = f"{GCS_PREFIX}/raw/{os.path.basename(path)}"
        blob = bucket.blob(remote)
        if blob.exists():
            blob.download_to_filename(path)
            print(f"-> Khôi phục RAW từ gs://{GCS_BUCKET}/{remote}")
    prefix = f"{GCS_PREFIX}/output/"
    # Dọn bảng đã bỏ khỏi phiên bản mới nếu còn sót trên GCS.
    old_covid_blob = bucket.blob(f"{prefix}covid_compare.csv")
    try:
        if old_covid_blob.exists():
            old_covid_blob.delete()
            print("-> Đã xóa bảng covid_compare.csv cũ khỏi output trên GCS.")
    except Exception as exc:
        print(f"[!] Không xóa được bảng COVID cũ trên GCS: {exc}")
    for blob in bucket.list_blobs(prefix=prefix):
        rel = blob.name[len(prefix):]
        if rel:
            local = os.path.join(OUTPUT_DIR, rel)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            blob.download_to_filename(local)
    print("-> Đã đồng bộ model và CSV đầu ra từ GCS nếu tồn tại.")


def sync_outputs_to_gcs():
    """Lưu RAW, model và CSV/ZIP cập nhật lên GCS sau mỗi lần chạy scheduled."""
    bucket = _gcs_bucket_client()
    if bucket is None:
        return
    items = [
        (AQ_FILE, f"{GCS_PREFIX}/raw/{os.path.basename(AQ_FILE)}"),
        (WEATHER_FILE, f"{GCS_PREFIX}/raw/{os.path.basename(WEATHER_FILE)}"),
        (SENSOR_POLL_STATE_FILE, f"{GCS_PREFIX}/raw/{os.path.basename(SENSOR_POLL_STATE_FILE)}"),
        (os.path.join(DATA_ROOT, "looker_studio_export.zip"), f"{GCS_PREFIX}/looker_studio_export.zip"),
    ]
    for root, _, names in os.walk(OUTPUT_DIR):
        for name in names:
            local = os.path.join(root, name)
            rel = os.path.relpath(local, OUTPUT_DIR).replace(os.sep, "/")
            items.append((local, f"{GCS_PREFIX}/output/{rel}"))
    errors = []
    count = 0
    for local, remote in items:
        if not os.path.exists(local):
            continue
        try:
            bucket.blob(remote).upload_from_filename(local)
            count += 1
        except Exception as exc:
            errors.append((remote, str(exc)))
    if errors:
        raise RuntimeError(f"Không đồng bộ được {len(errors)} file lên GCS; ví dụ: {errors[:3]}")
    print(f"-> Đã đẩy {count} file lên gs://{GCS_BUCKET}/{GCS_PREFIX}/")


def main():
    if not OPENAQ_API_KEY or OPENAQ_API_KEY in {"DÁN_API_KEY_OPENAQ_VÀO_ĐÂY", "DAN_API_KEY_OPENAQ_VAO_DAY"}:
        raise RuntimeError(
            "Chưa có OPENAQ_API_KEY hợp lệ. Vào Colab Secrets (biểu tượng chìa khóa), "
            "tạo secret OPENAQ_API_KEY, dán API key mới và bật quyền truy cập cho notebook; "
            "sau đó chạy lại từ đầu."
        )
    t0 = time.time()
    os.makedirs(DATA_ROOT, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sync_inputs_from_gcs()
    aq_csv, weather_csv = ingest_raw_data()
    df_feat, df_all = spark_bigdata_pipeline(aq_csv, weather_csv)
    df = prepare_features(df_feat, df_all)

    saved = load_saved_state()
    if saved is not None and not RETRAIN_MODELS_EVERY_RUN:
        feats, df_metrics, fact, xgb_models, xgb_best_n, metadata = saved
        if not set(feats).issubset(df.columns):
            print("[!] Dữ liệu cập nhật thiếu cột feature của model đã lưu; sẽ huấn luyện lại.")
            saved = None
    if saved is not None and not RETRAIN_MODELS_EVERY_RUN:
        # Các bản ghi mới cập nhật vào bảng quan trắc; metrics test vẫn ghi là lần huấn luyện gần nhất.
        df["split"] = "Cập nhật dữ liệu (dùng model đã lưu)"
        print(f"-> Dùng metrics/model đã huấn luyện lúc {metadata.get('trained_at_utc', 'không rõ')}; không train lại lần này.")
    else:
        df, feats, df_metrics, fact, xgb_models, xgb_best_n = train_multi_horizon(df)
        save_models(feats, xgb_models)

    latest = build_latest_forecast(df, feats, xgb_models, xgb_best_n)
    export_dashboard_tables(df, df_all, df_metrics, fact, latest)
    sync_outputs_to_gcs()
    print(f"Tổng thời gian: {(time.time() - t0) / 60:.1f} phút")


if __name__ == "__main__":
    main()
