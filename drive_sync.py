from __future__ import annotations

import base64
import binascii
import os
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import requests

# Small chunks keep Apps Script JSON responses below response-size limits.
CHUNK_BYTES = 256 * 1024
SHEET_CHUNK_ROWS = 1000
MAX_APPS_SCRIPT_FILE_BYTES = 40 * 1024 * 1024
RAW_EXPECTED = {"openaq_hanoi_2019_2026_raw.csv", "open_meteo_hanoi_2019_2026_raw.csv"}
RAW_ALIASES = {
    "openaq_hanoi_2019_2026_raw.csv": ["openaq_hanoi_2019_2026_raw (1).csv", "openaq_hanoi_2019_2026_raw(1).csv"],
    "open_meteo_hanoi_2019_2026_raw.csv": ["open_meteo_hanoi_2019_2026_raw (1).csv", "open_meteo_hanoi_2019_2026_raw(1).csv"],
}
PERSISTENT_ROOT_FILES = RAW_EXPECTED | {alias for aliases in RAW_ALIASES.values() for alias in aliases} | {"sensor_poll_state.csv"}
PERSISTENT_DIRS = {"cache_openaq", "output_hanoi_pm25"}
SHEETS_TABLES = [
    ("latest_forecast_current", "latest_forecast_current.csv"),
    ("latest_forecast", "latest_forecast.csv"),
    ("observed_hourly", "observed_hourly.csv"),
    ("daily_summary", "daily_summary.csv"),
    ("model_metrics", "model_metrics.csv"),
    ("station_freshness", "station_freshness.csv"),
    ("data_coverage", "data_coverage.csv"),
    ("run_status", "run_status.csv"),
]
SKIP_SUFFIXES = {".zip", ".pdf", ".docx", ".xlsx", ".ipynb", ".png", ".jpg", ".jpeg", ".svg", ".gif"}
SKIP_DOWNLOAD_NAMES = {
    "fact_forecast_long.csv", "fact_forecast_long.csv.tmp", "observed_hourly.csv",
    "daily_summary.csv", "latest_forecast.csv", "latest_forecast_current.csv", "quality_review_rows.csv",
}


def _bridge_url() -> str:
    url = os.environ.get("GOOGLE_APPS_SCRIPT_URL", "").strip()
    if not url:
        raise RuntimeError("Thiếu GitHub Secret GOOGLE_APPS_SCRIPT_URL. Dùng URL Web App kết thúc bằng /exec.")
    return url


def _bridge_token() -> str:
    token = os.environ.get("GOOGLE_DRIVE_BRIDGE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("Thiếu GitHub Secret GOOGLE_DRIVE_BRIDGE_TOKEN.")
    return token


def _call(action: str, **kwargs: Any) -> dict[str, Any]:
    payload = {"action": action, "token": _bridge_token(), **kwargs}
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            response = requests.post(_bridge_url(), json=payload, timeout=300, allow_redirects=True)
            response.raise_for_status()
            try:
                data = response.json()
            except ValueError as exc:
                preview = response.text[:500].replace("\n", " ")
                raise RuntimeError(f"Apps Script không trả JSON ({action}). Phản hồi: {preview}") from exc
            if not isinstance(data, dict):
                raise RuntimeError(f"Apps Script trả dữ liệu không phải JSON object ({action})")
            if not data.get("ok"):
                raise RuntimeError(str(data.get("error", f"Apps Script lỗi khi chạy {action}")))
            return data
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Không kết nối được Apps Script ({action}): {last_error}") from last_error


def _safe_path(relative_path: str) -> Path:
    p = PurePosixPath(relative_path)
    if p.is_absolute() or ".." in p.parts or not p.parts or "\\" in relative_path or ":" in relative_path:
        raise ValueError(f"Đường dẫn từ Apps Script không hợp lệ: {relative_path!r}")
    return Path(*p.parts)


def _download_chunk(relative_path: str, offset: int, chunk_size: int) -> bytes:
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            result = _call("download_chunk", path=relative_path, offset=offset, chunkSize=chunk_size)
            encoded = result.get("base64")
            if not isinstance(encoded, str):
                keys = sorted(result.keys())
                raise RuntimeError(
                    f"Apps Script không trả 'base64' khi tải {relative_path!r}, "
                    f"offset={offset}, chunkSize={chunk_size}. Keys={keys}. "
                    "Kiểm tra Code.gs action download_chunk và deploy New version."
                )
            chunk = base64.b64decode(encoded, validate=True)
            if not chunk or len(chunk) > chunk_size:
                raise RuntimeError(f"Chunk không hợp lệ: {len(chunk)} bytes, yêu cầu tối đa {chunk_size}")
            return chunk
        except (RuntimeError, ValueError, binascii.Error) as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Tải chunk thất bại sau 4 lần: {last_error}") from last_error


def _download_file(relative_path: str, size: int, local_root: Path) -> None:
    dest = local_root / _safe_path(relative_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if size <= 0:
        tmp.write_bytes(b"")
        tmp.replace(dest)
        print(f"  ✓ Tạo file rỗng {relative_path} (0 B)")
        return
    offset = 0
    try:
        with tmp.open("wb") as fh:
            while offset < size:
                chunk = _download_chunk(relative_path, offset, min(CHUNK_BYTES, size - offset))
                fh.write(chunk)
                offset += len(chunk)
        if offset != size:
            raise RuntimeError(f"Tải '{relative_path}' không đủ dữ liệu: {offset}/{size} bytes")
        tmp.replace(dest)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    print(f"  ✓ Tải {relative_path} ({size / (1024 * 1024):.1f} MiB)")


def _download_project_manifest() -> list[dict[str, Any]]:
    files = _call("list_files").get("files", [])
    if not isinstance(files, list):
        raise RuntimeError("Apps Script list_files không trả danh sách files")
    names_at_root = {x["path"] for x in files if isinstance(x, dict) and "path" in x and "/" not in x["path"]}
    missing = [expected for expected, aliases in RAW_ALIASES.items()
               if expected not in names_at_root and not any(alias in names_at_root for alias in aliases)]
    if missing:
        raise FileNotFoundError("Thiếu RAW trên Drive: " + ", ".join(missing))
    return files


def download_project_state(local_root: Path) -> None:
    local_root = Path(local_root)
    local_root.mkdir(parents=True, exist_ok=True)
    files = _download_project_manifest()
    print(f"-> Tìm thấy {len(files)} file trạng thái/RAW/đầu ra trên Drive.")
    for item in files:
        rel_path = item["path"]
        file_name = Path(rel_path).name
        file_size = int(item.get("size", 0))
        mime_type = str(item.get("mimeType", ""))
        if (mime_type.startswith("application/vnd.google-apps.") or file_name in SKIP_DOWNLOAD_NAMES
                or Path(rel_path).suffix.lower() in SKIP_SUFFIXES or file_size > MAX_APPS_SCRIPT_FILE_BYTES):
            print(f"  [-] Bỏ qua tải file: {rel_path} ({file_size / (1024 * 1024):.1f} MiB)")
            continue
        try:
            _download_file(rel_path, file_size, local_root)
        except Exception as exc:
            is_critical = (file_name in RAW_EXPECTED or
                           any(file_name in aliases for aliases in RAW_ALIASES.values()) or
                           file_name == "sensor_poll_state.csv")
            if is_critical:
                raise
            print(f"  [!] Cảnh báo: Bỏ qua {rel_path} do lỗi tải: {exc}")
    for canonical, aliases in RAW_ALIASES.items():
        destination = local_root / canonical
        if destination.exists():
            continue
        for alias in aliases:
            candidate = local_root / alias
            if candidate.exists():
                candidate.rename(destination)
                break
    missing = [name for name in RAW_EXPECTED if not (local_root / name).exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu file RAW sau khi tải từ Drive: {missing}")


def _iter_state_files(local_root: Path):
    for name in sorted(PERSISTENT_ROOT_FILES):
        path = local_root / name
        if path.is_file():
            yield path, path.relative_to(local_root).as_posix()
    for dirname in sorted(PERSISTENT_DIRS):
        base = local_root / dirname
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file() or path.suffix.lower() in SKIP_SUFFIXES:
                continue
            if "fact_forecast_long" in path.name or "observed_hourly" in path.name:
                print(f"  [-] Bỏ qua upload (tránh ghi đè lịch sử / quá dung lượng Cache Drive): {path.name}")
                continue
            if path.stat().st_size > MAX_APPS_SCRIPT_FILE_BYTES:
                print(f"  [-] Bỏ qua upload file quá lớn (>40MB): {path.name}")
                continue
            yield path, path.relative_to(local_root).as_posix()


def _upload_file(local_file: Path, relative_path: str) -> None:
    upload_id = uuid.uuid4().hex
    total_size = local_file.stat().st_size
    total_chunks = max(1, (total_size + CHUNK_BYTES - 1) // CHUNK_BYTES)
    with local_file.open("rb") as fh:
        for index in range(total_chunks):
            chunk = fh.read(CHUNK_BYTES)
            _call("upload_chunk", uploadId=upload_id, path=relative_path, chunkIndex=index,
                  totalChunks=total_chunks, base64=base64.b64encode(chunk).decode("ascii"))
    _call("finish_upload", uploadId=upload_id, path=relative_path, totalChunks=total_chunks)
    print(f"   ✓ Ghi {relative_path} ({total_size / (1024 * 1024):.1f} MiB)")


def upload_project_state(local_root: Path) -> None:
    count = 0
    for local_file, rel in _iter_state_files(Path(local_root)):
        _upload_file(local_file, rel)
        count += 1
    print(f"-> Đã gửi {count} file RAW/state/cache/output về Drive.")


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _sync_csv_sheet(title: str, csv_path: Path) -> None:
    frame = pd.read_csv(csv_path, low_memory=False)
    if title == "observed_hourly" and len(frame) > 25000:
        frame = (frame.sort_values("datetime") if "datetime" in frame.columns else frame).tail(25000).reset_index(drop=True)
        print(f"   [i] Rút gọn '{title}' xuống 25,000 dòng mới nhất.")
    rows_total, cols = frame.shape[0] + 1, frame.shape[1]
    if rows_total * max(cols, 1) > 9_500_000:
        print(f"[!] Bỏ qua Google Sheets '{title}': {rows_total:,} × {cols:,} ô gần/vượt giới hạn.")
        return
    headers = [str(col) for col in frame.columns]
    init = _call("begin_sheet", title=title, headers=headers, totalRows=rows_total, totalCols=cols)
    spreadsheet_id = init["spreadsheetId"]
    url = init.get("url", f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}/edit")
    for start in range(0, len(frame), SHEET_CHUNK_ROWS):
        part = frame.iloc[start:start + SHEET_CHUNK_ROWS]
        values = [[_json_value(v) for v in row] for row in part.itertuples(index=False, name=None)]
        _call("write_sheet_rows", spreadsheetId=spreadsheet_id, headers=headers, startRow=start + 2, rows=values)
        if (start // SHEET_CHUNK_ROWS) % 10 == 0:
            print(f"   -> {title}: {min(start + len(part), len(frame)):,}/{len(frame):,} dòng")
    _call("finish_sheet", spreadsheetId=spreadsheet_id)
    print(f"-> Google Sheets đã cập nhật: {title} | {rows_total:,} hàng | {url}")


def sync_dashboard_sheets(output_dir: Path) -> None:
    for title, filename in SHEETS_TABLES:
        csv_path = Path(output_dir) / filename
        if not csv_path.is_file():
            print(f"[!] Bỏ qua '{title}': chưa có file {filename}")
            continue
        try:
            _sync_csv_sheet(title, csv_path)
        except Exception as exc:
            print(f"[!] Không đồng bộ được '{title}' sang Google Sheets: {exc}")
