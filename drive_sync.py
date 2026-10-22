from __future__ import annotations

import base64
import os
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import pandas as pd
import requests

CHUNK_BYTES = 2 * 1024 * 1024
SHEET_CHUNK_ROWS = 1000
MAX_APPS_SCRIPT_FILE_BYTES = 40 * 1024 * 1024

RAW_EXPECTED = {
    "openaq_hanoi_2019_2026_raw.csv",
    "open_meteo_hanoi_2019_2026_raw.csv",
}
RAW_ALIASES = {
    "openaq_hanoi_2019_2026_raw.csv": [
        "openaq_hanoi_2019_2026_raw (1).csv",
        "openaq_hanoi_2019_2026_raw(1).csv",
    ],
    "open_meteo_hanoi_2019_2026_raw.csv": [
        "open_meteo_hanoi_2019_2026_raw (1).csv",
        "open_meteo_hanoi_2019_2026_raw(1).csv",
    ],
}
PERSISTENT_ROOT_FILES = RAW_EXPECTED | {
    alias for aliases in RAW_ALIASES.values() for alias in aliases
} | {"sensor_poll_state.csv"}
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
    "fact_forecast_long.csv",
    "fact_forecast_long.csv.tmp",
    "observed_hourly.csv",
    "daily_summary.csv",
    "latest_forecast.csv",
    "latest_forecast_current.csv",
    "quality_review_rows.csv",
}

def _bridge_url() -> str:
    url = os.environ.get("GOOGLE_APPS_SCRIPT_URL", "").strip()
    if not url:
        raise RuntimeError("Thiếu GitHub Secret GOOGLE_APPS_SCRIPT_URL.")
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
            data = response.json()
            if not isinstance(data, dict):
                raise RuntimeError(f"Apps Script trả dữ liệu không phải JSON object: {data}")
            if not data.get("ok"):
                raise RuntimeError(data.get("error", f"Apps Script lỗi khi chạy action={action}"))
            return data
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Không kết nối được Apps Script ({action}): {last_error}") from last_error

def _safe_path(relative_path: str) -> Path:
    p = PurePosixPath(relative_path)
    if p.is_absolute() or ".." in p.parts or not p.parts:
        raise ValueError(f"Đường dẫn từ Apps Script không hợp lệ: {relative_path!r}")
    return Path(*p.parts)

def _download_file(relative_path: str, size: int, local_root: Path) -> None:
    dest = Path(local_root) / _safe_path(relative_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    if size <= 0:
        dest.touch()
        return

    offset = 0
    with tmp.open("wb") as fh:
        while offset < size:
            result = _call("download_chunk", path=relative_path, offset=offset, chunkSize=CHUNK_BYTES)
            # DÙNG .get() AN TOÀN TUYỆT ĐỐI, KHÔNG BAO GIỜ GÂY KEYERROR
            chunk_str = result.get("base64") if isinstance(result, dict) else None
            if chunk_str is None:
                raise RuntimeError(f"Lỗi tải '{relative_path}' tại offset {offset}: không nhận được base64 từ Apps Script.")
            chunk = base64.b64decode(chunk_str) if chunk_str else b""
            if not chunk and offset < size:
                break
            fh.write(chunk)
            offset += len(chunk)

    tmp.replace(dest)

def _download_project_manifest() -> list[dict[str, Any]]:
    response = _call("list_files")
    return response.get("files", [])

def download_project_state(local_root: Path) -> None:
    local_root.mkdir(parents=True, exist_ok=True)
    files = _download_project_manifest()
    print(f"-> Tìm thấy {len(files)} file trên Drive.")
    for item in files:
        rel_path = item["path"]
        file_name = Path(rel_path).name
        file_size = int(item.get("size", 0))
        mime_type = str(item.get("mimeType", ""))

        if (
            mime_type.startswith("application/vnd.google-apps.")
            or file_name in SKIP_DOWNLOAD_NAMES
            or "fact_forecast_long" in file_name
            or Path(rel_path).suffix.lower() in SKIP_SUFFIXES
            or file_size > MAX_APPS_SCRIPT_FILE_BYTES
        ):
            continue

        try:
            _download_file(rel_path, file_size, local_root)
        except Exception as exc:
            is_critical = file_name in RAW_EXPECTED or "sensor_poll_state.csv" in file_name
            if is_critical:
                raise exc

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
        raise FileNotFoundError(f"Thiếu file RAW: {missing}")

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
            if path.is_file() and path.suffix.lower() not in SKIP_SUFFIXES:
                if "fact_forecast_long" in path.name or path.stat().st_size > MAX_APPS_SCRIPT_FILE_BYTES:
                    continue
                yield path, path.relative_to(local_root).as_posix()

def _upload_file(local_file: Path, relative_path: str) -> None:
    upload_id = uuid.uuid4().hex
    total_size = local_file.stat().st_size
    total_chunks = max(1, (total_size + CHUNK_BYTES - 1) // CHUNK_BYTES)
    with local_file.open("rb") as fh:
        for index in range(total_chunks):
            chunk = fh.read(CHUNK_BYTES)
            _call(
                "upload_chunk",
                uploadId=upload_id,
                path=relative_path,
                chunkIndex=index,
                totalChunks=total_chunks,
                base64=base64.b64encode(chunk).decode("ascii"),
            )
    _call("finish_upload", uploadId=upload_id, path=relative_path, totalChunks=total_chunks)

def upload_project_state(local_root: Path) -> None:
    for local_file, rel in _iter_state_files(local_root):
        _upload_file(local_file, rel)

def _json_value(value: Any) -> Any:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    return str(value)

def _sync_csv_sheet(title: str, csv_path: Path) -> None:
    frame = pd.read_csv(csv_path, low_memory=False)
    if title == "observed_hourly" and len(frame) > 20000:
        frame = frame.tail(20000).reset_index(drop=True)
    rows_total, cols = frame.shape[0] + 1, frame.shape[1]
    if rows_total * max(cols, 1) > 9_500_000:
        return
    headers = [str(col) for col in frame.columns]
    init = _call("begin_sheet", title=title, headers=headers, totalRows=rows_total, totalCols=cols)
    spreadsheet_id = init["spreadsheetId"]
    for start in range(0, len(frame), SHEET_CHUNK_ROWS):
        part = frame.iloc[start : start + SHEET_CHUNK_ROWS]
        values = [[_json_value(v) for v in row] for row in part.itertuples(index=False, name=None)]
        _call(
            "write_sheet_rows",
            spreadsheetId=spreadsheet_id,
            headers=headers,
            startRow=start + 2,
            rows=values,
        )
    _call("finish_sheet", spreadsheetId=spreadsheet_id)

def sync_dashboard_sheets(output_dir: Path) -> None:
    for title, filename in SHEETS_TABLES:
        csv_path = output_dir / filename
        if not csv_path.is_file():
            continue
        try:
            _sync_csv_sheet(title, csv_path)
        except Exception:
            pass
