from __future__ import annotations

import os
from pathlib import Path

from drive_sync import download_project_state, sync_dashboard_sheets, upload_project_state


def main() -> None:
    data_root = Path(os.environ.get("HANOI_PM25_DATA_ROOT", "runtime_data")).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    print("========== 1/4 TẢI RAW + STATE TỪ DRIVE ==========")
    download_project_state(data_root)

    # The converted V5 pipeline uses this path instead of Colab's drive.mount().
    os.environ["HANOI_PM25_DATA_ROOT"] = str(data_root)
    os.environ["COLAB_SCHEDULED_RUN"] = "true"

    import hanoi_pm25_pipeline as pipeline

    print("========== 2/4 CHẠY PIPELINE PM2.5 ==========")
    pipeline.main()

    print("========== 3/4 LƯU RAW / CACHE / MODEL / CSV VỀ DRIVE ==========")
    upload_project_state(data_root)

    print("========== 4/4 CẬP NHẬT NGUỒN GOOGLE SHEETS CHO LOOKER ==========")
    sync_dashboard_sheets(data_root / "output_hanoi_pm25")
    print("✅ Pipeline hoàn tất; đã gửi các bảng được hỗ trợ lên Drive/Sheets.")


if __name__ == "__main__":
    main()
