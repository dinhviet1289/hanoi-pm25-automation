# Tự động cập nhật PM2.5 bằng GitHub Actions + Google Drive (không cần tự cấu hình Google Cloud Console)

Gói này dùng **Google Apps Script làm cầu nối** giữa GitHub Actions và Drive/Sheets. Mã pipeline chạy trên runner của GitHub; Drive vẫn là nơi lưu RAW, cache, model và kết quả. Không dùng Service Account và không cần vào Google Cloud Console thủ công cho thư mục chia sẻ bình thường trong My Drive.

> Điều kiện: tài khoản tạo/deploy Apps Script phải có quyền đọc/ghi vào thư mục Drive dự án. Nếu đó là **Shared Drive** thực sự của Google Workspace (không phải thư mục My Drive được chia sẻ), chính sách/quyền Shared Drive có thể yêu cầu cấu hình bổ sung.

## 1. Tạo Google Apps Script bridge

1. Mở https://script.google.com/ bằng tài khoản Google có quyền Editor với thư mục dữ liệu nhóm.
2. Chọn **New project**.
3. Mở file `apps_script/Code.gs` trong gói này, sao chép toàn bộ nội dung vào `Code.gs`.
4. Trong **Project Settings**, đặt múi giờ dự án là `Asia/Ho_Chi_Minh` để tránh lệch thời gian địa phương.
5. Mở **Project Settings** (biểu tượng bánh răng) → **Script properties** → **Add script property**, tạo:
   - `ROOT_FOLDER_ID` = ID thư mục Drive nhóm chứa dữ liệu, hoặc thư mục cha của folder dự án. Với link dạng `https://drive.google.com/drive/folders/13i4goNfzrNO10J7jn72gSRkLQzn730Cp`, ID chính là chuỗi sau `/folders/`.
   - `BRIDGE_TOKEN` = chuỗi bí mật ngẫu nhiên dài ít nhất 32 ký tự. Tạo bằng Python trên máy cá nhân: `python -c "import secrets; print(secrets.token_urlsafe(36))"`. Giữ chuỗi này để dùng ở GitHub.
6. Bấm Save, chọn hàm `authorizeOnce` rồi **Run**. Chấp thuận quyền Drive/Sheets cho chính tài khoản của bạn. Hàm sẽ tìm folder chứa hai RAW CSV, đồng thời kiểm tra quyền Sheets.
7. Chọn **Deploy → New deployment → Web app**.
   - Execute as: **Me**
   - Who has access: **Anyone** (để runner GitHub gọi được mà không đăng nhập Google; endpoint vẫn kiểm tra token bí mật).
   - Deploy và sao chép URL kết thúc bằng `/exec`.

Nếu chính sách trường/cơ quan không cho phép Web App truy cập `Anyone`, phương án này không dùng được với tài khoản đó. Không đặt endpoint/token trong nội dung công khai hoặc trong repository.

## 2. Đưa bộ mã lên GitHub

1. Tạo repository **Private**, ví dụ `hanoi-pm25-automation`.
2. Chép toàn bộ nội dung gói này lên repository, giữ nguyên `.github/workflows/update-dashboard.yml`.
3. Không đưa RAW CSV, token hoặc dữ liệu cá nhân lên GitHub.

## 3. Thêm GitHub Secrets

Vào repository → **Settings → Secrets and variables → Actions → New repository secret**. Tạo ba secrets:

| Secret | Giá trị |
|---|---|
| `OPENAQ_API_KEY` | OpenAQ API key của nhóm |
| `GOOGLE_APPS_SCRIPT_URL` | URL Web App kết thúc bằng `/exec` |
| `GOOGLE_DRIVE_BRIDGE_TOKEN` | Đúng giá trị `BRIDGE_TOKEN` đã thêm trong Script properties |

Không cần `GOOGLE_SERVICE_ACCOUNT_JSON`, không cần repository variable `GOOGLE_DRIVE_FOLDER_ID`, và không cần cấu hình Service Account.

## 4. Chạy thử trước khi bật lịch

1. Commit code lên **default branch**.
2. Mở tab **Actions** → workflow **Hanoi PM2.5 Dashboard Update** → **Run workflow**.
3. Theo dõi log. Lần đầu có thể lâu hơn vì phải tải RAW và khôi phục/training model.
4. Sau khi pipeline chạy xong, script gửi RAW/state/cache/model/CSV về Drive và cập nhật các Google Sheets:
   - `latest_forecast_current`
   - `latest_forecast`
   - `observed_hourly`
   - `daily_summary`
   - `model_metrics`
   - `station_freshness`
   - `data_coverage`
   - `run_status`
5. Trong Looker Studio, kết nối từng Google Sheets đúng tên bằng **Create → Data source → Google Sheets**. Nếu đã có data source cũ, cần chọn lại spreadsheet này.

## 5. Lịch tự chạy

Workflow đặt lịch `17 */6 * * *`, dự kiến lúc 01:17, 07:17, 13:17 và 19:17 giờ Việt Nam. GitHub dùng UTC và cron có thể chạy trễ khi tải cao. Hãy giữ chu kỳ 6 giờ cho thử nghiệm; sau khi đo thời gian chạy thực tế, điều chỉnh lịch để không vượt hạn mức Actions miễn phí.

## Giới hạn / lưu ý

- Đây là cập nhật gần thời gian thực theo lịch, không phải stream từng giây. OpenAQ cũng có độ trễ riêng.
- GitHub Actions chạy script Python/PySpark; nó **không điều khiển phiên Colab đang mở**.
- Apps Script có giới hạn runtime và quota, nên bảng Google Sheets quá lớn có thể mất nhiều thời gian hoặc bị giới hạn. Mỗi spreadsheet có giới hạn tổng ô; script bỏ qua bảng vượt ngưỡng an toàn.
- Gói này đã chuyển phần gọi Drive/Sheets khỏi Service Account sang Apps Script bridge và kiểm tra cú pháp Python. Chưa thể xác minh lần chạy thực tế với tài khoản/Drive/OpenAQ của nhóm; phải chạy thử thủ công một lần và xem log.
