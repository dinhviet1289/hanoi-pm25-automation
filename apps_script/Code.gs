/**
 * Drive bridge cho GitHub Actions (hanoi-pm25-automation)
 *
 * Cách dùng:
 * 1. Sửa ROOT_FOLDER_ID bên dưới = ID thư mục Drive chứa 2 file RAW.
 * 2. Project Settings -> Script properties -> thêm BRIDGE_TOKEN
 *    (giá trị giống GitHub Secret GOOGLE_DRIVE_BRIDGE_TOKEN).
 * 3. Chạy hàm testSetup() 1 lần để cấp quyền (Drive, Sheets, External request).
 * 4. Deploy -> Manage deployments -> (bút chì) -> Version: New version -> Deploy.
 *    Execute as: Me | Who has access: Anyone. URL /exec giữ nguyên.
 */
 
const ROOT_FOLDER_ID = 'DAN_ID_THU_MUC_GOC_O_DAY';
const DEFAULT_CHUNK_BYTES = 2 * 1024 * 1024;
const MAX_CHUNK_BYTES = 8 * 1024 * 1024;
const UPLOAD_PREFIX = 'up_';
const DRIVE_FILES_API = 'https://www.googleapis.com/drive/v3/files/';
const DRIVE_UPLOAD_API = 'https://www.googleapis.com/upload/drive/v3/files';
 
// ---------------------------------------------------------------- entry
function doGet() {
  return json_({ ok: true, message: 'Drive bridge đang chạy' });
}
 
function doPost(e) {
  try {
    const req = JSON.parse(e.postData.contents);
    const token = PropertiesService.getScriptProperties().getProperty('BRIDGE_TOKEN');
    if (!token || req.token !== token) throw new Error('Sai hoặc thiếu token');
 
    switch (req.action) {
      case 'list_files':       return json_(listFiles_());
      case 'download_chunk':   return json_(downloadChunk_(req));
      case 'upload_chunk':     return json_(uploadChunk_(req));
      case 'finish_upload':    return json_(finishUpload_(req));
      case 'begin_sheet':      return json_(beginSheet_(req));
      case 'write_sheet_rows': return json_(writeSheetRows_(req));
      case 'finish_sheet':     return json_(finishSheet_(req));
      default: throw new Error('Action không hỗ trợ: ' + req.action);
    }
  } catch (err) {
    return json_({ ok: false, error: String((err && err.message) || err) });
  }
}
 
// ---------------------------------------------------------------- helpers
function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj))
    .setMimeType(ContentService.MimeType.JSON);
}
 
function authHeader_() {
  return { Authorization: 'Bearer ' + ScriptApp.getOAuthToken() };
}
 
function rootFolder_() {
  return DriveApp.getFolderById(ROOT_FOLDER_ID);
}
 
function splitPath_(path) {
  const parts = String(path || '').split('/').filter(function (p) { return p; });
  if (!parts.length || parts.indexOf('..') >= 0) throw new Error('Đường dẫn không hợp lệ: ' + path);
  return parts;
}
 
function newest_(iter) {
  let best = null;
  while (iter.hasNext()) {
    const f = iter.next();
    if (!best || f.getLastUpdated() > best.getLastUpdated()) best = f;
  }
  return best;
}
 
// Tìm file theo đường dẫn tương đối từ ROOT; trả về null nếu không có.
function resolveFile_(relPath) {
  const parts = splitPath_(relPath);
  let folder = rootFolder_();
  for (let i = 0; i < parts.length - 1; i++) {
    const it = folder.getFoldersByName(parts[i]);
    if (!it.hasNext()) return null;
    folder = it.next();
  }
  return newest_(folder.getFilesByName(parts[parts.length - 1]));
}
 
// Lấy thư mục theo mảng tên, tự tạo nếu chưa có.
function ensureFolder_(parts) {
  let folder = rootFolder_();
  parts.forEach(function (name) {
    const it = folder.getFoldersByName(name);
    folder = it.hasNext() ? it.next() : folder.createFolder(name);
  });
  return folder;
}
 
// ---------------------------------------------------------------- list
function listFiles_() {
  const byPath = {};
  walk_(rootFolder_(), '', byPath);
  const files = Object.keys(byPath).map(function (k) {
    const x = byPath[k];
    return { path: k, id: x.id, size: x.size, mimeType: x.mimeType };
  });
  return { ok: true, files: files };
}
 
function walk_(folder, prefix, out) {
  const fi = folder.getFiles();
  while (fi.hasNext()) {
    const f = fi.next();
    const mime = f.getMimeType();
    // Bỏ Google Sheets/Docs/Shortcut: không tải được dạng nhị phân
    if (mime.indexOf('application/vnd.google-apps.') === 0) continue;
    const path = prefix + f.getName();
    const updated = f.getLastUpdated().getTime();
    // Trùng tên trong cùng thư mục -> giữ bản mới nhất
    if (!out[path] || updated > out[path].updated) {
      out[path] = { id: f.getId(), size: f.getSize(), mimeType: mime, updated: updated };
    }
  }
  const di = folder.getFolders();
  while (di.hasNext()) {
    const d = di.next();
    walk_(d, prefix + d.getName() + '/', out);
  }
}
 
// ---------------------------------------------------------------- download
// Đọc đúng 1 khoảng byte bằng Drive API (Range) -> không phải nạp cả file vào RAM.
// LUÔN trả về key "base64" (chuỗi rỗng nếu hết file).
function downloadChunk_(req) {
  const file = resolveFile_(req.path);
  if (!file) throw new Error('Không tìm thấy file: ' + req.path);
 
  const size = file.getSize();
  const offset = Math.max(0, Number(req.offset) || 0);
  const chunkSize = Math.min(Number(req.chunkSize) || DEFAULT_CHUNK_BYTES, MAX_CHUNK_BYTES);
 
  if (offset >= size) return { ok: true, base64: '', offset: offset, length: 0, size: size };
 
  const end = Math.min(offset + chunkSize, size) - 1;
  const headers = authHeader_();
  headers['Range'] = 'bytes=' + offset + '-' + end;
 
  const resp = UrlFetchApp.fetch(
    DRIVE_FILES_API + file.getId() + '?alt=media&supportsAllDrives=true',
    { headers: headers, muteHttpExceptions: true }
  );
  const code = resp.getResponseCode();
  let bytes;
  if (code === 206) {
    bytes = resp.getContent();
  } else if (code === 200) {
    bytes = resp.getContent().slice(offset, end + 1); // server bỏ qua Range
  } else {
    throw new Error('Drive API ' + code + ' khi tải ' + req.path + ': ' + resp.getContentText().slice(0, 300));
  }
  return {
    ok: true,
    base64: Utilities.base64Encode(bytes),
    offset: offset,
    length: bytes.length,
    size: size
  };
}
 
// ---------------------------------------------------------------- upload
// Dùng resumable upload của Drive API: mỗi chunk đẩy thẳng lên Drive,
// không cần ghép file trong bộ nhớ Apps Script.
function uploadChunk_(req) {
  const props = PropertiesService.getScriptProperties();
  const key = UPLOAD_PREFIX + req.uploadId;
  const idx = Number(req.chunkIndex);
  const total = Number(req.totalChunks);
  const bytes = Utilities.base64Decode(req.base64 || '');
  const parts = splitPath_(req.path);
  const name = parts[parts.length - 1];
 
  // File rỗng (0 byte)
  if (total === 1 && bytes.length === 0) {
    const existing = resolveFile_(req.path);
    if (existing) existing.setContent('');
    else ensureFolder_(parts.slice(0, -1)).createFile(name, '');
    return { ok: true, empty: true };
  }
 
  let st = JSON.parse(props.getProperty(key) || 'null');
  if (!st && idx === 0) {
    st = { path: req.path, uri: null, nextIndex: 0, offset: 0, done: false, ts: Date.now() };
  }
  if (!st) throw new Error('Không có phiên upload ' + req.uploadId + ' (chunk ' + idx + ')');
  if (idx < st.nextIndex) return { ok: true, duplicate: true }; // client retry
  if (idx !== st.nextIndex) throw new Error('Sai thứ tự chunk: cần ' + st.nextIndex + ', nhận ' + idx);
 
  // Mở phiên resumable ở chunk đầu tiên
  if (!st.uri) {
    const existing = resolveFile_(req.path);
    let url, method, meta;
    if (existing) {
      url = DRIVE_UPLOAD_API + '/' + existing.getId() + '?uploadType=resumable&supportsAllDrives=true';
      method = 'patch';
      meta = { name: name };
    } else {
      const folder = ensureFolder_(parts.slice(0, -1));
      url = DRIVE_UPLOAD_API + '?uploadType=resumable&supportsAllDrives=true';
      method = 'post';
      meta = { name: name, parents: [folder.getId()] };
    }
    const init = UrlFetchApp.fetch(url, {
      method: method,
      contentType: 'application/json; charset=UTF-8',
      headers: authHeader_(),
      payload: JSON.stringify(meta),
      muteHttpExceptions: true
    });
    if (init.getResponseCode() !== 200) {
      throw new Error('Không mở được phiên upload (' + init.getResponseCode() + '): ' + init.getContentText().slice(0, 300));
    }
    const hdrs = init.getAllHeaders();
    st.uri = hdrs['Location'] || hdrs['location'];
    if (!st.uri) throw new Error('Drive không trả Location cho phiên upload');
  }
 
  const isLast = idx === total - 1;
  const start = st.offset;
  const end = start + bytes.length - 1;
  const putHeaders = authHeader_();
  putHeaders['Content-Range'] = 'bytes ' + start + '-' + end + '/' + (isLast ? (end + 1) : '*');
 
  const resp = UrlFetchApp.fetch(st.uri, {
    method: 'put',
    contentType: 'application/octet-stream',
    headers: putHeaders,
    payload: bytes,
    muteHttpExceptions: true
  });
  const code = resp.getResponseCode();
  const ok = isLast ? (code === 200 || code === 201) : (code === 308);
  if (!ok) throw new Error('Upload chunk ' + idx + ' lỗi (' + code + '): ' + resp.getContentText().slice(0, 300));
 
  st.offset = end + 1;
  st.nextIndex = idx + 1;
  st.done = isLast;
  props.setProperty(key, JSON.stringify(st));
  return { ok: true, received: bytes.length, last: isLast };
}
 
function finishUpload_(req) {
  const props = PropertiesService.getScriptProperties();
  const key = UPLOAD_PREFIX + req.uploadId;
  const raw = props.getProperty(key);
  if (raw) {
    const st = JSON.parse(raw);
    if (!st.done || st.nextIndex !== Number(req.totalChunks)) {
      throw new Error('Upload ' + req.path + ' chưa đủ chunk: ' + st.nextIndex + '/' + req.totalChunks);
    }
    props.deleteProperty(key);
  }
  cleanupOldUploads_();
  return { ok: true };
}
 
// Xoá trạng thái upload dở dang quá 24h
function cleanupOldUploads_() {
  const props = PropertiesService.getScriptProperties();
  const all = props.getProperties();
  const limit = Date.now() - 24 * 3600 * 1000;
  Object.keys(all).forEach(function (k) {
    if (k.indexOf(UPLOAD_PREFIX) !== 0) return;
    try {
      if (JSON.parse(all[k]).ts < limit) props.deleteProperty(k);
    } catch (e) {
      props.deleteProperty(k);
    }
  });
}
 
// ---------------------------------------------------------------- Google Sheets
function beginSheet_(req) {
  const root = rootFolder_();
  let ss = null;
  const it = root.getFilesByName(req.title);
  while (it.hasNext()) {
    const f = it.next();
    if (f.getMimeType() === MimeType.GOOGLE_SHEETS) {
      ss = SpreadsheetApp.openById(f.getId());
      break;
    }
  }
  if (!ss) {
    ss = SpreadsheetApp.create(req.title);
    DriveApp.getFileById(ss.getId()).moveTo(root);
  }
 
  const sheet = ss.getSheets()[0];
  const rows = Math.max(Number(req.totalRows) || 2, 2);
  const cols = Math.max(Number(req.totalCols) || 1, 1);
 
  sheet.setFrozenRows(0);
  sheet.clear();
  resizeSheet_(sheet, rows, cols);
  sheet.getRange(1, 1, 1, cols).setValues([req.headers]);
  sheet.setFrozenRows(1);
  return { ok: true, spreadsheetId: ss.getId(), url: ss.getUrl() };
}
 
function resizeSheet_(sheet, rows, cols) {
  const mr = sheet.getMaxRows();
  const mc = sheet.getMaxColumns();
  if (mr < rows) sheet.insertRowsAfter(mr, rows - mr);
  else if (mr > rows) sheet.deleteRows(rows + 1, mr - rows);
  if (mc < cols) sheet.insertColumnsAfter(mc, cols - mc);
  else if (mc > cols) sheet.deleteColumns(cols + 1, mc - cols);
}
 
function writeSheetRows_(req) {
  const rows = req.rows || [];
  if (!rows.length) return { ok: true, written: 0 };
  const sheet = SpreadsheetApp.openById(req.spreadsheetId).getSheets()[0];
  const cols = req.headers.length;
  const values = rows.map(function (r) {
    return r.map(function (v) { return (v === null || v === undefined) ? '' : v; });
  });
  sheet.getRange(Number(req.startRow), 1, values.length, cols).setValues(values);
  return { ok: true, written: values.length };
}
 
function finishSheet_(req) {
  SpreadsheetApp.flush();
  const ss = SpreadsheetApp.openById(req.spreadsheetId);
  return { ok: true, url: ss.getUrl() };
}
 
// ---------------------------------------------------------------- test thủ công
// Chạy hàm này trong editor để cấp quyền và kiểm tra nhanh.
function testSetup() {
  const r = listFiles_();
  Logger.log('Root: ' + rootFolder_().getName() + ' | ' + r.files.length + ' file');
  r.files.slice(0, 30).forEach(function (f) {
    Logger.log(f.path + ' | ' + f.size + ' B | ' + f.mimeType);
  });
  const first = r.files.filter(function (f) { return f.size > 0; })[0];
  if (first) {
    const c = downloadChunk_({ path: first.path, offset: 0, chunkSize: 1024 });
    Logger.log('Range OK: ' + first.path + ' -> ' + c.length + ' bytes, có key base64 = ' + ('base64' in c));
  }
}
 
