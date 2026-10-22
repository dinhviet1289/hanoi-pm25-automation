/**
 * Google Drive bridge for GitHub Actions.
 * Uses the script owner's Google Drive/Sheets permissions. No manual Cloud Console
 * project setup is needed for a normal My Drive folder shared with this account.
 * Configure ROOT_FOLDER_ID and BRIDGE_TOKEN under Project Settings > Script properties.
 */
function doGet() {
  return json_({ ok: true, service: "hanoi-pm25-drive-bridge", message: "POST actions only" });
}

function doPost(e) {
  try {
    if (!e || !e.postData || !e.postData.contents) throw new Error("Missing JSON POST body");
    const p = JSON.parse(e.postData.contents);
    const expected = PropertiesService.getScriptProperties().getProperty("BRIDGE_TOKEN") || "";
    if (!expected || !p.token || !constantTimeEqual_(String(p.token), expected)) {
      throw new Error("Unauthorized: bridge token mismatch");
    }
    const action = String(p.action || "");
    let result;
    switch (action) {
      case "list_files": result = { ok: true, files: listProjectFiles_() }; break;
      case "download_chunk": result = downloadChunk_(p); break;
      case "upload_chunk": result = uploadChunk_(p); break;
      case "finish_upload": result = finishUpload_(p); break;
      case "begin_sheet": result = beginSheet_(p); break;
      case "write_sheet_rows": result = writeSheetRows_(p); break;
      case "finish_sheet": result = finishSheet_(p); break;
      default: throw new Error("Unknown action: " + action);
    }
    return json_(result);
  } catch (err) {
    return json_({ ok: false, error: String(err && err.message ? err.message : err) });
  }
}

function authorizeOnce() {
  const folder = getDataFolder_();
  const temp = SpreadsheetApp.create("HANOI_PM25_AUTH_CHECK_TEMP");
  DriveApp.getFileById(temp.getId()).setTrashed(true);
  Logger.log("Authorization is ready. Resolved folder: " + folder.getName());
}

function json_(obj) {
  return ContentService.createTextOutput(JSON.stringify(obj)).setMimeType(ContentService.MimeType.JSON);
}

function constantTimeEqual_(a, b) {
  if (a.length !== b.length) return false;
  let x = 0;
  for (let i = 0; i < a.length; i++) x |= a.charCodeAt(i) ^ b.charCodeAt(i);
  return x === 0;
}

function getRootFolder_() {
  const id = PropertiesService.getScriptProperties().getProperty("ROOT_FOLDER_ID");
  if (!id) throw new Error("Set Script Property ROOT_FOLDER_ID first");
  return DriveApp.getFolderById(id);
}

function folderHasRawPair_(folder) {
  const names = {};
  const it = folder.getFiles();
  while (it.hasNext()) names[it.next().getName()] = true;
  const a = names["openaq_hanoi_2019_2026_raw.csv"] || names["openaq_hanoi_2019_2026_raw (1).csv"] || names["openaq_hanoi_2019_2026_raw(1).csv"];
  const b = names["open_meteo_hanoi_2019_2026_raw.csv"] || names["open_meteo_hanoi_2019_2026_raw (1).csv"] || names["open_meteo_hanoi_2019_2026_raw(1).csv"];
  return !!(a && b);
}

function findDataFolderRecursive_(folder, depth) {
  if (folderHasRawPair_(folder)) return folder;
  if (depth >= 6) return null;
  const folders = [];
  const it = folder.getFolders();
  while (it.hasNext()) folders.push(it.next());
  folders.sort(function(a, b) {
    const an = a.getName().toLowerCase(), bn = b.getName().toLowerCase();
    const ap = (an.indexOf("hanoi") >= 0 || an.indexOf("pm25") >= 0) ? 0 : 1;
    const bp = (bn.indexOf("hanoi") >= 0 || bn.indexOf("pm25") >= 0) ? 0 : 1;
    return ap - bp || an.localeCompare(bn);
  });
  for (let i = 0; i < folders.length; i++) {
    const found = findDataFolderRecursive_(folders[i], depth + 1);
    if (found) return found;
  }
  return null;
}

function getDataFolder_() {
  const props = PropertiesService.getScriptProperties();
  const cachedId = props.getProperty("RESOLVED_DATA_FOLDER_ID");
  if (cachedId) {
    try {
      const cached = DriveApp.getFolderById(cachedId);
      if (folderHasRawPair_(cached)) return cached;
    } catch (ignored) {}
  }
  const found = findDataFolderRecursive_(getRootFolder_(), 0);
  if (!found) {
    throw new Error("Cannot find folder containing both required RAW files. Set ROOT_FOLDER_ID to the folder or its parent.");
  }
  props.setProperty("RESOLVED_DATA_FOLDER_ID", found.getId());
  return found;
}

function allowedFile_(name) {
  const lower = name.toLowerCase();
  const skip = [".zip", ".pdf", ".docx", ".xlsx", ".ipynb", ".png", ".jpg", ".jpeg", ".svg", ".gif"];
  for (let i = 0; i < skip.length; i++) if (lower.endsWith(skip[i])) return false;
  return true;
}

function listProjectFiles_() {
  const root = getDataFolder_();
  const output = [];
  const rootFiles = root.getFiles();
  while (rootFiles.hasNext()) {
    const file = rootFiles.next();
    const name = file.getName();
    if (/^(openaq_hanoi_2019_2026_raw( \(1\)|\(1\))?\.csv|open_meteo_hanoi_2019_2026_raw( \(1\)|\(1\))?\.csv|sensor_poll_state\.csv)$/i.test(name) && allowedFile_(name)) {
      output.push({ path: name, size: file.getSize(), mimeType: file.getMimeType() });
    }
  }
  const dirs = root.getFolders();
  while (dirs.hasNext()) {
    const dir = dirs.next();
    if (dir.getName() === "cache_openaq" || dir.getName() === "output_hanoi_pm25") {
      listTree_(dir, dir.getName(), output, 0);
    }
  }
  return output;
}

function listTree_(folder, prefix, output, depth) {
  if (depth > 8) return;
  const files = folder.getFiles();
  while (files.hasNext()) {
    const file = files.next();
    if (allowedFile_(file.getName())) {
      output.push({ path: prefix + "/" + file.getName(), size: file.getSize(), mimeType: file.getMimeType() });
    }
  }
  const dirs = folder.getFolders();
  while (dirs.hasNext()) {
    const dir = dirs.next();
    if (dir.getName() !== ".github_actions_upload_staging") listTree_(dir, prefix + "/" + dir.getName(), output, depth + 1);
  }
}

function resolvePath_(relativePath, createFolders) {
  const parts = String(relativePath || "").split("/").filter(Boolean);
  if (!parts.length || parts.some(function(x) { return x === "." || x === ".."; })) throw new Error("Invalid path");
  let folder = getDataFolder_();
  for (let i = 0; i < parts.length - 1; i++) {
    let found = null;
    const it = folder.getFoldersByName(parts[i]);
    if (it.hasNext()) found = it.next();
    if (!found && createFolders) found = folder.createFolder(parts[i]);
    if (!found) throw new Error("Folder path not found: " + parts.slice(0, i + 1).join("/"));
    folder = found;
  }
  const fileName = parts[parts.length - 1];
  const files = folder.getFilesByName(fileName);
  return { folder: folder, fileName: fileName, file: files.hasNext() ? files.next() : null };
}

function downloadChunk_(p) {
  const resolved = resolvePath_(p.path, false);
  if (!resolved.file) throw new Error("File not found: " + p.path);
  const bytes = resolved.file.getBlob().getBytes();
  const offset = Math.max(0, Number(p.offset || 0));
  const size = Math.max(1, Math.min(2 * 1024 * 1024, Number(p.chunkSize || 2 * 1024 * 1024)));
  const chunk = bytes.slice(offset, Math.min(bytes.length, offset + size));
  return { ok: true, totalSize: bytes.length, offset: offset, base64: Utilities.base64Encode(chunk) };
}

function uploadChunk_(p) {
  const uploadId = String(p.uploadId || "");
  const chunkIndex = Number(p.chunkIndex);
  const encoded = String(p.base64 || "");
  if (!/^[a-f0-9]{20,64}$/i.test(uploadId)) throw new Error("Invalid uploadId");
  if (!Number.isInteger(chunkIndex) || chunkIndex < 0) throw new Error("Invalid chunkIndex");
  const root = getDataFolder_();
  let staging = null;
  const folders = root.getFoldersByName(".github_actions_upload_staging");
  staging = folders.hasNext() ? folders.next() : root.createFolder(".github_actions_upload_staging");
  const partName = uploadId + "__" + chunkIndex + ".txt";
  const existing = staging.getFilesByName(partName);
  while (existing.hasNext()) existing.next().setTrashed(true);
  staging.createFile(partName, encoded, MimeType.PLAIN_TEXT);
  return { ok: true, chunkIndex: chunkIndex };
}

function finishUpload_(p) {
  const uploadId = String(p.uploadId || "");
  const totalChunks = Number(p.totalChunks);
  const relativePath = String(p.path || "");
  if (!/^[a-f0-9]{20,64}$/i.test(uploadId)) throw new Error("Invalid uploadId");
  if (!Number.isInteger(totalChunks) || totalChunks < 1) throw new Error("Invalid totalChunks");
  const root = getDataFolder_();
  const stagingIt = root.getFoldersByName(".github_actions_upload_staging");
  if (!stagingIt.hasNext()) throw new Error("Upload staging folder missing");
  const staging = stagingIt.next();
  const bytes = [];
  const partsToTrash = [];
  for (let i = 0; i < totalChunks; i++) {
    const it = staging.getFilesByName(uploadId + "__" + i + ".txt");
    if (!it.hasNext()) throw new Error("Missing upload chunk " + i + " of " + totalChunks);
    const part = it.next();
    partsToTrash.push(part);
    const bin = Utilities.base64Decode(part.getBlob().getDataAsString());
    for (let j = 0; j < bin.length; j++) bytes.push(bin[j]);
  }
  const resolved = resolvePath_(relativePath, true);
  if (resolved.file) resolved.file.setTrashed(true);
  const blob = Utilities.newBlob(bytes, "application/octet-stream", resolved.fileName);
  resolved.folder.createFile(blob);
  partsToTrash.forEach(function(f) { f.setTrashed(true); });
  return { ok: true, path: relativePath, size: bytes.length };
}

function findSheetFile_(folder, title) {
  const files = folder.getFilesByName(title);
  while (files.hasNext()) {
    const file = files.next();
    if (file.getMimeType() === MimeType.GOOGLE_SHEETS) return file;
  }
  return null;
}

function beginSheet_(p) {
  const title = String(p.title || "").trim();
  const headers = p.headers;
  const totalRows = Number(p.totalRows || 1);
  const totalCols = Number(p.totalCols || (headers || []).length);
  if (!title || !Array.isArray(headers) || !headers.length) throw new Error("Invalid sheet title/headers");
  if (totalRows * Math.max(totalCols, 1) > 9500000) throw new Error("Sheet would exceed the 9.5M-cell safety threshold");
  const folder = getDataFolder_();
  let file = findSheetFile_(folder, title);
  let book;
  if (file) {
    book = SpreadsheetApp.openById(file.getId());
  } else {
    book = SpreadsheetApp.create(title);
    DriveApp.getFileById(book.getId()).moveTo(folder);
  }
  let sheet = book.getSheets()[0];
  if (!sheet) sheet = book.insertSheet("Sheet1");
  sheet.clear();
  if (sheet.getMaxRows() < totalRows) sheet.insertRowsAfter(sheet.getMaxRows(), totalRows - sheet.getMaxRows());
  if (sheet.getMaxColumns() < totalCols) sheet.insertColumnsAfter(sheet.getMaxColumns(), totalCols - sheet.getMaxColumns());
  sheet.getRange(1, 1, 1, headers.length).setValues([headers]);
  sheet.setFrozenRows(1);
  book.setSpreadsheetTimeZone("Asia/Ho_Chi_Minh");
  return { ok: true, spreadsheetId: book.getId(), url: book.getUrl(), rows: totalRows, cols: totalCols };
}

function writeSheetRows_(p) {
  const id = String(p.spreadsheetId || "");
  const headers = p.headers;
  const rows = p.rows;
  const startRow = Number(p.startRow);
  if (!id || !Array.isArray(headers) || !Array.isArray(rows) || !Number.isInteger(startRow) || startRow < 2) throw new Error("Invalid sheet row payload");
  if (rows.length === 0) return { ok: true, written: 0 };
  const book = SpreadsheetApp.openById(id);
  const sheet = book.getSheets()[0];
  const neededRows = startRow + rows.length - 1;
  if (sheet.getMaxRows() < neededRows) sheet.insertRowsAfter(sheet.getMaxRows(), neededRows - sheet.getMaxRows());
  if (sheet.getMaxColumns() < headers.length) sheet.insertColumnsAfter(sheet.getMaxColumns(), headers.length - sheet.getMaxColumns());
  const normalized = rows.map(function(row) {
    return headers.map(function(header, i) { return sheetValue_(header, row[i]); });
  });
  sheet.getRange(startRow, 1, normalized.length, headers.length).setValues(normalized);
  return { ok: true, written: normalized.length, startRow: startRow };
}

function sheetValue_(header, value) {
  if (value === null || value === undefined || value === "") return "";
  if (typeof value !== "string") return value;
  if (/(^date$|datetime|timestamp|_time|time_local|issue_time|target_time)/i.test(String(header))) {
    const candidate = value.trim();
    if (/^\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d+)?)?(?:Z|[+-]\d{2}:?\d{2})?)?$/.test(candidate)) {
      const d = new Date(candidate.replace(" ", "T"));
      if (!isNaN(d.getTime())) return d;
    }
  }
  return value;
}

function finishSheet_(p) {
  const book = SpreadsheetApp.openById(String(p.spreadsheetId || ""));
  SpreadsheetApp.flush();
  return { ok: true, spreadsheetId: book.getId(), url: book.getUrl() };
}
