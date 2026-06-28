"""
細胞計數網頁應用
演算法：背景相減法 + 亮度/面積雙條件過濾 + 亮度 NMS 相鄰細胞分裂
"""

import os
import base64

import cv2
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)


def count_cells(img_bytes: bytes, contrast_threshold=12, min_area=65,
                max_area=3000, bg_blur=101, split_adjacent=True):
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ksize = bg_blur if bg_blur % 2 == 1 else bg_blur + 1
    bg = cv2.GaussianBlur(gray.astype(np.float32), (ksize, ksize), 0)
    local_contrast = np.clip(gray.astype(np.float32) - bg, 0, 255).astype(np.uint8)

    _, binary = cv2.threshold(local_contrast, contrast_threshold, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)

    min_peak = 95
    split_threshold = max(2.5 * min_area, 200.0)
    lc_smooth = cv2.GaussianBlur(local_contrast.astype(np.float32), (5, 5), 1.5)

    def nms_in_blob(blob_mask):
        candidates = np.argwhere(blob_mask & (lc_smooth > 50))
        if len(candidates) == 0:
            return []
        brightnesses = lc_smooth[candidates[:, 0], candidates[:, 1]]
        order = np.argsort(-brightnesses)
        candidates = candidates[order]
        suppressed = np.zeros(lc_smooth.shape, dtype=bool)
        peaks = []
        for r, c in candidates:
            if suppressed[r, c]:
                continue
            peaks.append((float(c), float(r)))
            suppressed[max(0, r-7):r+8, max(0, c-7):c+8] = True
        return peaks

    valid_cells = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (20 <= area <= max_area):
            continue
        blob_mask = (labels == i)
        peak = int(local_contrast[blob_mask].max())
        if not (peak >= min_peak or area >= min_area):
            continue

        centroid = (float(centroids[i][0]), float(centroids[i][1]))

        if split_adjacent and area > split_threshold:
            nms_peaks = nms_in_blob(blob_mask)
            if len(nms_peaks) >= 2:
                for pos in nms_peaks:
                    valid_cells.append((pos, area // max(1, len(nms_peaks))))
                continue
        valid_cells.append((centroid, area))

    count = len(valid_cells)
    annotated = img.copy()
    for (cx, cy), area in valid_cells:
        r = max(8, int(np.sqrt(area / np.pi)))
        cv2.circle(annotated, (int(cx), int(cy)), r, (0, 255, 0), 2)
    cv2.putText(annotated, f"Count: {count}", (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

    _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    preview_b64 = base64.b64encode(buf.tobytes()).decode()
    return count, preview_b64


HTML = """<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>細胞計數工具</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
         background: #0f1117; color: #e0e0e0; min-height: 100vh; padding: 32px 24px; }
  h1 { font-size: 1.6rem; font-weight: 600; margin-bottom: 6px; color: #fff; }
  .subtitle { color: #888; font-size: 0.9rem; margin-bottom: 28px; }

  .drop-zone {
    border: 2px dashed #334;
    border-radius: 12px;
    padding: 48px 24px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    background: #161b22;
    margin-bottom: 24px;
  }
  .drop-zone:hover, .drop-zone.over { border-color: #58a6ff; background: #1c2230; }
  .drop-zone input { display: none; }
  .drop-zone .icon { font-size: 2.5rem; margin-bottom: 12px; }
  .drop-zone p { color: #8b949e; font-size: 0.95rem; }
  .drop-zone strong { color: #58a6ff; }

  .params { display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 16px; align-items: flex-end; }
  .param-group { display: flex; flex-direction: column; gap: 4px; }
  .param-group label { font-size: 0.8rem; color: #8b949e; }
  .param-group input[type="number"] {
    background: #161b22; border: 1px solid #334; border-radius: 6px;
    color: #e0e0e0; padding: 6px 10px; width: 100px; font-size: 0.9rem;
  }
  .toggle-group { display: flex; align-items: center; gap: 8px; padding-bottom: 2px; }
  .toggle-group label { font-size: 0.85rem; color: #8b949e; cursor: pointer; }
  .toggle-group input[type="checkbox"] { width: 16px; height: 16px; cursor: pointer; accent-color: #238636; }

  .btn {
    background: #238636; color: #fff; border: none; border-radius: 8px;
    padding: 10px 28px; font-size: 0.95rem; cursor: pointer; transition: background 0.2s;
  }
  .btn:hover { background: #2ea043; }
  .btn:disabled { background: #333; color: #666; cursor: not-allowed; }
  .btn-outline {
    background: transparent; border: 1px solid #334; color: #8b949e;
    border-radius: 8px; padding: 10px 20px; font-size: 0.95rem; cursor: pointer;
    transition: border-color 0.2s, color 0.2s;
  }
  .btn-outline:hover { border-color: #58a6ff; color: #58a6ff; }

  .actions { display: flex; gap: 12px; align-items: center; margin-bottom: 28px; flex-wrap: wrap; }

  #progress-bar-wrap {
    display: none; background: #161b22; border-radius: 8px;
    height: 6px; margin-bottom: 20px; overflow: hidden;
  }
  #progress-bar { height: 100%; background: #238636; width: 0%; transition: width 0.3s; }
  #progress-text { font-size: 0.85rem; color: #8b949e; margin-bottom: 12px; display: none; }

  .summary {
    display: none; background: #161b22; border: 1px solid #334; border-radius: 10px;
    padding: 20px 24px; margin-bottom: 24px;
  }
  .summary h2 { font-size: 1rem; color: #8b949e; margin-bottom: 8px; }
  .summary .total { font-size: 2.2rem; font-weight: 700; color: #58a6ff; }
  .summary .meta { font-size: 0.85rem; color: #666; margin-top: 4px; }

  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  th { text-align: left; padding: 10px 14px; color: #8b949e; font-weight: 500;
       border-bottom: 1px solid #21262d; }
  td { padding: 12px 14px; border-bottom: 1px solid #161b22; vertical-align: middle; }
  tr:last-child td { border-bottom: none; }
  tr:hover td { background: #161b22; }
  .count-badge {
    display: inline-block; background: #1f3a1f; color: #3fb950;
    border-radius: 6px; padding: 2px 10px; font-weight: 600;
  }
  .thumb { width: 80px; height: 56px; object-fit: cover; border-radius: 4px;
           cursor: pointer; transition: opacity 0.2s; }
  .thumb:hover { opacity: 0.8; }
  #result-section { display: none; }

  #lightbox {
    display: none; position: fixed; inset: 0; background: rgba(0,0,0,0.85);
    z-index: 999; align-items: center; justify-content: center;
  }
  #lightbox.show { display: flex; }
  #lightbox img { max-width: 90vw; max-height: 90vh; border-radius: 8px; }
  #lightbox-close {
    position: absolute; top: 20px; right: 28px; font-size: 2rem;
    color: #fff; cursor: pointer; line-height: 1;
  }
</style>
</head>
<body>

<h1>細胞計數工具</h1>
<p class="subtitle">選擇包含顯微鏡影像的資料夾，自動計算每張圖的細胞數</p>

<div class="drop-zone" id="drop-zone" onclick="document.getElementById('folder-input').click()">
  <input type="file" id="folder-input" webkitdirectory multiple>
  <div class="icon">📂</div>
  <p><strong>點此選擇資料夾</strong>，或將資料夾拖放至此</p>
  <p style="margin-top:8px; font-size:0.8rem;">支援 JPG、PNG、BMP、TIF</p>
</div>

<div class="params">
  <div class="param-group">
    <label>對比閾值（越大越嚴格）</label>
    <input type="number" id="contrast" value="12" min="1" max="100">
  </div>
  <div class="param-group">
    <label>最小細胞面積（px²）</label>
    <input type="number" id="min-area" value="65" min="1">
  </div>
  <div class="param-group">
    <label>最大細胞面積（px²）</label>
    <input type="number" id="max-area" value="3000" min="1">
  </div>
  <div class="param-group">
    <label>&nbsp;</label>
    <div class="toggle-group">
      <input type="checkbox" id="split-toggle" checked>
      <label for="split-toggle">自動分裂相鄰細胞</label>
    </div>
  </div>
</div>

<div class="actions">
  <button class="btn" id="run-btn" onclick="runAnalysis()" disabled>開始計算</button>
  <button class="btn-outline" id="csv-btn" onclick="exportCSV()" style="display:none">匯出 CSV</button>
  <span id="file-count" style="color:#8b949e; font-size:0.85rem;"></span>
</div>

<div id="progress-text"></div>
<div id="progress-bar-wrap"><div id="progress-bar"></div></div>

<div class="summary" id="summary">
  <h2>總細胞數</h2>
  <div class="total" id="total-count">—</div>
  <div class="meta" id="summary-meta"></div>
</div>

<div id="result-section">
  <table>
    <thead><tr><th>預覽</th><th>檔名</th><th>細胞數</th></tr></thead>
    <tbody id="result-body"></tbody>
  </table>
</div>

<div id="lightbox">
  <span id="lightbox-close" onclick="closeLightbox()">✕</span>
  <img id="lightbox-img" src="">
</div>

<script>
let selectedFiles = [];
let results = [];

const dropZone = document.getElementById('drop-zone');
const folderInput = document.getElementById('folder-input');

dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', async e => {
  e.preventDefault();
  dropZone.classList.remove('over');
  const items = e.dataTransfer.items;
  if (!items) return;
  const files = [];
  const readEntry = (entry) => new Promise(resolve => {
    if (entry.isFile) {
      entry.file(f => { files.push(f); resolve(); });
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      const readAll = () => reader.readEntries(async entries => {
        if (!entries.length) return resolve();
        await Promise.all(entries.map(readEntry));
        readAll();
      });
      readAll();
    } else resolve();
  });
  await Promise.all(Array.from(items).map(item => {
    const entry = item.webkitGetAsEntry?.();
    return entry ? readEntry(entry) : Promise.resolve();
  }));
  handleFiles(files);
});
folderInput.addEventListener('change', e => handleFiles(Array.from(e.target.files)));

function handleFiles(fileList) {
  const all = Array.from(fileList);
  selectedFiles = all.filter(f => !f.name.startsWith('.') && f.size > 0);
  document.getElementById('file-count').textContent =
    `已選 ${selectedFiles.length} 張圖片（共 ${all.length} 個檔案）`;
  document.getElementById('run-btn').disabled = selectedFiles.length === 0;
}

async function runAnalysis() {
  if (!selectedFiles.length) return;

  results = [];
  document.getElementById('result-body').innerHTML = '';
  document.getElementById('result-section').style.display = 'none';
  document.getElementById('summary').style.display = 'none';
  document.getElementById('csv-btn').style.display = 'none';
  document.getElementById('run-btn').disabled = true;

  const progressWrap = document.getElementById('progress-bar-wrap');
  const progressBar = document.getElementById('progress-bar');
  const progressText = document.getElementById('progress-text');
  progressWrap.style.display = 'block';
  progressText.style.display = 'block';

  const contrast = document.getElementById('contrast').value;
  const minArea = document.getElementById('min-area').value;
  const maxArea = document.getElementById('max-area').value;
  const splitAdjacent = document.getElementById('split-toggle').checked;

  let done = 0;
  for (const file of selectedFiles) {
    progressText.textContent = `處理中 ${done + 1} / ${selectedFiles.length}：${file.name}`;
    progressBar.style.width = `${(done / selectedFiles.length) * 100}%`;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('contrast', contrast);
    formData.append('min_area', minArea);
    formData.append('max_area', maxArea);
    formData.append('split_adjacent', splitAdjacent ? '1' : '0');

    try {
      const res = await fetch('/analyze', { method: 'POST', body: formData });
      const data = await res.json();
      results.push({ name: file.name, count: data.count, preview: data.preview });
      appendRow(file.name, data.count, data.preview);
    } catch {
      results.push({ name: file.name, count: null, preview: null });
      appendRow(file.name, 'Error', null);
    }
    done++;
  }

  progressBar.style.width = '100%';
  progressText.textContent = `完成！共處理 ${done} 張`;

  const validCounts = results.filter(r => r.count !== null).map(r => r.count);
  const total = validCounts.reduce((a, b) => a + b, 0);
  document.getElementById('total-count').textContent = total.toLocaleString();
  document.getElementById('summary-meta').textContent =
    `${validCounts.length} 張圖片，平均 ${(total / validCounts.length).toFixed(1)} 個／張`;
  document.getElementById('summary').style.display = 'block';
  document.getElementById('result-section').style.display = 'block';
  document.getElementById('csv-btn').style.display = 'inline-block';
  document.getElementById('run-btn').disabled = false;
}

function appendRow(name, count, preview) {
  const tbody = document.getElementById('result-body');
  const tr = document.createElement('tr');
  const thumbHtml = preview
    ? `<img class="thumb" src="data:image/jpeg;base64,${preview}" onclick="openLightbox(this.src)">`
    : '<span style="color:#666">—</span>';
  tr.innerHTML = `
    <td>${thumbHtml}</td>
    <td style="color:#ccc">${name}</td>
    <td><span class="count-badge">${count}</span></td>`;
  tbody.appendChild(tr);
}

function exportCSV() {
  let csv = '檔名,細胞數\n';
  results.forEach(r => { csv += `"${r.name}",${r.count ?? ''}\n`; });
  const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8;' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'cell_counts.csv';
  a.click();
}

function openLightbox(src) {
  document.getElementById('lightbox-img').src = src;
  document.getElementById('lightbox').classList.add('show');
}
function closeLightbox() {
  document.getElementById('lightbox').classList.remove('show');
}
document.getElementById('lightbox').addEventListener('click', e => {
  if (e.target === document.getElementById('lightbox')) closeLightbox();
});
</script>
</body>
</html>"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/analyze", methods=["POST"])
def analyze():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "no file"}), 400

    contrast = int(request.form.get("contrast", 12))
    min_area = int(request.form.get("min_area", 65))
    max_area = int(request.form.get("max_area", 3000))
    split_adjacent = request.form.get("split_adjacent", "1") == "1"

    img_bytes = file.read()
    count, preview = count_cells(img_bytes, contrast_threshold=contrast,
                                 min_area=min_area, max_area=max_area,
                                 split_adjacent=split_adjacent)
    if count is None:
        return jsonify({"error": "無法讀取圖片"}), 400

    return jsonify({"count": count, "preview": preview})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print(f"啟動細胞計數網頁：http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
