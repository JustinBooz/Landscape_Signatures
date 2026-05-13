"""
Pipeline Status Monitor — Lightweight HTTP Dashboard
=====================================================
Serves a live status page showing pipeline progress.
Auto-refreshes every 30 seconds.

Usage:
    conda run -n baukultur_vpr python monitor.py
    Then expose via: cloudflared tunnel --url http://localhost:8765
"""

import os
import sys
import json
import time
import glob
import http.server
import socketserver
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

PORT = 8765

STEPS = [
    (1,  "Manifest Construction",         "s01_manifest",          config.manifest_path()),
    (2,  "Embedding Normalization",       "s02_normalize",         config.normed_embeddings_shape_path()),
    (3,  "PCA Compression & Diagnostics", "s03_pca",               config.pca_selected_dim_path()),
    (4,  "FAISS kNN Search",              "s04_faiss_knn",         config.faiss_recall_path()),
    (5,  "Graph Construction",            "s05_graph",             config.graph_stats_path()),
    (6,  "Leiden Community Detection",    "s06_leiden",            config.leiden_parquet_path()),
    (7,  "Cluster Stability Diagnostics", "s07_stability",         config.stability_matrix_path()),
    (8,  "Medoid Extraction",             "s08_medoids",           None),
    (9,  "Auxiliary HDBSCAN",             "s09_hdbscan_aux",       config.hdbscan_labels_path()),
    (10, "UMAP Visualization",            "s10_umap_viz",          None),
    (11, "Spatial Validation",             "s11_spatial",           config.spatial_summary_path()),
    (12, "External Validation",            "s12_external",          config.external_validation_path()),
    (13, "Negative Controls",              "s13_negative_controls", config.negative_control_path()),
    (14, "Final Assembly & Report",        "s14_final_assembly",    config.master_parquet_path()),
]


def get_step_status(step_num, name, log_prefix, completion_file):
    """Determine if a step is complete, running, or pending."""
    log_file = os.path.join(config.BASE_DIR, f"{log_prefix}.log")

    # Check completion
    is_complete = False
    if completion_file and os.path.exists(completion_file):
        is_complete = True
    elif step_num == 8:  # medoids — check any resolution file
        for res in config.LEIDEN_RESOLUTIONS:
            if os.path.exists(config.medoids_path(res)):
                is_complete = True
                break
    elif step_num == 10:  # umap — check any sample
        for ns in config.UMAP_SAMPLE_SIZES:
            if os.path.exists(config.umap_sample_path(ns)):
                is_complete = True
                break

    # Get last log lines — for step 6 also check parallel log
    last_lines = []
    log_mtime = None
    log_candidates = [log_file]
    if step_num == 6:
        log_candidates.append(os.path.join(config.BASE_DIR, "leiden_parallel.log"))
    for lf in log_candidates:
        if os.path.exists(lf):
            mt = os.path.getmtime(lf)
            if log_mtime is None or mt > log_mtime:
                log_mtime = mt
                try:
                    with open(lf, 'r') as f:
                        lines = f.readlines()
                        last_lines = [l.rstrip() for l in lines[-5:]]
                except:
                    pass

    # Determine status
    if is_complete:
        status = "complete"
        icon = "✅"
    elif log_mtime and (time.time() - log_mtime) < 120:
        status = "running"
        icon = "🔄"
    elif log_mtime:
        status = "stalled_or_done"
        icon = "⏸️"
    else:
        status = "pending"
        icon = "⏳"

    return {
        "step": step_num,
        "name": name,
        "status": status,
        "icon": icon,
        "last_lines": last_lines,
        "log_mtime": datetime.fromtimestamp(log_mtime).strftime("%H:%M:%S") if log_mtime else "—",
    }


def get_process_info():
    """Check if the pipeline or parallel Leiden process is running."""
    try:
        import subprocess
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
        # Look for any pipeline-related python process
        markers = ['run_pipeline.py', 'leiden_parallel.py', 'run_pipeline_resilient']
        procs = []
        for line in result.stdout.split('\n'):
            if 'grep' in line or 'python' not in line:
                continue
            if any(m in line for m in markers):
                parts = line.split()
                if len(parts) >= 11:
                    procs.append({
                        "pid": parts[1],
                        "cpu": parts[2],
                        "mem_kb": int(parts[5]) if parts[5].isdigit() else 0,
                    })

        if procs:
            # Also count active leiden worker processes
            n_leiden = sum(1 for l in result.stdout.split('\n')
                         if 'leiden' in l.lower() and 'python' in l
                         and 'grep' not in l and 'monitor' not in l)
            total_mem = sum(p['mem_kb'] for p in procs)
            total_cpu = sum(float(p['cpu']) for p in procs)
            return {
                "running": True,
                "pid": procs[0]['pid'],
                "cpu": f"{total_cpu:.0f}%",
                "mem_pct": f"{total_mem / (125 * 1024 * 1024) * 100:.1f}%",
                "mem_gb": f"{total_mem / 1024 / 1024:.1f} GB",
                "workers": n_leiden,
            }
        return {"running": False}
    except:
        return {"running": False}


def get_disk_usage():
    """Get output directory size."""
    try:
        total = 0
        for f in glob.glob(os.path.join(config.OUTPUT_DIR, "*")):
            if os.path.isfile(f):
                total += os.path.getsize(f)
        return f"{total / 1e9:.1f} GB"
    except:
        return "?"


def get_output_files():
    """List output files with sizes."""
    files = []
    try:
        for f in sorted(glob.glob(os.path.join(config.OUTPUT_DIR, "*"))):
            if os.path.isfile(f):
                size = os.path.getsize(f)
                mtime = datetime.fromtimestamp(os.path.getmtime(f)).strftime("%Y-%m-%d %H:%M")
                files.append({
                    "name": os.path.basename(f),
                    "size": f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size / 1e3:.1f} KB",
                    "mtime": mtime,
                })
    except:
        pass
    return files


def generate_html():
    """Generate the status dashboard HTML."""
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    proc = get_process_info()
    disk = get_disk_usage()
    files = get_output_files()

    steps_html = ""
    current_step = None
    for step_num, name, log_prefix, completion_file in STEPS:
        info = get_step_status(step_num, name, log_prefix, completion_file)
        if info["status"] == "running":
            current_step = info

        # Color coding
        if info["status"] == "complete":
            bg = "#1a3a2a"
            border = "#2d8a4e"
        elif info["status"] == "running":
            bg = "#1a2a3a"
            border = "#4a9eff"
        elif info["status"] == "stalled_or_done":
            bg = "#3a2a1a"
            border = "#8a6a2d"
        else:
            bg = "#1a1a1a"
            border = "#333"

        log_html = ""
        if info["last_lines"]:
            log_text = "\n".join(info["last_lines"])
            log_html = f'<pre class="log-preview">{log_text}</pre>'

        steps_html += f"""
        <div class="step-card" style="background:{bg}; border-left: 4px solid {border};">
            <div class="step-header">
                <span class="step-icon">{info['icon']}</span>
                <span class="step-num">Step {step_num:02d}</span>
                <span class="step-name">{info['name']}</span>
                <span class="step-time">{info['log_mtime']}</span>
            </div>
            {log_html}
        </div>
        """

    # Process status
    if proc["running"]:
        workers_text = f" &nbsp;|&nbsp; Workers: {proc.get('workers', 1)}" if proc.get('workers', 0) > 1 else ""
        proc_html = f"""
        <div class="process-info running">
            <span class="proc-dot pulse"></span>
            Pipeline Running — PID {proc['pid']}
            &nbsp;|&nbsp; CPU: {proc['cpu']}
            &nbsp;|&nbsp; RAM: {proc['mem_gb']} ({proc['mem_pct']}){workers_text}
        </div>
        """
    else:
        proc_html = """
        <div class="process-info stopped">
            <span class="proc-dot"></span>
            Pipeline Not Running
        </div>
        """

    # Files table
    files_rows = ""
    for f in files:
        files_rows += f"<tr><td>{f['name']}</td><td>{f['size']}</td><td>{f['mtime']}</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="30">
    <title>Pipeline Monitor</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, sans-serif;
            background: #0d0d0d;
            color: #e0e0e0;
            padding: 24px;
            max-width: 900px;
            margin: 0 auto;
        }}
        h1 {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #fff;
            margin-bottom: 4px;
            letter-spacing: -0.02em;
        }}
        .subtitle {{
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 20px;
        }}
        .process-info {{
            padding: 12px 16px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-size: 0.9rem;
            font-weight: 500;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .process-info.running {{
            background: linear-gradient(135deg, #0a1a2a, #0d2240);
            border: 1px solid #1a3a5a;
            color: #7ab8ff;
        }}
        .process-info.stopped {{
            background: #1a1010;
            border: 1px solid #3a1a1a;
            color: #ff6b6b;
        }}
        .proc-dot {{
            width: 10px; height: 10px;
            border-radius: 50%;
            display: inline-block;
        }}
        .running .proc-dot {{ background: #4a9eff; }}
        .stopped .proc-dot {{ background: #ff4444; }}
        .pulse {{
            animation: pulse 1.5s ease-in-out infinite;
        }}
        @keyframes pulse {{
            0%, 100% {{ opacity: 1; }}
            50% {{ opacity: 0.3; }}
        }}
        .step-card {{
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 8px;
            transition: all 0.2s;
        }}
        .step-header {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .step-icon {{ font-size: 1.1rem; }}
        .step-num {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #888;
            min-width: 55px;
        }}
        .step-name {{
            font-weight: 500;
            flex: 1;
        }}
        .step-time {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.8rem;
            color: #666;
        }}
        .log-preview {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.72rem;
            color: #888;
            margin-top: 8px;
            padding: 8px 12px;
            background: rgba(0,0,0,0.3);
            border-radius: 4px;
            overflow-x: auto;
            white-space: pre;
            line-height: 1.5;
            max-height: 120px;
            overflow-y: auto;
        }}
        .section-title {{
            font-size: 0.85rem;
            font-weight: 600;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin: 24px 0 12px;
        }}
        .disk-info {{
            font-size: 0.85rem;
            color: #666;
            margin-bottom: 16px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
        }}
        th {{
            text-align: left;
            padding: 8px 12px;
            color: #666;
            font-weight: 500;
            border-bottom: 1px solid #222;
        }}
        td {{
            padding: 6px 12px;
            border-bottom: 1px solid #1a1a1a;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.75rem;
            color: #999;
        }}
        .footer {{
            margin-top: 24px;
            font-size: 0.75rem;
            color: #444;
            text-align: center;
        }}
    </style>
</head>
<body>
    <h1>🏔️ Landscape Embedding Pipeline</h1>
    <p class="subtitle">Graph-based visual community detection — {now} — auto-refresh 30s</p>

    {proc_html}

    <div class="section-title">Pipeline Steps</div>
    {steps_html}

    <div class="section-title">Output Files</div>
    <p class="disk-info">Total: {disk}</p>
    <table>
        <tr><th>File</th><th>Size</th><th>Modified</th></tr>
        {files_rows}
    </table>

    <p class="footer">Graph Pipeline Monitor · Auto-refreshes every 30 seconds</p>
</body>
</html>"""
    return html


class StatusHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/' or self.path == '/status':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            self.wfile.write(generate_html().encode('utf-8'))
        elif self.path == '/api/status':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            data = {
                "time": datetime.now().isoformat(),
                "process": get_process_info(),
                "disk": get_disk_usage(),
                "steps": [],
            }
            for step_num, name, log_prefix, completion_file in STEPS:
                info = get_step_status(step_num, name, log_prefix, completion_file)
                data["steps"].append(info)
            self.wfile.write(json.dumps(data, indent=2).encode('utf-8'))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress access logs


if __name__ == "__main__":
    print(f"Pipeline Monitor starting on http://localhost:{PORT}")
    print(f"Expose with: cloudflared tunnel --url http://localhost:{PORT}")
    with socketserver.TCPServer(("", PORT), StatusHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nMonitor stopped.")
