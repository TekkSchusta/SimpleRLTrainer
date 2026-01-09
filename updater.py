import os
import sys
import json
import time
import threading
import webbrowser

def _log(msg):
    try:
        ts=time.strftime('%Y-%m-%d %H:%M:%S')
    except Exception:
        ts=''
    line=f"{ts} [updater] {msg}"
    try:
        print(line)
    except Exception:
        pass
    try:
        out_dir=os.path.join('outputs','update')
        os.makedirs(out_dir,exist_ok=True)
        with open(os.path.join(out_dir,'updater.log'),'a',encoding='utf-8') as f:
            f.write(line+'\n')
    except Exception:
        pass
def _read_json(path):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except Exception:
        return {}

def _read_version():
    try:
        with open('VERSION', 'r') as f:
            return f.read().strip()
    except Exception:
        return '0.0.0'

def _fetch_latest(repo):
    url = f'https://api.github.com/repos/{repo}/releases/latest'
    try:
        import requests  # likely present via other deps
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            _log(f"Fetched latest release metadata from {url}")
            return r.json()
    except Exception:
        pass
    try:
        import urllib.request, json as _json
        with urllib.request.urlopen(url, timeout=5) as resp:
            _log(f"Fetched latest release metadata via urllib from {url}")
            return _json.loads(resp.read().decode('utf-8'))
    except Exception:
        _log("Failed to fetch latest release metadata")
        return None

def _download_zip(repo, tag, out_path):
    zip_url = f'https://github.com/{repo}/archive/refs/tags/{tag}.zip'
    try:
        import requests
        _log(f"Downloading zip: {zip_url}")
        r = requests.get(zip_url, timeout=30)
        if r.status_code == 200:
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, 'wb') as f:
                f.write(r.content)
            _log(f"Saved zip to {out_path}")
            return True
    except Exception:
        _log("Requests download failed, trying urllib")
        pass
    try:
        import urllib.request
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        urllib.request.urlretrieve(zip_url, out_path)
        _log(f"Saved zip to {out_path} (urllib)")
        return True
    except Exception:
        _log("Failed to download zip")
        return False

def _extract_zip(zip_path, extract_dir):
    try:
        import zipfile
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(extract_dir)
        _log(f"Extracted zip to {extract_dir}")
        return True
    except Exception:
        _log("Failed to extract zip")
        return False

def _first_component(path):
    p = path.replace('\\', '/').strip('/')
    return p.split('/')[0] if p else ''

def _strip_top_folder(extract_dir):
    # Many Github source zips contain a single top-level folder repo-tag/
    try:
        entries = os.listdir(extract_dir)
        if len(entries) == 1:
            top = os.path.join(extract_dir, entries[0])
            if os.path.isdir(top):
                return top
        return extract_dir
    except Exception:
        return extract_dir

def _safe_merge(src_root, dst_root):
    # Merge files from src_root into dst_root without deleting existing content.
    # Special handling for models directories: only add new files/dirs, never overwrite or delete.
    models_dirs = {
        os.path.normpath('models'),
        os.path.normpath('rlbot/TekksTrainer/models')
    }
    for root, dirs, files in os.walk(src_root):
        rel_root = os.path.relpath(root, src_root)
        rel_root = '' if rel_root == '.' else rel_root
        target_dir = os.path.normpath(os.path.join(dst_root, rel_root))
        # Ensure directory exists
        try:
            os.makedirs(target_dir, exist_ok=True)
        except Exception:
            pass
        # Determine if this path is inside a protected models directory
        parts = rel_root.replace('\\', '/').split('/') if rel_root else []
        inside_models = False
        if parts:
            # Check any prefix that ends with 'models'
            for i in range(1, len(parts)+1):
                check = '/'.join(parts[:i])
                if os.path.normpath(check).endswith('models') or os.path.normpath(check) in models_dirs:
                    inside_models = True
                    break
        # Copy files
        for f in files:
            src_file = os.path.join(root, f)
            dst_file = os.path.join(target_dir, f)
            try:
                if inside_models:
                    # Only add new files, never overwrite existing
                    if not os.path.exists(dst_file):
                        # write file atomically
                        tmp = dst_file + '.tmp__update'
                        with open(src_file, 'rb') as sf, open(tmp, 'wb') as df:
                            df.write(sf.read())
                        os.replace(tmp, dst_file)
                else:
                    # Overwrite other files safely
                    tmp = dst_file + '.tmp__update'
                    with open(src_file, 'rb') as sf, open(tmp, 'wb') as df:
                        df.write(sf.read())
                    os.replace(tmp, dst_file)
            except Exception:
                # ignore copy errors to avoid breaking runtime
                pass
    _log("Merge complete")

def _should_prompt():
    if os.getenv('RLBot') or os.getenv('NO_UPDATE_PROMPT'):
        return False
    try:
        return sys.stdin.isatty()
    except Exception:
        return False

def _check_impl(script_name):
    cache_path = os.path.join('.cache_update_check')
    now = time.time()
    try:
        if os.path.exists(cache_path):
            last = float(open(cache_path, 'r').read().strip() or '0')
            if now - last < 6 * 3600:
                _log("Skipping update check due to TTL cache")
                return
    except Exception:
        pass
    cfg = _read_json('config.json')
    repo = (cfg.get('update', {}) or {}).get('repo', '')
    if not repo:
        _log("No repo configured for updater")
        return
    latest = _fetch_latest(repo)
    if not latest:
        _log("No latest release data returned")
        return
    local_ver = _read_version()
    tag = str(latest.get('tag_name') or '').strip()
    _log(f"Latest tag {tag}, local version {local_ver}")
    if not tag or tag == local_ver:
        try:
            with open(cache_path, 'w') as f:
                f.write(str(now))
        except Exception:
            pass
        _log("No update available")
        return
    rel_url = latest.get('html_url') or f'https://github.com/{repo}/releases/tag/{tag}'
    msg = f'New release available: {tag} (current {local_ver}). Open in browser? [y/N] '
    auto_update = bool((cfg.get('update', {}) or {}).get('auto', True))
    out_zip = os.path.join('outputs', 'update', f'{tag}.zip')
    if auto_update:
        _log("Auto-update is enabled")
        ok = _download_zip(repo, tag, out_zip)
        if ok:
            extract_dir = os.path.join('outputs', 'update', f'unpack_{tag}')
            os.makedirs(extract_dir, exist_ok=True)
            if _extract_zip(out_zip, extract_dir):
                src_root = _strip_top_folder(extract_dir)
                _safe_merge(src_root, os.getcwd())
                try:
                    with open('VERSION', 'w') as vf:
                        vf.write(str(tag))
                except Exception:
                    pass
                _log(f"Updated project to {tag}")
            else:
                _log('Failed to extract update zip')
        else:
            _log('Failed to download update zip')
    else:
        # prompt user if interactive
        if _should_prompt():
            try:
                ans = input(msg).strip().lower()
            except Exception:
                ans = ''
            if ans == 'y':
                try:
                    webbrowser.open(rel_url)
                except Exception:
                    pass
            _download_zip(repo, tag, out_zip)
        else:
            _log(f"New release {tag} available: {rel_url}")
    try:
        with open(cache_path, 'w') as f:
            f.write(str(now))
    except Exception:
        pass

def check_for_update(script_name=None):
    try:
        # run in background to avoid delaying startup
        t = threading.Thread(target=_check_impl, args=(script_name,), daemon=True)
        t.start()
    except Exception:
        pass

def check_for_update_blocking_ui(script_name=None):
    try:
        cfg = _read_json('config.json')
        repo = (cfg.get('update', {}) or {}).get('repo', '')
        if not repo:
            _log('No repo configured for updater')
            return
        latest = _fetch_latest(repo)
        if not latest:
            _log('No latest release data returned')
            return
        local_ver = _read_version()
        tag = str(latest.get('tag_name') or '').strip()
        _log(f"Latest tag {tag}, local version {local_ver}")
        if not tag or tag == local_ver:
            return
        msg = f'New release available: {tag} (current {local_ver}).\nDownload and update now?'
        try:
            import tkinter as tk
            from tkinter import messagebox
            root=tk.Tk(); root.withdraw()
            yes=messagebox.askyesno('TekksTrainer Update', msg)
            root.destroy()
        except Exception:
            if not _should_prompt():
                return
            try:
                yes=input(msg+' [y/N] ').strip().lower()=="y"
            except Exception:
                yes=False
        if yes:
            _log('User accepted update')
            out_zip = os.path.join('outputs', 'update', f'{tag}.zip')
            ok=_download_zip(repo, tag, out_zip)
            if ok:
                extract_dir = os.path.join('outputs', 'update', f'unpack_{tag}')
                os.makedirs(extract_dir, exist_ok=True)
                if _extract_zip(out_zip, extract_dir):
                    src_root=_strip_top_folder(extract_dir)
                    _safe_merge(src_root, os.getcwd())
                    try:
                        with open('VERSION','w') as vf:
                            vf.write(str(tag))
                    except Exception:
                        pass
                    return True
            return False
        else:
            _log('User declined update')
            return False
    except Exception:
        return False
