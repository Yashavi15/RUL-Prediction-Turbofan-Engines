"""
Dataset download and extraction script for NASA PHM08 turbofan engine degradation data.
Retrieves train.txt, test.txt, and final_test.txt from verified repositories.
"""

import os
import sys
import urllib.request
from pathlib import Path

DATA_FILES = {
    "train.txt": [
        "https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/PHM08/train.txt",
        "https://data.nasa.gov/docs/legacy/Challenge_Data.zip"
    ],
    "test.txt": [
        "https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/PHM08/test.txt"
    ],
    "final_test.txt": [
        "https://raw.githubusercontent.com/LahiruJayasinghe/RUL-Net/master/PHM08/final_test.txt"
    ]
}


def download_file(url: str, dest_path: Path) -> bool:
    print(f"[INFO] Downloading {dest_path.name} from {url}...")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) PHM08-Pipeline/1.0"}
    try:
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=40) as response:
            total_size = response.headers.get("Content-Length")
            total_bytes = int(total_size) if total_size else 0
            
            chunk_size = 64 * 1024
            downloaded = 0
            temp_path = dest_path.with_suffix(".tmp")
            
            with open(temp_path, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0:
                        percent = (downloaded / total_bytes) * 100
                        print(f"\r  Progress: {downloaded / (1024*1024):.2f}MB / {total_bytes / (1024*1024):.2f}MB ({percent:.1f}%)", end="", flush=True)
                    else:
                        print(f"\r  Progress: {downloaded / (1024*1024):.2f}MB downloaded", end="", flush=True)
            print()
            if temp_path.exists():
                temp_path.replace(dest_path)
            print(f"[SUCCESS] Saved {dest_path.name} ({dest_path.stat().st_size:,} bytes)")
            return True
    except Exception as e:
        print(f"\n[WARNING] Failed to download {dest_path.name} from {url}: {e}")
        return False


def setup_dataset(target_dir: str = "data") -> bool:
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    all_ok = True
    for fname, url_list in DATA_FILES.items():
        file_path = target_path / fname
        if file_path.exists() and file_path.stat().st_size > 1000:
            print(f"[INFO] '{fname}' already present ({file_path.stat().st_size:,} bytes). Skipping.")
            continue
        
        success = False
        for url in url_list:
            if download_file(url, file_path):
                success = True
                break
        if not success:
            all_ok = False
            print(f"[ERROR] Failed to acquire '{fname}'.")

    return all_ok


if __name__ == "__main__":
    dest = sys.argv[1] if len(sys.argv) > 1 else "data"
    ok = setup_dataset(dest)
    sys.exit(0 if ok else 1)
