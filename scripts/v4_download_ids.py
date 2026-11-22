#!/usr/bin/env python3
"""Download cjkvi-ids.txt to the remote machine."""
import urllib.request
import sys

url = "https://raw.githubusercontent.com/Unifont/cjkvi-ids/master/ids.txt"
out_path = "/home/llm24g/cns_font/data_v2/ids.txt"

print(f"Downloading {url} -> {out_path}...")
try:
    urllib.request.urlretrieve(url, out_path)
    print("Done.")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)
