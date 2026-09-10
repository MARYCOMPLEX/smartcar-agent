from __future__ import annotations
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def json_default(obj):
    if isinstance(obj, np.ndarray): return obj.tolist()
    if isinstance(obj, np.generic): return obj.item()
    if isinstance(obj, Path): return str(obj)
    raise TypeError(type(obj).__name__)


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=json_default, allow_nan=False), encoding="utf-8")


def sha256(path):
    with open(path, "rb") as f: return hashlib.file_digest(f, "sha256").hexdigest()


def write_csv(path, rows):
    if not rows: return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: json.dumps(v, default=json_default) if isinstance(v, (dict, list, np.ndarray)) else v for k,v in r.items()})
