from __future__ import annotations
import hashlib
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
MANIFEST = ROOT / "SHA256SUMS.txt"
PROHIBITED_EXTS = {".csv", ".parquet", ".duckdb", ".sqlite", ".sqlite3", ".db", ".gz"}
PERSONAL_PATH = re.compile(r"/Users/panweibai|/home/panweibai", re.I)
# Exact fitted coefficients that were present in the locked private eICU scripts;
# none should be redistributed in the public package.
PROHIBITED_MODEL_VALUES = [
    "-4.206290240312703",
    "0.10652541500432669",
    "-4.524611492256407",
    "0.11926193422315796",
    "0.13093370264290544",
    "-4.5907312341513915",
    "0.12160476746253195",
]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    failures = []
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.name in {"SHA256SUMS.txt", "verify_release.py"}:
            continue
        rel = p.relative_to(ROOT)
        if p.suffix.lower() in PROHIBITED_EXTS or p.name.lower().endswith(".csv.gz"):
            failures.append(f"prohibited data/model artifact format: {rel}")
        if p.suffix.lower() in {".py", ".md", ".txt", ".json", ".cff"} or p.name == ".gitignore":
            text = p.read_text(encoding="utf-8", errors="ignore")
            if PERSONAL_PATH.search(text):
                failures.append(f"personal absolute path: {rel}")
            for value in PROHIBITED_MODEL_VALUES:
                if value in text:
                    failures.append(f"embedded fitted MIMIC coefficient in {rel}: {value}")

    if not MANIFEST.exists():
        failures.append("missing SHA256SUMS.txt")
    else:
        for line in MANIFEST.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            expected, rel = line.split(None, 1)
            p = ROOT / rel.strip()
            if not p.exists():
                failures.append(f"missing manifest file: {rel.strip()}")
            elif sha256(p) != expected:
                failures.append(f"hash mismatch: {rel.strip()}")

    if failures:
        print("PUBLIC RELEASE VERIFICATION: FAIL")
        for x in failures:
            print("-", x)
        return 1
    print("PUBLIC RELEASE VERIFICATION: PASS")
    return 0

if __name__ == "__main__":
    sys.exit(main())
