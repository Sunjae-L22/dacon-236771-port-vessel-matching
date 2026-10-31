# author: opus
"""v20t: the user's v20 package with only model/catalog_ko.json replaced by the 72B Hangul table
(catalog_ko_72b.json, same format: English catalog name -> Hangul string). v20 uses the table for its
Korean-similarity weight (KO_W=4) and its 50 Korean-space candidates; no code changes.
On 342 real train Korean names: v20 table exact .456 / R@1 .667, 72B table exact .582 / R@1 .813.

usage: python make_v20t.py OUT.zip V20_ZIP TABLE_JSON
"""
import sys
import zipfile
from pathlib import Path


def main(out, v20_zip, table):
    out = Path(out)
    assert not out.exists(), out
    data = Path(table).read_bytes()
    src = zipfile.ZipFile(v20_zip)
    assert "model/catalog_ko.json" in src.namelist()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as w:
        for info in src.infolist():
            if info.is_dir():
                continue
            if info.filename == "model/catalog_ko.json":
                w.writestr(info.filename, data)
                continue
            with src.open(info) as fsrc, w.open(zipfile.ZipInfo(info.filename, info.date_time), "w",
                                                force_zip64=info.file_size > 50_000_000) as fdst:
                while chunk := fsrc.read(1 << 24):
                    fdst.write(chunk)
    print(out, round(out.stat().st_size / 1e9, 3), "GB")


if __name__ == "__main__":
    main(*sys.argv[1:])
