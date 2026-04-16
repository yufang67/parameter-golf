#!/usr/bin/env python3
"""Pack a plain Python training script into a self-extracting train_gpt.py.

Usage:
    python pack_train_gpt.py <input.py> [output.py]

If output is omitted, writes to train_gpt.py in the same directory as input.
"""
import base64
import lzma
import os
import sys


def pack(source_path: str, dest_path: str) -> None:
    with open(source_path, "rb") as f:
        raw = f.read()

    compressed = lzma.compress(
        raw,
        format=lzma.FORMAT_RAW,
        filters=[{"id": lzma.FILTER_LZMA2}],
    )
    encoded = base64.b85encode(compressed).decode("ascii")

    bootstrap = (
        f'import lzma as L,base64 as B,os,sys,runpy\n'
        f'_c=L.decompress(B.b85decode("{encoded}"),format=L.FORMAT_RAW,filters=[{{"id":L.FILTER_LZMA2}}])\n'
        f'_d=os.path.join(os.path.dirname(os.path.abspath(__file__)),"_train.py")\n'
        f'with open(_d,"wb") as _f:_f.write(_c)\n'
        f'os.environ["_ORIG_SCRIPT"]=os.path.abspath(__file__)\n'
        f'sys.argv[0]=_d;runpy.run_path(_d,run_name="__main__")'
    )

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(bootstrap)

    raw_kb = len(raw) / 1024
    packed_kb = os.path.getsize(dest_path) / 1024
    print(f"Packed {source_path} ({raw_kb:.1f} KB) -> {dest_path} ({packed_kb:.1f} KB)")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <input.py> [output.py]", file=sys.stderr)
        sys.exit(1)

    source = sys.argv[1]
    if len(sys.argv) >= 3:
        dest = sys.argv[2]
    else:
        dest = os.path.join(os.path.dirname(source), "train_gpt.py")

    pack(source, dest)


if __name__ == "__main__":
    main()
