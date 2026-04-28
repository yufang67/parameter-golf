#!/usr/bin/env python3
"""Pack a plain Python training script into a self-extracting train_gpt.py.

Usage:
    python pack_train_gpt.py <input.py> [output.py]

If output is omitted, writes to train_gpt.py in the same directory as input.
"""
import base64
import ast
import lzma
import os
import sys


class _AnnotationStripper(ast.NodeTransformer):
    @staticmethod
    def _is_triton_jit(node):
        for decorator in node.decorator_list:
            if (
                isinstance(decorator, ast.Attribute)
                and decorator.attr == "jit"
                and isinstance(decorator.value, ast.Name)
                and decorator.value.id == "triton"
            ):
                return True
        return False

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        if self._is_triton_jit(node):
            return node
        node.returns = None
        for arg in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        ):
            arg.annotation = None
        if node.args.vararg is not None:
            node.args.vararg.annotation = None
        if node.args.kwarg is not None:
            node.args.kwarg.annotation = None
        return node

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_AnnAssign(self, node):
        self.generic_visit(node)
        if node.value is None:
            return ast.Pass()
        return ast.Assign(targets=[node.target], value=node.value)


def _minify_source(raw: bytes, source_path: str) -> bytes:
    source = raw.decode("utf-8")
    tree = ast.parse(source, filename=source_path)
    tree = _AnnotationStripper().visit(tree)
    ast.fix_missing_locations(tree)
    tree.type_ignores = []
    minified = ast.unparse(tree)
    compile(minified, source_path, "exec")
    return minified.encode("utf-8")


def pack(source_path: str, dest_path: str) -> None:
    with open(source_path, "rb") as f:
        source_raw = f.read()
    raw = _minify_source(source_raw, source_path)

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

    source_kb = len(source_raw) / 1024
    raw_kb = len(raw) / 1024
    packed_kb = os.path.getsize(dest_path) / 1024
    print(
        f"Packed {source_path} ({source_kb:.1f} KB -> {raw_kb:.1f} KB minified) "
        f"-> {dest_path} ({packed_kb:.1f} KB)"
    )


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
