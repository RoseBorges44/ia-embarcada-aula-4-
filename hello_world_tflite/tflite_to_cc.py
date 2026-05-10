"""Converte um arquivo .tflite em model.cc no formato esperado por
esp-tflite-micro (variaveis g_model[] e g_model_len). Substitui o uso de
`xxd -i` para quem nao tem o utilitario instalado.

Uso:
    python tflite_to_cc.py <caminho_para_modelo.tflite> [destino_main_dir]

Por padrao escreve em ./main/model.cc.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

LICENSE_HEADER = """\
/* Generated from a .tflite flatbuffer.
 * Equivalent to: xxd -i model.tflite > model.cc
 * Variavel renomeada para g_model[] / g_model_len para casar com main_functions.cc.
 */
"""


def to_cc(tflite_path: Path, out_path: Path) -> None:
    data = tflite_path.read_bytes()
    lines: list[str] = []
    lines.append(LICENSE_HEADER)
    lines.append('#include "model.h"\n')
    lines.append("// Aligned to 8 bytes as required by tflite::GetModel().\n")
    lines.append("alignas(8) const unsigned char g_model[] = {\n")

    bytes_per_line = 12
    for i in range(0, len(data), bytes_per_line):
        chunk = data[i : i + bytes_per_line]
        hex_bytes = ", ".join(f"0x{b:02x}" for b in chunk)
        lines.append(f"  {hex_bytes},\n")

    lines.append("};\n")
    lines.append(f"const int g_model_len = {len(data)};\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"OK: {out_path} ({len(data)} bytes)")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    tflite_path = Path(sys.argv[1]).expanduser().resolve()
    if not tflite_path.is_file():
        print(f"ERRO: arquivo nao encontrado: {tflite_path}")
        return 2

    if len(sys.argv) >= 3:
        main_dir = Path(sys.argv[2]).expanduser().resolve()
    else:
        main_dir = Path(__file__).parent / "main"

    out_path = main_dir / "model.cc"
    to_cc(tflite_path, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
