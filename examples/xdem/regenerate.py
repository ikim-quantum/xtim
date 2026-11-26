"""Rewrite every <name>.xdem in this folder from its <name>.stim (default export options).

    python examples/xdem/regenerate.py
"""
from pathlib import Path

import xtim

HERE = Path(__file__).resolve().parent


def export(stim_path: Path) -> str:
    return xtim.Circuit(stim_path.read_text()).detector_error_model_text(format="xdem")


def main() -> None:
    for f in sorted(HERE.glob("*.stim")):
        out = f.with_suffix(".xdem")
        text = export(f)
        if not out.exists() or out.read_text() != text:
            out.write_text(text)
            print("wrote", out.name)


if __name__ == "__main__":
    main()
