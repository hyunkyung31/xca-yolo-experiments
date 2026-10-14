#!/usr/bin/env python3
"""Verify curated historical evidence without medical data or model training.

Optional local originals/outputs permit deeper preservation checks. No files are
modified, no notebook cells are executed, and no network operation is performed.
"""
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_text(cell: dict) -> str:
    text = cell.get("source", [])
    return "".join(text) if isinstance(text, list) else text


def verify(root: Path, original_root: Path | None = None,
           private_output_dir: Path | None = None) -> list[str]:
    manifest = json.loads((root / "docs/source_manifest.json").read_text())
    changes = json.loads((root / "docs/code_changes.json").read_text())
    change_cells = {(c["file"], c["cell"]): c for c in changes if "cell" in c}
    images = json.loads((root / "results/image_manifest.json").read_text())
    assert len(manifest["original_files"]) == 14, "Expected 11 notebooks and 3 original scripts"
    assert len(images) == 22, "Expected 22 original image outputs"
    assert sum(bool(r["public_path"]) for r in images) == 5, "Only aggregate charts may be public"
    assert len(list((root / "notebooks").glob("*.ipynb"))) == 11
    messages = []
    notebook_outputs = {}
    for item in manifest["original_files"]:
        organized = root / item["organized_path"]
        assert digest(organized.read_bytes()) == item["organized_sha256"], str(organized)
        original = None
        if original_root:
            original = original_root / Path(item["original_path"]).relative_to("yolo")
            assert digest(original.read_bytes()) == item["original_sha256"], str(original)
        if organized.suffix != ".ipynb":
            compile(organized.read_text(), str(organized), "exec")
            continue
        nb = json.loads(organized.read_text())
        assert nb["nbformat"] == 4
        assert len(nb["cells"]) == item["cells"]
        assert sum(len(c.get("outputs", [])) for c in nb["cells"]) == item["organized_outputs"]
        assert sum(c["cell_type"] == "code" for c in nb["cells"]) == item["code_cells"]
        notebook_outputs[organized.name] = nb
        if original:
            old_nb = json.loads(original.read_text())
            for i, (old, new) in enumerate(zip(old_nb["cells"], nb["cells"]), 1):
                assert old["cell_type"] == new["cell_type"]
                assert old.get("execution_count") == new.get("execution_count")
                expected = change_cells.get((item["organized_path"], i))
                if expected:
                    assert digest(source_text(old).encode()) == expected["before_sha256"]
                    assert digest(source_text(new).encode()) == expected["after_sha256"]
                else:
                    assert source_text(old) == source_text(new), f"Undocumented cell edit: {organized.name}:{i}"
            if private_output_dir:
                saved = json.loads((private_output_dir / (organized.name + ".outputs.json")).read_text())
                expected = [{"cell": i, "execution_count": c.get("execution_count"),
                             "outputs": c.get("outputs", [])}
                            for i, c in enumerate(old_nb["cells"], 1) if c["cell_type"] == "code"]
                assert saved == expected, f"Original outputs not preserved: {organized.name}"
    for record in images:
        if record["public_path"]:
            assert record["publication"] == "aggregate_chart"
            assert digest((root / record["public_path"]).read_bytes()) == record["sha256"]
        else:
            assert record["publication"] == "restricted_medical_image"
        if private_output_dir:
            assert digest((private_output_dir / record["filename"]).read_bytes()) == record["sha256"]
    for name, nb in notebook_outputs.items():
        for i, cell in enumerate(nb["cells"], 1):
            image_outputs = [o for o in cell.get("outputs", []) if "image/png" in o.get("data", {})]
            if (name == "01_dataset.ipynb" and i == 30) or name == "02_preprocessing.ipynb":
                assert not image_outputs, f"Restricted medical pixels: {name}:{i}"
            for output in image_outputs:
                data = output["data"]["image/png"]
                raw = base64.b64decode("".join(data) if isinstance(data, list) else data)
                assert any(r["sha256"] == digest(raw) and r["public_path"] for r in images)
    with (root / "results/tables/yolo_best_checkpoint_validation.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    assert [float(r["mAP50-95"]) for r in rows[:3]] == [0.121, 0.123, 0.127]
    assert all(r["images"] == "200" and r["instances"] == "406" for r in rows)
    epoch_rows = 0
    for p in (root / "results/tables").glob("05B_cell_*_history.csv"):
        with p.open() as f:
            rows = list(csv.DictReader(f))
        assert [int(r["epoch"]) for r in rows] == list(range(1, len(rows) + 1))
        epoch_rows += len(rows)
    assert epoch_rows == 190
    messages.append("Verified: 11 notebooks, 3 original scripts, 5 public charts, 6 YOLO validation rows, 190 FRCNN epoch rows.")
    if original_root:
        messages.append("Verified: all 14 original hashes, all cell sources against documented changes, unchanged execution counts.")
    if original_root and private_output_dir:
        messages.append("Verified: every original code-cell output object and all 22 privately preserved PNG hashes.")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-root", type=Path, help="Local extracted original yolo/ directory; never commit it")
    parser.add_argument("--private-output-dir", type=Path, help="Local private output archive; never commit it")
    args = parser.parse_args()
    if args.private_output_dir and not args.original_root:
        parser.error("--private-output-dir requires --original-root")
    root = Path(__file__).resolve().parents[1]
    for message in verify(root, args.original_root, args.private_output_dir):
        print(message)


if __name__ == "__main__":
    main()
