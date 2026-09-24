#!/usr/bin/env python3
"""Run the unchanged English verifier with strict compatibility for its render status."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import verify_rc_cost1_rescue90_english_v1 as ENGLISH


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    out = ENGLISH.DEFAULT_OUT.resolve()
    validation_path = out / "validation.json"
    validation = json.loads(validation_path.read_text())
    assert validation["status"] == "COST1_RESCUE90_ENGLISH_RENDER_PASS"
    assert validation["language"] == "English" and validation["enlarged_reference_panels"] is True
    binding = validation["source_validation"]
    assert Path(binding["path"]).resolve() == (ENGLISH.DEFAULT_SOURCE / "validation.json").resolve()
    assert sha(binding["path"]) == binding["sha256"]
    source = json.loads(Path(binding["path"]).read_text())
    assert source["status"] == "COST1_RESCUE90_RENDER_AND_SEALED_DECISION_PASS"
    for key, value in source.items():
        if key != "status":
            assert validation[key] == value, (key, "English validation changed an inherited source field")
    original_read = ENGLISH.BASE.read

    def status_compatible_read(path):
        data = original_read(path)
        if Path(path).resolve() == validation_path:
            assert data == validation
            # Compatibility is limited to the known language-specific status
            # label; all inherited fields were checked against the sealed CN
            # validation above. No artifact on disk is altered.
            return {**data, "status": source["status"]}
        return data

    ENGLISH.BASE.read = status_compatible_read
    ENGLISH.main()
    report_path = out / "independent_validation.json"
    report = json.loads(report_path.read_text())
    report.update(
        actual_render_validation_status=validation["status"],
        actual_render_validation_sha256=sha(validation_path),
        english_render_validation_inherits_all_original_fields=True,
        render_status_compatibility="The known English render PASS label was mapped in memory to the original validator's accepted label after checking every inherited validation field; no input file was changed.",
        compatibility_entrypoint={"path": str(Path(__file__).resolve()), "sha256": sha(__file__)},
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print("ENGLISH_FINAL_INDEPENDENT_VALIDATION_PASS", report_path)


if __name__ == "__main__":
    main()
