#!/usr/bin/env python3
"""Replay original frozen accounting, normalizing its JSON tuple/list boundary.

The producer stores intervention pairs as Python tuples; JSON restores lists.
Only that representation is converted before its strict whole-result compare.
No numerical comparison or source binding is relaxed.
"""
import sys
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import explain_rc_shared_query_prior_mass_content_v1 as source

read = source.base.read
stored = read(source.OUT / "result.json")
source.base.need(source.base.sha(Path(source.__file__)) == stored["sources"]["program"]["sha256"], "ORIGINAL_PRODUCER_SOURCE_DRIFT")
source.base.need(source.base.encode(stored["interventions"]) == source.base.encode(source.MODES), "INTERVENTION_SCHEMA_DRIFT")


def compatible_read(path):
    value = read(path)
    if Path(path).resolve() == (source.OUT / "result.json").resolve():
        value["interventions"] = {key: tuple(pair) for key, pair in value["interventions"].items()}
    return value


source.base.read = compatible_read
source.main(validate=True)
source.base.atomic(source.OUT / "validation_json_compatibility.json", {
    "status": "TUPLE_LIST_JSON_REPRESENTATION_ONLY_REPAIR",
    "validator": source.base.binding(Path(__file__)),
    "original_producer": stored["sources"]["program"],
    "result_sha256": source.base.sha(source.OUT / "result.json"),
    "numeric_or_source_comparisons_relaxed": False,
})
