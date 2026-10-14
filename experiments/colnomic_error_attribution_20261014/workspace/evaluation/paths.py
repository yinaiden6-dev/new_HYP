"""Resolve archived paths without changing signed experiment manifests."""
from pathlib import Path

from drugpack.runtime import LEGACY_ROOTS, archived_path

ROOT = Path(__file__).resolve().parents[1]


def project_path(value, root=ROOT):
    return archived_path(value, root)
