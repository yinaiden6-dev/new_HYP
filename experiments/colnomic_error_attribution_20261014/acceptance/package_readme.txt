ColNomic complete frozen audit archive

Status: final producer readiness confirmed on 2026-10-14. This note is captured
before sealing. The actual final archive/SHA/CRC and independent decoded-workspace
receipts are delivered alongside the archive; do not infer their status from
this note. The member manifest records the final inventory and source digests.

Build (from the workspace root, after readiness):
  python scripts/colnomic_mechanism/package_full_audit.py --build --ready

The default archive is one tar.gz with internal hardlinks for identical file
contents. All workspace-relative paths are preserved. Each hardlink points to
an earlier regular member of this same archive; no symlink, external workspace
target, absolute link target or link chain is accepted. A ZIP64 fallback is:
  python scripts/colnomic_mechanism/package_full_audit.py --build --ready --archive-format zip64 --volume-gib 2

Regular file permissions are 0644 plus the source's ordinary execute bits.
Setuid, setgid and sticky bits are removed. Identical bytes with different
execute permissions are stored separately, so internal hardlinks never merge
different executable modes. Safe decoded validation restores and checks these
permissions; REPRODUCE.sh remains directly executable after extraction.

Build validation actually hashes every logical source file, writes the archive,
reads every archived payload, verifies its SHA256, validates the complete gzip
CRC32/ISIZE (or all ZIP member CRC32 values), and hashes all sources again. By
default it also decodes into delivery_support/archive_TIMESTAMP/
extracted_validation_workspace and independently checks every logical file's
SHA, both raw Token bank axes, 1117 queries/5405 references, and the 35 original
full-gallery score matrices against all 39095 original predictions. Relative
HTML, Markdown and per-case JSON links are checked from this decoded workspace. Actual
receipts, archive SHA256 sidecars and the complete member manifest are written
to delivery_support. No verification pass is assumed from an older report.

Read-only verification and safe extraction:
  python scripts/colnomic_mechanism/package_full_audit.py --verify-archive PATH/colnomic_full_audit.tar.gz --extract-to EMPTY_DIRECTORY

When the archive's .sha256 sidecar is adjacent to it, verification also checks
that external archive digest. Extract-to must be empty. Standard Linux tar can
also extract the internal hardlinks. After extraction, from its root:
  python scripts/colnomic_mechanism/package_full_audit.py --root . --verify-extracted --axes --links

The archive requires installed Python analysis libraries; their actual versions
and repository requirements are included, rather than a copied virtualenv.
Large backbone, adapter, GPU encoder weights and private credentials are not
included. Frozen raw Token banks, OCR, original scalar scores, 1117 query images,
all 5405 gallery images, source metadata, annotations, old/new manual boxes,
current and four-stage historical analyses and their code are included.
Archive paths recreate the workspace layout used to derive ROOT in the code.
Absolute historical paths remain provenance and may need explicit rebasing for
historical scripts. This is a self-contained frozen-bank/score audit; it does
not claim portable bitwise reproduction of historical neural encoding.

Known historical provenance gaps are reported separately: old SAM3 intermediate
region records and the original upstream gallery checkpoint/runtime can be
absent while the completed 35 score matrices and frozen gallery bank remain
fully replayable. External encoding/deployment dependencies are not missing
frozen-analysis evidence. The final manifest and dependency_gaps.json state the
actual counts after the last inventory.

Linked repository README files also bring their concrete small documents,
figures and protocol/verification records into the archive. One historical
report contains an already incorrect relative link. Its original bytes remain
unchanged; an archive-only same-SHA JSON compatibility copy makes that link
work after extraction. The manifest records compatibility_alias_of and the
original source defect, and decoded validation checks both copies' actual SHA.
No compatibility file is created in the source workspace.

Preliminary duplicate audit (2026-10-14): 8,924,843,189 bytes of equal-SHA content
across 4351 groups / 6335 duplicate paths can be stored once. Initial inventory
logical bytes 43,899,811,921 (~40.89 GiB), unique payload ~32.57 GiB. A preliminary
tar.gz estimate is ~29.92-31.98 GiB; compression was not measured and pending
modules will change this. This earlier byte-only estimate also predates the
executable-mode separation rule. Final sealing re-inventories and re-hashes
everything, and reports actual byte-and-mode hardlink savings in its manifest.
The recovered pre-readiness inventory contains 54,899 logical source files,
46,728,627,105 bytes (~43.52 GiB), and no missing required frozen-analysis
path. Final module production is still in progress and no complete archive has
yet been sealed. The final build re-inventories before hashing and compression.

The 1,009 declared provenance/deployment entries are classified rather than
treated as missing scientific output: 1,000 historical SAM3 intermediate region
records, eight external encoding/deployment declarations covering four unique
paths, and one configuration path template. Original SAM3 scalar scores and
completed processed inputs are included. This does not recreate those missing
historical preprocessing records or the unavailable upstream gallery runtime.

Current packager source SHA256 is recorded in self_test_validation.json.
Twenty-two real archive fixtures passed, including SHA/CRC corruption rejection,
member/hardlink path escape rejection, explicit readiness gating, actual
extraction and resolution of the historical compatibility JSON member, an
actually executed decoded REPRODUCE.sh, and equal-SHA/different-mode inode
separation. A hardlink that merges different execute permissions is rejected.
The ZIP64 fallback also has a real fixture checking executable and virtual
member permission metadata against the member manifest.
