"""Extract nested DailyMed releases, verify every file, then remove source ZIPs."""

import argparse
import hashlib
import io
import json
import os
import stat
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath


def hash_file(path, algorithm="sha256"):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name:
        raise ValueError(f"Unsafe archive path: {name}")
    return path


def extract_label(root, archive_name, member_name, payload):
    with zipfile.ZipFile(io.BytesIO(payload)) as inner:
        files = [item for item in inner.infolist() if not item.is_dir()]
        names = [item.filename for item in files]
        if len(names) != len(set(names)):
            raise ValueError(f"Duplicate entries: {member_name}")
        xml_names = [name for name in names if name.lower().endswith(".xml")]
        if len(xml_names) != 1:
            raise ValueError(f"Expected one SPL document: {member_name}, {xml_names}")
        xml_bytes = inner.read(xml_names[0])
        document = ET.fromstring(xml_bytes)

        def attr(tag, attribute):
            element = document.find("{*}" + tag)
            return element.get(attribute) if element is not None else None

        document_id = attr("id", "root")
        if not document_id or "/" in document_id or ".." in document_id:
            raise ValueError(f"Invalid SPL document ID: {document_id}")
        destination = root / "labels" / document_id.lower()
        destination.mkdir(parents=True, exist_ok=True)
        outputs = []
        for item in files:
            relative = safe_name(item.filename)
            if stat.S_ISLNK(item.external_attr >> 16):
                raise ValueError(f"Archive symlink: {item.filename}")
            content = xml_bytes if item.filename == xml_names[0] else inner.read(item)
            checksum = hashlib.sha256(content).hexdigest()
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                if hash_file(target) != checksum:
                    raise ValueError(f"Conflicting existing file: {target}")
            else:
                temporary = target.with_name(target.name + ".extracting")
                temporary.write_bytes(content)
                if hash_file(temporary) != checksum:
                    raise IOError(f"Extracted file verification failed: {target}")
                temporary.replace(target)
            outputs.append({"name": item.filename, "bytes": len(content), "sha256": checksum})
    return {"document_id": document_id, "set_id": attr("setId", "root"),
            "version": attr("versionNumber", "value"), "effective_time": attr("effectiveTime", "value"),
            "category": PurePosixPath(member_name).parts[0],
            "directory": destination.relative_to(root).as_posix(), "xml": xml_names[0],
            "archive": archive_name, "archive_member": member_name, "files": outputs}


def extract_archive(root, path, expected_md5, workers, delete):
    initial = path.stat()
    began = time.monotonic()
    audit = root / "manifests"
    audit.mkdir(parents=True, exist_ok=True)
    actual_md5 = hash_file(path, "md5")
    if expected_md5 and actual_md5 != expected_md5:
        raise ValueError(f"Official MD5 mismatch: {path.name}: {actual_md5} != {expected_md5}")
    checksum = hash_file(path)
    output = audit / (path.stem + ".jsonl")
    temporary = output.with_suffix(".jsonl.tmp")
    count = file_count = total_bytes = 0
    with zipfile.ZipFile(path) as outer, ThreadPoolExecutor(max_workers=workers) as pool, temporary.open("w") as stream:
        members = [item for item in outer.infolist() if not item.is_dir()]
        # Keep only a small batch of compressed labels in memory.
        for start in range(0, len(members), workers * 4):
            futures = []
            for member in members[start:start + workers * 4]:
                safe_name(member.filename)
                if not member.filename.lower().endswith(".zip"):
                    raise ValueError(f"Unexpected outer entry: {member.filename}")
                payload = outer.read(member)
                futures.append(pool.submit(extract_label, root, path.name, member.filename, payload))
            for future in futures:
                row = future.result()
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                count += 1
                file_count += len(row["files"])
                total_bytes += sum(f["bytes"] for f in row["files"])
            if start % (workers * 4 * 25) == 0:
                stream.flush()
                print(f"{path.name}: {count}/{len(members)} labels verified", flush=True)
        stream.flush()
        os.fsync(stream.fileno())
    final = path.stat()
    if (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise RuntimeError(f"Source archive changed during extraction: {path}")
    temporary.replace(output)
    report = {"archive": path.name, "bytes": initial.st_size, "md5": actual_md5, "sha256": checksum,
              "official_md5_match": bool(expected_md5), "labels": count, "files": file_count,
              "uncompressed_bytes": total_bytes, "seconds": time.monotonic() - began,
              "verified": "outer and inner ZIP CRC; extracted file SHA-256 readback",
              "manifest_sha256": hash_file(output), "source_deleted": False}
    receipt = audit / (path.stem + ".json")
    write_json(receipt, report)
    if delete:
        path.unlink()
        report["source_deleted"] = True
        write_json(receipt, report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--checksums", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--delete-verified-zips", action="store_true")
    args = parser.parse_args()
    expected = json.loads(args.checksums.read_text()) if args.checksums else {}
    for path in sorted(args.root.glob("*.zip")):
        extract_archive(args.root, path, expected.get(path.name), args.workers, args.delete_verified_zips)


if __name__ == "__main__":
    main()
