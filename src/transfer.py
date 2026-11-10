"""Move the project to another computer and verify the copy.

With a USB/external drive (on the Mac):
    python -m src.transfer copy /Volumes/<DRIVE> [--without-raw]
Without a drive, over the home Wi-Fi (on the Mac):
    python -m src.transfer archive ~/Desktop/ecg_swiss_transfer.tar [--without-raw]
    python -m src.transfer serve ~/Desktop/ecg_swiss_transfer.tar
On the new computer, inside the copied or unpacked folder:
    python -m src.transfer verify

`copy` writes <DRIVE>/ecg swiss: every project file except personal documents matching
EXCLUDE (the passport scan), macOS metadata and caches. Each file is hashed while it is copied,
the hashes go to TRANSFER_MANIFEST.json, and the copy is then re-read from the drive and checked.
`archive` writes the same files and manifest into one uncompressed .tar (the images are already
compressed) and re-reads it to check it. `serve` offers that one file to computers on the same
network at a URL with a random part, prints the URL, and serves nothing else; stop it with Ctrl+C.
`verify` re-checks every listed file wherever the folder now sits. `--without-raw` leaves out the
3.2 GB PTB-XL download (free from PhysioNet; only needed to re-render pages).
Refuses to copy or archive while a training run is active (a .training.lock exists).
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import http.server
import io
import json
from pathlib import Path
import secrets
import socket
import sys
import tarfile
import time

from src.common import ROOT

MANIFEST = "TRANSFER_MANIFEST.json"
MARKER = ".transfer_in_progress"
EXCLUDE = ["*passport*", ".DS_Store", "._*", "__pycache__", ".pytest_cache", ".ipynb_checkpoints", ".venv",
           ".experiment_log.lock", MARKER]
RAW = "ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3"
CHUNK = 8 * 1024 * 1024


def project_files(without_raw=False):
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(fnmatch.fnmatch(part, pattern) for part in relative.parts for pattern in EXCLUDE):
            continue
        if without_raw and relative.parts[0] == RAW:
            continue
        if path.is_file() and relative.as_posix() != MANIFEST:
            yield relative


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def verify(root=ROOT):
    manifest = json.loads((root / MANIFEST).read_text())
    problems = []
    for index, (name, entry) in enumerate(manifest["files"].items(), start=1):
        path = root / name
        if not path.is_file():
            problems.append(f"missing: {name}")
        elif path.stat().st_size != entry["bytes"] or sha256(path) != entry["sha256"]:
            problems.append(f"damaged: {name}")
        if index % 5000 == 0:
            print(f"  checked {index:,} of {len(manifest['files']):,} files", flush=True)
    return manifest, problems


def refuse_while_training():
    locks = list(ROOT.glob("models/*/*/.training.lock"))
    if locks:
        raise SystemExit(f"A training run is still active ({locks[0].parent.name}); wait for it to finish.")


def manifest_for(files, without_raw):
    return {"created_from": str(ROOT), "excluded_patterns": EXCLUDE, "raw_ptbxl_included": not without_raw,
            "files": files, "total_bytes": sum(entry["bytes"] for entry in files.values())}


class HashingReader:
    def __init__(self, handle):
        self.handle, self.digest = handle, hashlib.sha256()

    def read(self, size=-1):
        chunk = self.handle.read(size)
        self.digest.update(chunk)
        return chunk


def archive(output, without_raw=False):
    refuse_while_training()
    output = Path(output).expanduser()
    if output.exists():
        raise SystemExit(f"{output} already exists; choose another name.")
    partial = output.with_name(output.name + ".partial")
    files, started = {}, time.perf_counter()
    with tarfile.open(partial, "w", format=tarfile.PAX_FORMAT) as tar:
        for relative in project_files(without_raw):
            source = ROOT / relative
            info = tar.gettarinfo(source, arcname=f"{ROOT.name}/{relative.as_posix()}")
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            with source.open("rb") as handle:
                reader = HashingReader(handle)
                tar.addfile(info, reader)
            files[relative.as_posix()] = {"bytes": info.size, "sha256": reader.digest.hexdigest()}
            if len(files) % 5000 == 0:
                print(f"  archived {len(files):,} files, {time.perf_counter() - started:.0f} s", flush=True)
        payload = (json.dumps(manifest_for(files, without_raw), indent=1) + "\n").encode()
        info = tarfile.TarInfo(f"{ROOT.name}/{MANIFEST}")
        info.size, info.mtime = len(payload), int(time.time())
        tar.addfile(info, io.BytesIO(payload))
    print(f"Archived {len(files):,} files. Re-reading the archive to check it...", flush=True)
    seen = 0
    with tarfile.open(partial) as tar:
        for member in tar:
            name = member.name.split("/", 1)[1]
            if not member.isfile() or name == MANIFEST:
                continue
            digest, handle = hashlib.sha256(), tar.extractfile(member)
            while chunk := handle.read(CHUNK):
                digest.update(chunk)
            if files.get(name, {}).get("sha256") != digest.hexdigest():
                raise SystemExit(f"Archive check FAILED at {name}")
            seen += 1
    if seen != len(files):
        raise SystemExit(f"Archive check FAILED: {seen} of {len(files)} files readable")
    partial.replace(output)
    print(f"Archive complete and verified: {output} ({output.stat().st_size / 1e9:.1f} GB)")


def local_address():
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.connect(("192.0.2.1", 9))  # no packet is sent; picks the network interface
        return probe.getsockname()[0]


def serve(path, port=8000):
    path = Path(path).expanduser()
    url_path = f"/{secrets.token_urlsafe(8)}/{path.name}"

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path != url_path:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/x-tar")
            self.send_header("Content-Length", str(path.stat().st_size))
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
            self.end_headers()
            with path.open("rb") as handle:
                while chunk := handle.read(CHUNK):
                    self.wfile.write(chunk)
            print(f"Sent the whole file to {self.client_address[0]}.", flush=True)

    print(f"On the other computer (same Wi-Fi), open this address in a web browser:\n\n"
          f"    http://{local_address()}:{port}{url_path}\n\nPress Ctrl+C here once the download has finished.", flush=True)
    http.server.ThreadingHTTPServer(("", port), Handler).serve_forever()


def copy(drive, without_raw=False):
    refuse_while_training()
    destination = Path(drive) / ROOT.name
    if destination.exists() and not (destination / MARKER).exists():
        raise SystemExit(f"{destination} already exists and is not an unfinished copy; choose another drive or folder.")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / MARKER).write_text("copy started; rerun the same command to finish\n")
    files, total, started = {}, 0, time.perf_counter()
    for relative in project_files(without_raw):
        source, target = ROOT / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        with source.open("rb") as reader, target.open("wb") as writer:
            while chunk := reader.read(CHUNK):
                digest.update(chunk)
                writer.write(chunk)
        files[relative.as_posix()] = {"bytes": source.stat().st_size, "sha256": digest.hexdigest()}
        total += files[relative.as_posix()]["bytes"]
        if len(files) % 2000 == 0:
            print(f"  copied {len(files):,} files, {total / 1e9:.1f} GB, {time.perf_counter() - started:.0f} s", flush=True)
    (destination / MANIFEST).write_text(json.dumps(manifest_for(files, without_raw), indent=1) + "\n")
    print(f"Copied {len(files):,} files ({total / 1e9:.1f} GB). Re-reading the copy from the drive to check it...", flush=True)
    _, problems = verify(destination)
    if problems:
        raise SystemExit("Copy check FAILED:\n" + "\n".join(problems[:20]))
    (destination / MARKER).unlink()
    print(f"Copy complete and verified: {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["copy", "archive", "serve", "verify"])
    parser.add_argument("target", nargs="?", help="drive (copy) or .tar file (archive, serve)")
    parser.add_argument("--without-raw", action="store_true")
    arguments = parser.parse_args()
    if arguments.mode != "verify" and not arguments.target:
        parser.error(f"{arguments.mode} needs a target: a drive for copy, a .tar path for archive/serve")
    if arguments.mode == "copy":
        copy(arguments.target, arguments.without_raw)
    elif arguments.mode == "archive":
        archive(arguments.target, arguments.without_raw)
    elif arguments.mode == "serve":
        serve(arguments.target)
    else:
        manifest, problems = verify()
        if problems:
            print("Copy check FAILED:\n" + "\n".join(problems[:20]))
            sys.exit(1)
        print(f"All {len(manifest['files']):,} files present and intact.")
