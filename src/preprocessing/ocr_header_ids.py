"""Audit printed ECG header IDs with macOS Vision; never infer patient identity.

Run with a Python environment containing PyObjC and Pillow. This helper hashes
source files, recognizes the fixed header region of one image per exact-byte
group, and records its OCR evidence. Printed IDs are unverified report/patient
identifiers: use them conservatively as grouping keys, not as proven patients.
No source files are changed. No manual corrections are performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path


def vision_classes():
    import Foundation
    import objc

    objc.loadBundle(
        "Vision", globals(), bundle_path="/System/Library/Frameworks/Vision.framework"
    )
    objc.registerMetaDataForSelector(
        b"VNImageRequestHandler", b"performRequests:error:",
        {"arguments": {3: {"type_modifier": objc._C_OUT}}},
    )
    return Foundation, objc.lookUpClass("VNRecognizeTextRequest"), objc.lookUpClass("VNImageRequestHandler")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("data/audit/header_ids.csv"))
    parser.add_argument("--limit", type=int, default=0, help="Debug limit; zero audits all unique images")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    groups = {}
    for path in sorted((root / "ecg files").glob("*/*.jpg")):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        groups.setdefault(digest, path)
    items = list(groups.items())[:args.limit or None]
    Foundation, Request, Handler = vision_classes()
    rows = []
    for index, (digest, path) in enumerate(items, 1):
        request = Request.alloc().init()
        request.setRecognitionLevel_(0)  # VNRequestTextRecognitionLevelAccurate.
        request.setUsesLanguageCorrection_(False)
        request.setUsesCPUOnly_(True)
        # Vision coordinates begin at bottom left; this covers the top-left header.
        request.setRegionOfInterest_(((0.0, 0.84), (0.35, 0.16)))
        handler = Handler.alloc().initWithURL_options_(
            Foundation.NSURL.fileURLWithPath_(str(path)), {}
        )
        ok, error = handler.performRequests_error_([request], None)
        observations = request.results() or []
        candidates = [observation.topCandidates_(1)[0] for observation in observations]
        texts = [str(candidate.string()) for candidate in candidates]
        matches = []
        for candidate in candidates:
            match = re.search(r"\b[I1l][D0]\s*[:;.]?\s*(\d{4,10})\b", str(candidate.string()))
            if match:
                matches.append((match.group(1), float(candidate.confidence())))
        ids = {value for value, _ in matches}
        status = "ocr_error" if not ok else "ambiguous" if len(ids) > 1 else "recognized" if ids else "unreadable"
        printed_id = next(iter(ids)) if len(ids) == 1 else ""
        confidence = max((score for value, score in matches if value == printed_id), default=0.0)
        rows.append({
            "path": path.relative_to(root).as_posix(), "file_sha256": digest,
            "printed_id": printed_id, "confidence": confidence, "status": status,
            "raw_text": " | ".join(texts), "error": str(error or "") if not ok else "",
            "method": "macOS_Vision_accurate_CPU_header_ROI_0_0.84_0.35_0.16",
            "manual_correction": "none", "vision_revision": int(request.revision()),
        })
        if index == 1 and not ok:
            raise RuntimeError(f"Vision OCR initialization failed: {error!r}; no output written")
        if index % 50 == 0:
            print(f"Recognized {index}/{len(items)} unique image headers", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    from collections import Counter
    print(json.dumps({"output": str(output), "rows": len(rows), "statuses": dict(Counter(row["status"] for row in rows)), "unique_recognized_ids": len({row["printed_id"] for row in rows if row["printed_id"]})}, sort_keys=True))


if __name__ == "__main__":
    main()
