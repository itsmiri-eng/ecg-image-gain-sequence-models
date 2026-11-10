"""Check that source JPEGs exactly match the retained original image archive."""
import hashlib
import zipfile
from pathlib import Path

from src.common import ROOT, sha256_file, write_json


def main():
    archive = ROOT / "mendely images.zip"
    local = {}
    for path in (ROOT / "ecg files").glob("*/*.jpg"):
        if path.name in local:
            raise ValueError(f"Ambiguous basename: {path.name}")
        local[path.name] = path
    matches, mismatches, absent, archived = 0, [], [], set()
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            if not name.lower().endswith((".jpg", ".jpeg", ".png")) or name.startswith("__MACOSX/"):
                continue
            basename = Path(name).name
            if basename in archived:
                raise ValueError(f"Ambiguous archive basename: {basename}")
            archived.add(basename)
            if basename not in local:
                absent.append(name)
            elif hashlib.sha256(handle.read(name)).hexdigest() == sha256_file(local[basename]):
                matches += 1
            else:
                mismatches.append(name)
    result = {"archive": archive.name, "archive_sha256": sha256_file(archive),
              "archive_image_entries": len(archived), "local_source_images": len(local),
              "exact_byte_matches": matches, "archive_images_missing_locally": absent,
              "byte_mismatches": mismatches, "local_images_not_in_archive": sorted(set(local) - archived),
              "notebook_checkpoints_included": False, "raw_files_changed": False}
    write_json(ROOT / "data/audit/archive_provenance.json", result)
    print(result)


if __name__ == "__main__":
    main()
