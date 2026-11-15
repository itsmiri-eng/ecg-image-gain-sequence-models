"""Word copy of docs/DOCTOR_RESPONSE.md (plain Office Open XML written with the standard library).

Usage:
    python -X utf8 -m src.make_response_docx

Handles the Markdown subset used in that file: headings, paragraphs, bullet and numbered lists, pipe tables,
horizontal rules, **bold** and `code`. Writes docs/DOCTOR_RESPONSE.docx.
"""
from __future__ import annotations

import re
from xml.sax.saxutils import escape
import zipfile

from src.common import ROOT

SOURCE = ROOT / "docs/DOCTOR_RESPONSE.md"
OUT = ROOT / "docs/DOCTOR_RESPONSE.docx"


def runs(text, size=20, bold=False):
    """Inline **bold** and `code` to Word runs."""
    out = []
    for part in re.split(r"(\*\*[^*]+\*\*|`[^`]+`)", text):
        if not part:
            continue
        strong, mono = bold, False
        if part.startswith("**") and part.endswith("**"):
            part, strong = part[2:-2], True
        elif part.startswith("`") and part.endswith("`"):
            part, mono = part[1:-1], True
        part = part.replace("*", "")
        props = f"<w:sz w:val=\"{size}\"/>" + ("<w:b/>" if strong else "") + \
            ("<w:rFonts w:ascii=\"Consolas\" w:hAnsi=\"Consolas\"/>" if mono else "")
        out.append(f"<w:r><w:rPr>{props}</w:rPr><w:t xml:space=\"preserve\">{escape(part)}</w:t></w:r>")
    return "".join(out)


def paragraph(text, size=20, bold=False, indent=0, space_before=0):
    props = f"<w:spacing w:before=\"{space_before}\" w:after=\"80\"/>" + (f"<w:ind w:left=\"{indent}\"/>" if indent else "")
    return f"<w:p><w:pPr>{props}</w:pPr>{runs(text, size, bold)}</w:p>"


def table(rows):
    border = "".join(f"<w:{s} w:val=\"single\" w:sz=\"4\" w:space=\"0\" w:color=\"999999\"/>"
                     for s in ("top", "left", "bottom", "right", "insideH", "insideV"))
    out = [f"<w:tbl><w:tblPr><w:tblW w:w=\"5000\" w:type=\"pct\"/><w:tblBorders>{border}</w:tblBorders></w:tblPr>"]
    for k, row in enumerate(rows):
        out.append("<w:tr>" + "".join(f"<w:tc>{paragraph(cell, 18, bold=k == 0)}</w:tc>" for cell in row) + "</w:tr>")
    return "".join(out) + "</w:tbl>" + paragraph("")


def body(markdown):
    parts, buffer, rows = [], [], []

    def flush():
        if buffer:
            parts.append(paragraph(" ".join(buffer)))
            buffer.clear()
        if rows:
            parts.append(table(list(rows)))
            rows.clear()

    for line in markdown.splitlines():
        stripped = line.strip()
        if stripped.startswith("|"):
            if buffer:
                parts.append(paragraph(" ".join(buffer)))
                buffer.clear()
            if not re.fullmatch(r"[\s|:-]+", stripped):
                rows.append([c.strip() for c in stripped.strip("|").split("|")])
            continue
        if rows:
            flush()
        heading = re.match(r"^(#+)\s+(.*)", line)
        item = re.match(r"^(\s*)([-*]|\d+\.)\s+(.*)", line)
        if not stripped or stripped == "---":
            flush()
        elif heading:
            flush()
            level = len(heading.group(1))
            parts.append(paragraph(heading.group(2), {1: 32, 2: 26}.get(level, 22), bold=True, space_before=240))
        elif item:
            flush()
            marker = "•" if item.group(2) in "-*" else item.group(2)
            buffer.append(f"{marker} {item.group(3)}")
            parts.append(paragraph(" ".join(buffer), indent=360 + 30 * len(item.group(1))))
            buffer.clear()
        elif line.startswith("  ") and parts and not buffer:
            # continuation line of the previous list item: append to its last run
            parts[-1] = parts[-1].replace("</w:p>", runs(" " + stripped) + "</w:p>")
        else:
            buffer.append(stripped)
    flush()
    return "".join(parts)


def main():
    document = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
                "<w:document xmlns:w=\"http://schemas.openxmlformats.org/wordprocessingml/2006/main\"><w:body>"
                + body(SOURCE.read_text(encoding="utf-8")) +
                "<w:sectPr><w:pgSz w:w=\"11906\" w:h=\"16838\"/>"
                "<w:pgMar w:top=\"1100\" w:right=\"1100\" w:bottom=\"1100\" w:left=\"1100\"/></w:sectPr></w:body></w:document>")
    files = {
        "[Content_Types].xml": "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Types xmlns=\"http://"
            "schemas.openxmlformats.org/package/2006/content-types\"><Default Extension=\"rels\" ContentType=\"application/"
            "vnd.openxmlformats-package.relationships+xml\"/><Default Extension=\"xml\" ContentType=\"application/xml\"/>"
            "<Override PartName=\"/word/document.xml\" ContentType=\"application/vnd.openxmlformats-officedocument."
            "wordprocessingml.document.main+xml\"/></Types>",
        "_rels/.rels": "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?><Relationships xmlns=\"http://"
            "schemas.openxmlformats.org/package/2006/relationships\"><Relationship Id=\"rId1\" Type=\"http://schemas."
            "openxmlformats.org/officeDocument/2006/relationships/officeDocument\" Target=\"word/document.xml\"/>"
            "</Relationships>",
        "word/document.xml": document,
    }
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files.items():
            z.writestr(name, text)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
