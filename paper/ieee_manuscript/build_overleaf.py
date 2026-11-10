"""Assemble the Overleaf upload: inline the generated tables into main.tex, check the LaTeX statically, zip.

Usage (from the project folder):
    python -X utf8 paper/ieee_manuscript/build_overleaf.py

main_source.tex is the editable source (tables included with \\input). The script writes overleaf/main.tex with
the table rows pasted in (safer than \\input inside tabular), copies figures and references.bib, runs static
checks (environments, braces, labels/refs, citation keys, figure files) and writes overleaf.zip.
"""
import re
import shutil
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "overleaf"


def inline_tables(text):
    def repl(match):
        return (HERE / match.group(1)).read_text(encoding="utf-8").rstrip("\n")
    return re.sub(r"\\input\{(tables/[^}]+\.tex)\}", repl, text)


def check(text, bib):
    problems = []
    body = re.sub(r"(?<!\\)%.*", "", text)
    begins = re.findall(r"\\begin\{([^}]+)\}", body)
    ends = re.findall(r"\\end\{([^}]+)\}", body)
    stack = []
    for kind, name in re.findall(r"\\(begin|end)\{([^}]+)\}", body):
        if kind == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            problems.append(f"environment mismatch at \\end{{{name}}}")
    if stack:
        problems.append(f"unclosed environments: {stack}")
    depth = 0
    for i, ch in enumerate(re.sub(r"\\[{}]", "", body)):
        depth += ch == "{"
        depth -= ch == "}"
        if depth < 0:
            problems.append(f"unbalanced closing brace near character {i}")
            break
    if depth > 0:
        problems.append(f"{depth} unclosed brace(s)")
    labels = set(re.findall(r"\\label\{([^}]+)\}", body))
    for ref in re.findall(r"\\ref\{([^}]+)\}", body):
        if ref not in labels:
            problems.append(f"undefined reference {ref}")
    keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    for group in re.findall(r"\\cite\{([^}]+)\}", body):
        for key in group.split(","):
            if key.strip() not in keys:
                problems.append(f"missing citation key {key.strip()}")
    for fig in re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", body):
        if not (HERE / fig).exists():
            problems.append(f"missing figure {fig}")
    dollars = len(re.findall(r"(?<!\\)\$", body))
    if dollars % 2:
        problems.append("odd number of $ signs")
    for row in re.findall(r"\\begin\{tabular\}\{([^}]+)\}(.*?)\\end\{tabular\}", body, re.S):
        columns = len(re.findall(r"[lcrp]", row[0]))
        for line in row[1].split("\\\\"):
            line = re.sub(r"\\(toprule|midrule|bottomrule|addlinespace\[[^]]*\]|cmidrule\([^)]*\)\{[^}]*\})", "", line).strip()
            if not line or line.startswith("\\multicolumn") and line.count("&") == 0:
                continue
            cells = line.count("&") + 1 + sum(int(n) - 1 for n in re.findall(r"\\multicolumn\{(\d+)\}", line))
            if cells != columns:
                problems.append(f"table row has {cells} cells, expected {columns}: {line[:60]}")
    return problems, len(begins), len(ends)


def main():
    source = (HERE / "main_source.tex").read_text(encoding="utf-8")
    bib = (HERE / "references.bib").read_text(encoding="utf-8")
    text = inline_tables(source)
    problems, b, e = check(text, bib)
    print(f"environments: {b} begin / {e} end; problems: {len(problems)}")
    for p in problems:
        print("  -", p)
    if problems:
        sys.exit(1)
    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "figures").mkdir(parents=True)
    (OUT / "main.tex").write_text(text, encoding="utf-8")
    shutil.copy(HERE / "references.bib", OUT / "references.bib")
    for fig in sorted((HERE / "figures").glob("*.pdf")):
        shutil.copy(fig, OUT / "figures" / fig.name)
    with zipfile.ZipFile(HERE / "overleaf.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(OUT.rglob("*")):
            if path.is_file():
                z.write(path, path.relative_to(OUT).as_posix())
    print(f"wrote {HERE / 'overleaf.zip'}")


if __name__ == "__main__":
    main()
