from src.make_supplementary import esc, markdown


def test_escape_handles_latex_specials_and_symbols():
    assert esc("a_b & 5% ≥ 2 → x") == r"a\_b \& 5\% $\geq$ 2 $\rightarrow$ x"


def test_markdown_lists_paragraphs_and_inline():
    text = ("# Title\n\nFirst line\ncontinued.\n\n## Part\n\n- **Bold** item with `code_x`\n"
            "  continuation\n  - nested\n1. numbered\n")
    out = markdown(text)
    assert "Title" not in out
    assert "First line\ncontinued." in out
    assert "\\paragraph{Part}" in out
    assert "\\item \\textbf{Bold} item with \\nolinkurl{code_x}\ncontinuation" in out
    assert markdown("Run `a b_c`.").strip() == "Run \\texttt{a b\\_c}."
    assert out.count("\\begin{itemize}") == 2 and out.count("\\end{itemize}") == 2
    assert out.count("\\begin{enumerate}") == 1 and out.rstrip().endswith("\\end{enumerate}")
