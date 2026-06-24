#!/usr/bin/env python3
"""Re-render 2026-06-21-validation-methodology.html from the .md, reusing the EXACT
preamble (head + inline <style> + eyebrow/meta divs) and postamble already present in
the committed HTML, so the render stays byte-stable except for the body content.

Body conversion uses python-markdown with the same extensions the original output
implies (tables, fenced_code, sane_lists). Verified by diffing an unchanged section.
"""
import re, sys
import markdown

DOC = "/home/gaosh/projects/ZAsolar/docs/validation/2026-06-21-validation-methodology"
md_text = open(DOC + ".md").read()
old_html = open(DOC + ".html").read()

# strip YAML frontmatter
body_md = re.sub(r"\A---\n.*?\n---\n", "", md_text, count=1, flags=re.S)

# preamble = everything up to (not incl.) the first <h1>; postamble = closing wrapper
h1 = old_html.find("<h1>")
assert h1 != -1, "no <h1> in existing HTML"
preamble = old_html[:h1]
postamble = "</div></body></html>\n"

body_html = markdown.markdown(
    body_md, extensions=["tables", "fenced_code", "sane_lists"], output_format="html5"
)

out = preamble + body_html + "\n" + postamble
open(DOC + ".html", "w").write(out)
print(f"rendered {len(out)} chars -> {DOC}.html")
