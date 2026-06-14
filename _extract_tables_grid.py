# -*- coding: utf-8 -*-
"""Grid-accurate dump of every table: full cell matrix with (row,col) coords, NO dedup.
Lets the audit read merged ON|OFF columns unambiguously."""
from docx import Document
from docx.oxml.ns import qn
from docx.table import Table

doc = Document("DoAnTotNghiep_PhatDT.docx")

# collect all tables in document order (including SDT-nested)
tables = []
def walk(parent):
    for child in parent.iterchildren():
        if child.tag == qn('w:tbl'):
            tables.append(Table(child, doc))
        elif child.tag == qn('w:sdt'):
            c = child.find(qn('w:sdtContent'))
            if c is not None:
                walk(c)
        # also recurse paragraphs? tables only live at body/sdt/cell level
walk(doc.element.body)

out = []
for ti, t in enumerate(tables, 1):
    grid = t._tbl  # CT_Tbl
    nrows = len(t.rows)
    ncols = len(t.columns)
    out.append(f"===== TABLE T{ti:02d}  ({nrows} rows x {ncols} cols) =====")
    for ri, row in enumerate(t.rows):
        # full cell list WITHOUT dedup (shows merged-cell repeats so column index is explicit)
        vals = []
        for ci, c in enumerate(row.cells):
            vals.append(f"c{ci}={' '.join(c.text.split())}")
        out.append(f"  R{ri}: " + " || ".join(vals))
    out.append("")

with open("_docx_tables_grid.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print(f"wrote _docx_tables_grid.txt  tables={len(tables)}")
