"""Update the table of contents and export the report to PDF with Microsoft Word.

Optional step - needs Windows, Microsoft Word and the pywin32 package:
    python tools/export_pdf.py [path\\to\\report.docx]
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCX = ROOT / "reports" / "CCA3_Technical_Report.docx"


def export(docx_path: Path) -> Path:
    try:
        import win32com.client
    except ImportError:
        sys.exit("pywin32 is not installed (pip install pywin32), or this is not Windows. "
                 "Open the DOCX in Word, update the table of contents and use File > Save As > PDF instead.")
    docx_path = docx_path.resolve()
    pdf_path = docx_path.with_suffix(".pdf")
    word = win32com.client.DispatchEx("Word.Application")  # separate, invisible Word instance
    word.Visible = False
    word.DisplayAlerts = 0
    try:
        doc = word.Documents.Open(str(docx_path), ReadOnly=False, AddToRecentFiles=False)
        for index in range(1, doc.TablesOfContents.Count + 1):
            doc.TablesOfContents(index).Update()
        doc.Repaginate()
        doc.Save()
        doc.ExportAsFixedFormat(str(pdf_path), 17)  # 17 = PDF
        try:
            doc.Close(False)
        except Exception:
            pass
    finally:
        try:
            word.Quit()
        except Exception:
            pass
    return pdf_path


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DOCX
    print(f"PDF written to {export(target)}")
