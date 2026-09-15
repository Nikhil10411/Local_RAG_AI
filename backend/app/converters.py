import io
import os
import shutil
import fitz
import tempfile
from pathlib import Path
from typing import Any, Generator, cast

from PIL import Image
import pytesseract
from docx import Document
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.shapes.autoshape import Shape
from openpyxl import Workbook, load_workbook
from openpyxl.worksheet.worksheet import Worksheet
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from pypdf import PdfReader
from pdf2docx import Converter

from app.config import settings

# ----------------- TESSERACT AUTO-DETECTION -----------------

def resolve_tesseract_binary() -> None:
    """Locates and verifies Tesseract OCR executable path on host system."""
    which_path = shutil.which("tesseract")
    if which_path:
        pytesseract.pytesseract.tesseract_cmd = which_path
        return

    standard_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expandvars(r"%LOCALAPPDATA%\Programs\Tesseract-OCR\tesseract.exe"),
        os.path.expandvars(r"%USERPROFILE%\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
    ]
    for path in standard_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            return

resolve_tesseract_binary()

EXPORTS_DIR = Path(settings.DATABASE_PATH).parent / "exports"
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ----------------- INGESTION EXTRACTION (OCR & PARSING) -----------------

def extract_text_from_image(image_bytes: bytes) -> str:
    """Extracts raw text from image formats using local Tesseract OCR engine."""
    cmd = pytesseract.pytesseract.tesseract_cmd
    if not os.path.exists(cmd) and not shutil.which(cmd):
        raise RuntimeError(
            "Tesseract OCR executable not detected in PATH or standard directories."
        )
    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image)

def extract_content_from_file(filename: str, content: bytes) -> str:
    """Extracts plain textual representation across common file types."""
    lower = filename.lower()
    text = ""

    if lower.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(content))
        for page in reader.pages:
            page_str = page.extract_text()
            if page_str:
                text += page_str + "\n"

    elif lower.endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
        text = extract_text_from_image(content)

    elif lower.endswith(".docx"):
        doc = Document(io.BytesIO(content))
        text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])

    elif lower.endswith(".pptx"):
        prs = Presentation(io.BytesIO(content))
        for slide in prs.slides:
            for raw_shape in slide.shapes:
                shape: Any = raw_shape
                if getattr(shape, "has_text_frame", False):
                    tf = getattr(shape, "text_frame", None)
                    if tf and getattr(tf, "text", None):
                        text += str(tf.text) + "\n"

    elif lower.endswith((".xlsx", ".xls")):
        wb = load_workbook(io.BytesIO(content), data_only=True)
        for sheet in wb.sheetnames:
            ws = cast(Worksheet, wb[sheet])
            text += f"\n--- Sheet: {sheet} ---\n"
            for row in ws.iter_rows(values_only=True):
                row_vals = [str(v) for v in row if v is not None]
                if row_vals:
                    text += ", ".join(row_vals) + "\n"

    elif lower.endswith(".csv"):
        text = content.decode("utf-8", errors="ignore")

    elif lower.endswith(".txt"):
        text = content.decode("utf-8", errors="ignore")

    return text.strip()

# ------------- EXACT LAYOUT CONVERSIONS (PDF <-> DOCX) -------------

def convert_pdf_to_docx_exact(file_bytes: bytes, base_name: str) -> str:
    """
    Converts PDF directly to DOCX preserving coordinates, font weights,
    horizontal rule lines, and right-aligned dates without creating broken tables.
    """
    # 1. Compatibility patch for PyMuPDF 1.24+
    if not hasattr(fitz.Rect, "get_area"):
        setattr(fitz.Rect, "get_area", lambda self: abs(self.width * self.height))

    target_path = EXPORTS_DIR / f"{base_name}.docx"

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_pdf:
        temp_pdf.write(file_bytes)
        temp_pdf_path = temp_pdf.name

    cv = None
    try:
        # First check: If Microsoft Word is locally installed on Windows,
        # its native rendering engine produces 100% vector-identical conversions.
        try:
            import win32com.client
            import pythoncom
            pythoncom.CoInitialize()
            word = win32com.client.DispatchEx("Word.Application")
            word.Visible = False
            word.DisplayAlerts = False
            
            # Word's native Open handles PDF-to-DOCX with native fidelity
            doc = word.Documents.Open(os.path.abspath(temp_pdf_path), False, False, False)
            doc.SaveAs2(os.path.abspath(str(target_path)), FileFormat=16) # wdFormatDocumentDefault
            doc.Close(False)
            word.Quit()
            return target_path.name
        except Exception:
            pass # Fall back to tuned pdf2docx

        # Tuned parameters: stop aggressive table generation for clean paragraph tabs
        cv = Converter(temp_pdf_path)
        conversion_settings = {
            "parse_lattice_table": False,   # Prevents resume lines from turning into broken tables
            "parse_stream_table": False,    # Keeps right-aligned dates on tab stops
            "line_overlap_threshold": 0.3,
            "connected_border_tolerance": 0.5,
            "min_border_clearance": 1.0,
            "split_multiline_text": False,
        }
        cv.convert(str(target_path), start=0, **conversion_settings)

    finally:
        if cv is not None:
            try:
                cv.close()
            except Exception:
                pass
        if os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except PermissionError:
                pass

    return target_path.name

def convert_docx_to_pdf_exact(file_bytes: bytes, base_name: str) -> str:
    """
    Converts DOCX to PDF using native OS Word engine (docx2pdf) to preserve exact styling.
    Falls back to ReportLab text generation if Microsoft Word COM interface is unavailable.
    """
    target_path = EXPORTS_DIR / f"{base_name}.pdf"
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as temp_docx:
        temp_docx.write(file_bytes)
        temp_docx_path = temp_docx.name

    conversion_success = False
    try:
        import docx2pdf
        docx2pdf.convert(temp_docx_path, str(target_path))
        conversion_success = True
    except Exception:
        conversion_success = False

    if not conversion_success or not target_path.exists():
        doc = Document(temp_docx_path)
        full_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
        export_to_pdf(full_text, base_name)

    if os.path.exists(temp_docx_path):
        try:
            os.remove(temp_docx_path)
        except PermissionError:
            pass

    return target_path.name

# ---------------- LOGICAL TEXT CONVERSION GENERATORS ----------------

def export_to_docx(content: str, filename: str) -> str:
    """Generates standard structured DOCX document from input text."""
    filepath = EXPORTS_DIR / f"{filename}.docx"
    doc = Document()
    doc.add_heading("AI Generated / Converted Document", 0)
    for paragraph in content.split("\n\n"):
        if paragraph.strip():
            doc.add_paragraph(paragraph.strip())
    doc.save(str(filepath))
    return filepath.name

def export_to_pptx(content: str, filename: str) -> str:
    """Transforms markdown-style section headings into presentation slides."""
    prs = Presentation()
    blank_slide_layout = prs.slide_layouts[6]

    sections = content.split("## ")
    for sec in sections:
        if not sec.strip():
            continue
        slide = prs.slides.add_slide(blank_slide_layout)
        lines = sec.strip().split("\n")
        title_text = lines[0]
        body_text = "\n".join(lines[1:])

        tx_box_title = cast(
            Shape,
            slide.shapes.add_textbox(Inches(0.8), Inches(0.5), Inches(8.4), Inches(1)),
        )
        tf = tx_box_title.text_frame
        p = tf.paragraphs[0]
        p.text = title_text
        if p.font:
            p.font.bold = True
            p.font.size = Pt(28)

        tx_box_body = cast(
            Shape,
            slide.shapes.add_textbox(Inches(0.8), Inches(1.8), Inches(8.4), Inches(4.5)),
        )
        tf_body = tx_box_body.text_frame
        tf_body.text = body_text

    filepath = EXPORTS_DIR / f"{filename}.pptx"
    prs.save(str(filepath))
    return filepath.name

def export_to_excel(content: str, filename: str) -> str:
    """Parses delimited rows into tabular spreadsheet cells."""
    filepath = EXPORTS_DIR / f"{filename}.xlsx"
    wb = Workbook()
    ws = cast(Worksheet, wb.active)
    ws.title = "Extracted Data"

    lines = content.strip().split("\n")
    for row_idx, line in enumerate(lines, start=1):
        cols = [c.strip() for c in (line.split(",") if "," in line else line.split("\t"))]
        for col_idx, val in enumerate(cols, start=1):
            ws.cell(row=row_idx, column=col_idx, value=val)

    wb.save(str(filepath))
    return filepath.name

def export_to_csv(content: str, filename: str) -> str:
    """Exports structured plain text directly to CSV format."""
    filepath = EXPORTS_DIR / f"{filename}.csv"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    return filepath.name

def export_to_pdf(content: str, filename: str) -> str:
    """Generates standard letter-sized paginated PDF using ReportLab canvas."""
    filepath = EXPORTS_DIR / f"{filename}.pdf"
    c = canvas.Canvas(str(filepath), pagesize=letter)
    _, height = letter
    y = height - 50

    for line in content.split("\n"):
        if y < 50:
            c.showPage()
            y = height - 50
        c.drawString(50, y, line[:90])
        y -= 15

    c.save()
    return filepath.name

# ----------------- DOWNLOAD & STREAMING UTILITIES -----------------

def get_export_file_path(filename: str) -> Path:
    """Validates and returns absolute path within the configured export directory."""
    target_path = (EXPORTS_DIR / filename).resolve()
    if not str(target_path).startswith(str(EXPORTS_DIR.resolve())):
        raise PermissionError("Access denied: Invalid file path target.")
    if not target_path.exists() or not target_path.is_file():
        raise FileNotFoundError(f"Requested converted file '{filename}' does not exist.")
    return target_path

def get_export_file_bytes(filename: str) -> bytes:
    """Reads binary content of exported document for forced download transmission."""
    target_path = get_export_file_path(filename)
    with open(target_path, "rb") as f:
        return f.read()

def get_export_file_stream(filename: str, chunk_size: int = 65536) -> Generator[bytes, None, None]:
    """Streams file bytes in fixed memory buffers to support downloading large documents."""
    target_path = get_export_file_path(filename)
    with open(target_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk