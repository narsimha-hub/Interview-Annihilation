# backend/app/services/pdf.py

import fitz  # this is PyMuPDF

def extract_text_from_pdf(content: bytes) -> str:
    try:
        doc = fitz.open(stream=content, filetype="pdf")
        text = ""
        for page in doc:
            text += page.get_text("text") + "\n"
        doc.close()
        return text.strip()
    except Exception as e:
        raise ValueError(f"PDF extraction failed: {str(e)}")