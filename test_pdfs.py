"""
Test PDF text extraction – checks if PDFs contain readable text.
Run this BEFORE running the full RAG pipeline.
"""

from pathlib import Path
from pypdf import PdfReader

def test_pdf_readability(pdf_path):
    """Attempt to extract text from a PDF. Returns (success, first_200_chars)."""
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text
        if text.strip():
            return True, text[:200].replace('\n', ' ')
        else:
            return False, "No extractable text found (likely scanned image PDF)."
    except Exception as e:
        return False, f"Error: {str(e)}"

def main():
    data_folder = Path("data")
    if not data_folder.exists():
        print("❌ 'data' folder not found. Create it and add your PDFs.")
        return
    
    pdf_files = list(data_folder.glob("*.pdf"))
    if not pdf_files:
        print("❌ No PDF files found in 'data/' folder.")
        return
    
    print(f"🔍 Testing {len(pdf_files)} PDF(s) for text readability...\n")
    for pdf_file in pdf_files:
        success, info = test_pdf_readability(pdf_file)
        status = "✅ READABLE" if success else "❌ NOT READABLE"
        print(f"{status} - {pdf_file.name}")
        print(f"   Sample: {info}\n")

if __name__ == "__main__":
    main()