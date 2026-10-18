import os
from pypdf import PdfReader

PDF_PATH = "data/james.pdf"
CHUNK_SIZE = 500  # characters per chunk
CHUNK_OVERLAP = 50

def extract_text_from_pdf(pdf_path):
    print(f"Reading {pdf_path}...")
    reader = PdfReader(pdf_path)
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += f"\n[Page {i+1}]\n" + text
    return full_text

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

if __name__ == "__main__":
    if not os.path.exists(PDF_PATH):
        print(f"Error: {PDF_PATH} not found!")
    else:
        text = extract_text_from_pdf(PDF_PATH)
        chunks = chunk_text(text)
        print(f"Successfully extracted and split {len(chunks)} chunks from {PDF_PATH}.")
        print("\nSample Chunk 1:")
        print("-" * 40)
        print(chunks[0] if chunks else "No text found.")
        print("-" * 40)
