import os
import re
import json
from pathlib import Path
from pypdf import PdfReader

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
TEXTBOOK_DIR = BASE_DIR / "textbook"
DATA_DIR = BASE_DIR / "data"

CHAPTER_METADATA = [
    {
        "grade": 9,
        "chapter_num": 4,
        "chapter_title": "Describing Motion Around Us",
        "file_rel": "9th grade/iesc1dd/iesc104.pdf",
        "concept_ids": ["G09-MOT"],
        "misconception_ids": ["MISC-G09-MOT-01"]
    },
    {
        "grade": 9,
        "chapter_num": 6,
        "chapter_title": "How Forces Affect Motion",
        "file_rel": "9th grade/iesc1dd/iesc106.pdf",
        "concept_ids": ["G09-MOT", "G09-GRV"],
        "misconception_ids": ["MISC-G09-MOT-02", "MISC-G09-GRV-01"]
    },
    {
        "grade": 9,
        "chapter_num": 7,
        "chapter_title": "Work, Energy, and Simple Machines",
        "file_rel": "9th grade/iesc1dd/iesc107.pdf",
        "concept_ids": ["G09-MOT"],
        "misconception_ids": []
    },
    {
        "grade": 9,
        "chapter_num": 10,
        "chapter_title": "Sound Waves: Characteristics and Applications",
        "file_rel": "9th grade/iesc1dd/iesc110.pdf",
        "concept_ids": [],
        "misconception_ids": []
    },
    {
        "grade": 10,
        "chapter_num": 9,
        "chapter_title": "Light – Reflection and Refraction",
        "file_rel": "jesc1dd/jesc109.pdf",
        "concept_ids": ["G10-OPT"],
        "misconception_ids": ["MISC-G10-OPT-01", "MISC-G10-OPT-02", "MISC-G10-OPT-03"]
    },
    {
        "grade": 10,
        "chapter_num": 10,
        "chapter_title": "The Human Eye and the Colourful World",
        "file_rel": "jesc1dd/jesc110.pdf",
        "concept_ids": ["G10-OPT"],
        "misconception_ids": []
    },
    {
        "grade": 10,
        "chapter_num": 11,
        "chapter_title": "Electricity",
        "file_rel": "jesc1dd/jesc111.pdf",
        "concept_ids": ["G10-ELE"],
        "misconception_ids": ["MISC-G10-ELE-01", "MISC-G10-ELE-02", "MISC-G10-ELE-03"]
    },
    {
        "grade": 10,
        "chapter_num": 12,
        "chapter_title": "Magnetic Effects of Electric Current",
        "file_rel": "jesc1dd/jesc112.pdf",
        "concept_ids": ["G10-ELE"],
        "misconception_ids": []
    },
    {
        "grade": 11,
        "chapter_num": 3,
        "chapter_title": "Motion in a Plane",
        "file_rel": "11th grade part 1/keph1dd/keph103.pdf",
        "concept_ids": ["G11-KIN"],
        "misconception_ids": ["MISC-G11-KIN-01"]
    },
    {
        "grade": 11,
        "chapter_num": 4,
        "chapter_title": "Laws of Motion",
        "file_rel": "11th grade part 1/keph1dd/keph104.pdf",
        "concept_ids": ["G11-DYN"],
        "misconception_ids": ["MISC-G11-DYN-01"]
    },
    {
        "grade": 11,
        "chapter_num": 7,
        "chapter_title": "Gravitation",
        "file_rel": "11th grade part 1/keph1dd/keph107.pdf",
        "concept_ids": ["G09-GRV"],
        "misconception_ids": ["MISC-G09-GRV-01"]
    },
    {
        "grade": 12,
        "chapter_num": 2,
        "chapter_title": "Electrostatic Potential and Capacitance",
        "file_rel": "leph1dd/leph102.pdf",
        "concept_ids": ["G12-EST"],
        "misconception_ids": ["MISC-G12-EST-01"]
    },
    {
        "grade": 12,
        "chapter_num": 6,
        "chapter_title": "Electromagnetic Induction",
        "file_rel": "leph1dd/leph106.pdf",
        "concept_ids": ["G12-EMI"],
        "misconception_ids": ["MISC-G12-EMI-01"]
    }
]

def clean_page_text(raw_text: str) -> str:
    """Cleans noisy headers, repeated whitespace, and joins wrapped lines."""
    if not raw_text:
        return ""
    # Normalize unicode spaces & line breaks
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    # Remove recurring NCERT headers / footers like "Rationalised 2023-24" or chapter banner headers
    text = re.sub(r'Rationalised\s*202[0-9]-[0-9]+', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Reprint\s*202[0-9]-[0-9]+', '', text, flags=re.IGNORECASE)
    # Remove lines with isolated page numbers
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped.isdigit() and len(stripped) <= 3:
            continue
        lines.append(line)
    text = "\n".join(lines)
    # Collapse 3+ newlines to 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple horizontal spaces
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text.strip()

def chunk_text(text: str, max_words: int = 350, overlap_words: int = 50) -> list[str]:
    """Splits chapter text into logical semantic chunks with overlap."""
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_word_count = 0

    for para in paragraphs:
        p_clean = para.strip()
        if not p_clean:
            continue
        words = p_clean.split()
        p_len = len(words)

        if current_word_count + p_len <= max_words:
            current_chunk.append(p_clean)
            current_word_count += p_len
        else:
            if current_chunk:
                chunk_str = "\n\n".join(current_chunk)
                chunks.append(chunk_str)
                # Keep overlap from previous chunk
                overlap_text = " ".join(chunk_str.split()[-overlap_words:]) if overlap_words > 0 else ""
                current_chunk = [overlap_text, p_clean] if overlap_text else [p_clean]
                current_word_count = len(" ".join(current_chunk).split())
            else:
                chunks.append(p_clean)
                current_chunk = []
                current_word_count = 0

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))

    return [c.strip() for c in chunks if len(c.strip()) > 30]

def extract_all():
    print("=" * 60)
    print("Starting NCERT Physics Textbook Extraction Pipeline")
    print("=" * 60)

    corpus = {
        "metadata": {
            "source": "NCERT Physics Textbooks (Grades 9-12)",
            "total_chapters": len(CHAPTER_METADATA),
            "generated_at": "2026-10-04"
        },
        "chapters": [],
        "chunks": []
    }

    total_pages_extracted = 0
    total_words_extracted = 0
    total_chunks_created = 0

    for meta in CHAPTER_METADATA:
        pdf_path = TEXTBOOK_DIR / meta["file_rel"]
        if not pdf_path.exists():
            print(f"⚠️ Warning: File not found: {pdf_path}")
            continue

        print(f"\nProcessing Grade {meta['grade']} | Ch {meta['chapter_num']}: {meta['chapter_title']}...")
        reader = PdfReader(str(pdf_path))
        num_pages = len(reader.pages)
        total_pages_extracted += num_pages

        chapter_pages = []
        full_chapter_text_parts = []

        for p_idx, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            cleaned = clean_page_text(raw_text)
            if cleaned:
                chapter_pages.append({
                    "page_number": p_idx + 1,
                    "text": cleaned
                })
                full_chapter_text_parts.append(cleaned)

        full_chapter_text = "\n\n".join(full_chapter_text_parts)
        words = len(full_chapter_text.split())
        total_words_extracted += words

        # Create semantic chunks for RAG
        raw_chunks = chunk_text(full_chapter_text)
        chapter_chunks = []
        for c_idx, c_text in enumerate(raw_chunks):
            chunk_obj = {
                "chunk_id": f"G{meta['grade']:02d}-CH{meta['chapter_num']:02d}-{c_idx+1:03d}",
                "grade": meta["grade"],
                "chapter_num": meta["chapter_num"],
                "chapter_title": meta["chapter_title"],
                "file_rel": meta["file_rel"],
                "concept_ids": meta["concept_ids"],
                "misconception_ids": meta["misconception_ids"],
                "word_count": len(c_text.split()),
                "text": c_text
            }
            chapter_chunks.append(chunk_obj)
            corpus["chunks"].append(chunk_obj)

        total_chunks_created += len(chapter_chunks)

        corpus["chapters"].append({
            "grade": meta["grade"],
            "chapter_num": meta["chapter_num"],
            "chapter_title": meta["chapter_title"],
            "file_rel": meta["file_rel"],
            "pages_count": num_pages,
            "word_count": words,
            "chunks_count": len(chapter_chunks),
            "concept_ids": meta["concept_ids"],
            "misconception_ids": meta["misconception_ids"]
        })

        print(f"  [OK] {num_pages} pages, {words:,} words, {len(chapter_chunks)} chunks generated.")

    # Save corpus to JSON
    output_json = DATA_DIR / "physics_textbook_corpus.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(corpus, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("Extraction Summary:")
    print(f"  Total Chapters Processed: {len(corpus['chapters'])}")
    print(f"  Total Pages Extracted:    {total_pages_extracted}")
    print(f"  Total Words Extracted:    {total_words_extracted:,}")
    print(f"  Total Chunks Created:     {total_chunks_created:,}")
    print(f"  Saved to:                 {output_json}")
    print("=" * 60)

if __name__ == "__main__":
    extract_all()
