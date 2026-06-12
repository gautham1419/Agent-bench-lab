import re
import zlib
from pathlib import Path

def extract_pdf_text(pdf_path):
    pdf_path = Path(pdf_path)
    content = pdf_path.read_bytes()
    
    # Find all streams
    streams = re.findall(b'stream\r?\n(.*?)\r?\nendstream', content, re.DOTALL)
    
    text_pieces = []
    
    for stream in streams:
        try:
            decompressed = zlib.decompress(stream)
            # Find Tj or TJ text blocks
            # e.g., (text) Tj or [(t1) 10 (t2)] TJ
            matches = re.findall(b'\((.*?)\)\s*Tj', decompressed)
            for m in matches:
                try:
                    text_pieces.append(m.decode('utf-8', errors='ignore'))
                except Exception:
                    pass
            
            matches_tj = re.findall(b'\[(.*?)\]\s*TJ', decompressed)
            for m in matches_tj:
                # Find all strings inside brackets: (str1) (str2) etc
                strs = re.findall(b'\((.*?)\)', m)
                for s in strs:
                    try:
                        text_pieces.append(s.decode('utf-8', errors='ignore'))
                    except Exception:
                        pass
        except Exception:
            # Not a zlib stream or other issue
            pass
            
    # Also check uncompressed text in the file
    matches = re.findall(b'\((.*?)\)\s*Tj', content)
    for m in matches:
        try:
            text_pieces.append(m.decode('utf-8', errors='ignore'))
        except Exception:
            pass
            
    return text_pieces

def main():
    docs_dir = Path("docs")
    for f in sorted(docs_dir.glob("fig_rq*.pdf")):
        text = extract_pdf_text(f)
        # Unique and filtered text to see labels and numbers
        unique_text = []
        seen = set()
        for t in text:
            t_clean = t.strip()
            if t_clean and t_clean not in seen:
                seen.add(t_clean)
                unique_text.append(t_clean)
        print(f"=== File: {f.name} ===")
        print(f"Extracted labels/text ({len(unique_text)}): {unique_text[:40]}")
        print()

if __name__ == "__main__":
    main()
