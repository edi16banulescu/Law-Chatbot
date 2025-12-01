import uuid
import re
import os

# Numele fisierului incarcat de tine
CORPUS_FILE = "codul_rutier.txt"

def load_and_chunk_data():
    """
    Încarcă textul legal, îl segmentează pe Articole, iar Articolele lungi (liste)
    le sub-segmentează pentru a crește precizia RAG.
    """
    try:
        if not os.path.exists(CORPUS_FILE):
             print(f"Eroare: Fișierul corpus '{CORPUS_FILE}' nu a fost găsit.")
             return [], [], []

        with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except Exception as e:
        print(f"Eroare la citirea fișierului: {e}")
        return [], [], []

    # --- 1. Curățare Inițială ---
    # Eliminăm titlurile care nu sunt relevante pentru vectorizare
    cleaned_text = re.sub(r'(^#+.*$|^\s*CAPITOLUL\s*[^A-Z]*[^\n]*|^SECTIUNEA\s*[^A-Z]*[^\n]*)', '', full_text, flags=re.MULTILINE)
    
    # --- 2. Segmentare Primară (Pe Articole) ---
    # Adaugăm un marker temporar înainte de fiecare Articol
    # Regex-ul acceptă "Art. 1." sau "Art. 11.1"
    text_with_markers = re.sub(r'((?:Articolul|Art\.)\s*\d+(?:\.\d+)?[\.\s]*)', r'|--CHUNK--|\g<0>', cleaned_text, flags=re.IGNORECASE | re.DOTALL)
    
    # Gestionăm schimbarea sursei (OUG/HG) folosind separatorii tai din fisier
    text_with_markers = text_with_markers.replace('----- START OUG 195/2002', '|--SOURCE_OUG--|')
    text_with_markers = text_with_markers.replace('----- START HG 1391/2006', '|--SOURCE_HG--|')
    text_with_markers = text_with_markers.replace('----- START CODUL PENAL', '|--SOURCE_PENAL--|')

    raw_chunks = [chunk.strip() for chunk in text_with_markers.split('|--CHUNK--|') if chunk.strip()]
    
    # --- 3. Procesare Fină și Sub-segmentare ---
    final_chunks = []
    metadata_list = []
    document_ids = []
    
    current_source = "OUG 195/2002" 
    
    for chunk in raw_chunks:
        # Actualizare sursă
        if "|--SOURCE_HG--|" in chunk:
            current_source = "HG 1391/2006"
            chunk = chunk.replace("|--SOURCE_HG--|", "").strip()
        elif "|--SOURCE_OUG--|" in chunk:
            current_source = "OUG 195/2002"
            chunk = chunk.replace("|--SOURCE_OUG--|", "").strip()
        elif "|--SOURCE_PENAL--|" in chunk: # NOU
            current_source = "Codul Penal"
            chunk = chunk.replace("|--SOURCE_PENAL--|", "").strip()
            
        # Identificare Număr Articol
        art_match = re.search(r'(?:Art\.|Articolul)\s*(\d+(?:\.\d+)?)', chunk, flags=re.IGNORECASE)
        art_num = art_match.group(1) if art_match else "N/A"
        
        # --- LOGICA DE SUB-SEGMENTARE ---
        # Dacă articolul este lung (> 600 caractere) și conține o listă numerotată (1., 2. etc)
        if len(chunk) > 600 and re.search(r'\n\s*\d+\.', chunk):
            # Spargem după modelul "linie nouă + număr + punct" (ex: "\n1.")
            header_match = re.split(r'(\n\s*\d+\.)', chunk, maxsplit=1)
            
            if len(header_match) >= 3:
                header_text = header_match[0] # Textul de inceput ("Constituie contravenții...")
                body_text = header_match[1] + header_match[2] # Restul listei
                
                # Împărțim lista în puncte individuale
                sub_points = re.split(r'(\n\s*\d+\.)', body_text)
                
                for i in range(1, len(sub_points), 2):
                    if i+1 < len(sub_points):
                        point_marker = sub_points[i].strip() # "1."
                        point_content = sub_points[i+1].strip()
                        
                        # Creăm un sub-chunk care păstrează esențialul din header (Sancțiunea/Clasa)
                        # Luăm primele 150 caractere din header care de obicei conțin "Clasa a IV-a de sancțiuni"
                        full_sub_chunk = f"{current_source} - Art. {art_num} ({point_marker}) {header_text[:150]}... : {point_content}"
                        
                        final_chunks.append(full_sub_chunk)
                        metadata_list.append({"sursa": current_source, "articol": f"{art_num} (pct {point_marker})"})
                        document_ids.append(str(uuid.uuid4()))
            else:
                final_chunks.append(chunk)
                metadata_list.append({"sursa": current_source, "articol": art_num})
                document_ids.append(str(uuid.uuid4()))
        else:
            final_chunks.append(chunk)
            metadata_list.append({"sursa": current_source, "articol": art_num})
            document_ids.append(str(uuid.uuid4()))
        
    print(f"[PAS 1] Segmentare finalizată. Total segmente: {len(final_chunks)}")
    return final_chunks, metadata_list, document_ids

if __name__ == '__main__':
    load_and_chunk_data()