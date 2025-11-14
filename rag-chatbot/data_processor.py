import uuid
import re
import os

# Numele fisierului care contine legislatia completa
CORPUS_FILE = "codul_rutier.txt"

def load_and_chunk_data():
    """
    Încarcă textul legal din fișier, îl segmentează logic pe Articole și generează metadate.
    Aceasta este Faza I: Data Engineering.
    """
    try:
        # 1. Incarca textul din fisierul extern
        with open(CORPUS_FILE, 'r', encoding='utf-8') as f:
            full_text = f.read()
    except FileNotFoundError:
        print(f"Eroare: Fișierul corpus '{CORPUS_FILE}' nu a fost găsit. Creați-l mai întâi și umpleți-l cu textul legii.")
        return [], [], []

    # 2. Segmentarea Logică (Pe bază de "Art. X.")
    # Folosim o expresie regulată pentru a despărți textul la începutul fiecărui Articol.
    # Pattern-ul: cauta "Art." urmat de spatiu si un numar (fara a include Art. in segmentul anterior)
    
    # Adaugam un marker temporar inaintea fiecarui Articol
    # NOTE: Folosesc un pattern mai permisiv pentru "Art." sau "Articolul"
    text_with_markers = re.sub(r'(Art\.\s*\d+[\.\s]*)', r'|--CHUNK--|\g<0>', full_text, flags=re.IGNORECASE | re.DOTALL)
    
    # Impartim la marker si eliminam primul element (care e titlul capitolului)
    chunks_list = [chunk.strip() for chunk in text_with_markers.split('|--CHUNK--|') if chunk.strip()]
    
    # 3. Generarea Metadatelor
    document_ids = [str(uuid.uuid4()) for _ in chunks_list]
    metadata_list = []
    current_source = "OUG 195/2002" # Presupunem ca incepe cu OUG
    
    for chunk in chunks_list:
        meta = {"sursa": "Nespecificata", "articol": "N/A"}
        
        # Actualizam sursa daca gasim titluri de lege (presupunand ca sunt marcate la inceput de fisier)
        if "OUG 195/2002" in chunk:
            current_source = "OUG 195/2002"
        elif "HG 1391/2006" in chunk:
            current_source = "HG 1391/2006"
        
        meta["sursa"] = current_source
        
        # Extrage numarul Articolului (cauta "Art." urmat de numar)
        match = re.search(r'(Art\.|Articolul)\s*(\d+)', chunk, flags=re.IGNORECASE)
        if match:
            meta["articol"] = match.group(2)
        
        metadata_list.append(meta)
        
    print(f"[PAS 1] Segmentare finalizată. Număr de articole (chunks): {len(chunks_list)}")
    return chunks_list, metadata_list, document_ids

if __name__ == '__main__':
    chunks, meta, _ = load_and_chunk_data()
    print(f"Număr de segmente (Articole) create: {len(chunks)}")
    if chunks:
        print("-" * 50)
        print(f"Exemplu segment 1 ({meta[0]['sursa']} Art. {meta[0]['articol']}): {chunks[0][:80]}...")