import os
import sys
from google import genai
from google.genai.errors import APIError
from data_processor import load_and_chunk_data

# --- Configuratii LLM ---
GENERATION_MODEL = 'gemini-2.5-flash'

def get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY nu este setată.")
    return genai.Client(api_key=api_key)

def optimize_query_with_llm(user_query: str) -> str:
    """
    Reformulează întrebarea utilizatorului în limbaj juridic folosind LLM.
    """
    client = get_gemini_client()
    
    optimization_prompt = (
        "Ești un asistent specializat în legislația rutieră din România. "
        "Sarcina ta este să REFORMULEZI întrebarea utilizatorului pentru a fi găsită ușor în OUG 195/2002, HG 1391/2006 și Codul Penal (Art. 334-338).\n"
        "Reguli:\n"
        "1. Înlocuiește termenii colocviali cu termeni legali (ex: 'carnet' -> 'permis de conducere', 'băut' -> 'sub influența alcoolului', 'dosar penal' -> 'infracțiune').\n"
        "2. Păstrează sensul întrebării, dar fă-o să sune ca un text de lege.\n"
        "3. Returnează DOAR întrebarea reformulată.\n\n"
        f"Întrebare Utilizator: {user_query}\n"
        "Întrebare Optimizată:"
    )
    
    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[optimization_prompt]
        )
        return response.text.strip()
    except Exception as e:
        return user_query

def generate_response_with_llm(retrieved_chunks: list[dict], user_query: str) -> str:
    """
    Generează răspunsul final bazat pe context.
    """
    client = get_gemini_client()
    
    context_text = "\n".join([f"[{chunk['articol']}]: {chunk['text']}" for chunk in retrieved_chunks])
    citations = sorted(list(set([chunk['articol'] for chunk in retrieved_chunks])))
    citations_str = ", ".join(citations[:5]) 

    system_prompt = (
        "Ești un asistent juridic expert în Codul Rutier Român și Infracțiuni Rutiere (OUG 195, HG 1391, Cod Penal). "
        "Răspunde la întrebarea utilizatorului bazându-te **DOAR** pe CONTEXTUL FURNIZAT.\n"
        "- Dacă fapta este o CONTRAVENȚIE (amendă), specifică clasa de sancțiuni sau punctele.\n"
        "- Dacă fapta este o INFRACȚIUNE (închisoare), specifică pedeapsa conform Codului Penal din context.\n"
        "- Citează articolul de lege relevant.\n"
        "- Dacă informația nu există în context, spune clar 'Nu am găsit informația în articolele regăsite'."
    )
    
    prompt = f"CONTEXT LEGISLATIV:\n---\n{context_text}\n---\n\nÎNTREBARE UTILIZATOR: {user_query}\n\nRĂSPUNS:"
    
    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[prompt],
            config={'system_instruction': system_prompt}
        )
        final_answer = response.text
        if final_answer:
             final_answer += f"\n\n(Surse: {citations_str})"
        return final_answer
    except Exception as e:
        return f"Eroare generare: {e}"

# --- FUNCTII MODUL INTERACTIV ---

def initialize_rag_system():
    print("\n" + "="*60)
    print(" 🚗  INITIALIZARE AGENT RUTIER... ")
    print("="*60)
    
    reindex = input("Dorești re-indexarea completă a bazei de date? (da/nu) [nu]: ").lower().strip()
    
    if reindex in ['da', 'y', 'yes']:
        print("... Ștergerea bazei de date vechi ...")
        try:
            from vector_db_manager import clear_db
            clear_db()
        except ImportError:
            pass

    print("... Încărcare și verificare Bază de Cunoștințe ...")
    chunks_list, metadata_list, document_ids = load_and_chunk_data()
    
    if not chunks_list:
        print("[EROARE] Nu s-au putut încărca datele. Verifică 'codul_rutier.txt'.")
        sys.exit(1)

    print(f"... Conectare la ChromaDB ({len(chunks_list)} segmente)...")
    from vector_db_manager import create_or_update_db
    collection = create_or_update_db(chunks_list, metadata_list, document_ids)
    
    print("\n✅ Sistem pregătit!")
    return collection

def process_query(collection, user_input, k_results=15):
    # 1. Optimizare
    print(" 🤖 (Gândesc...) Reformulez întrebarea...")
    enhanced_query = optimize_query_with_llm(user_input)
    
    # 2. Retrieval
    print(" 🔍 (Caut...) Analizez legislația...")
    from vector_db_manager import retrieve_chunks
    retrieved_chunks = retrieve_chunks(collection, enhanced_query, k=k_results)
    
    if not retrieved_chunks:
        return "Nu am găsit articole relevante în baza de date."

    # 3. Generation
    print(" ✍️  (Scriu...) Generez răspunsul...")
    answer = generate_response_with_llm(retrieved_chunks, user_input)
    return answer

def start_interactive_chat():
    try:
        vector_db_collection = initialize_rag_system()
    except Exception as e:
        print(f"[FATAL] Eroare la inițializare: {e}")
        return

    # --- DISCLAIMER LEGAL ---
    print("\n" + "!" * 60)
    print(" AVERTISMENT LEGAL:")
    print(" Acest asistent este un proiect academic demonstrativ.")
    print(" Informațiile oferite nu reprezintă consultanță juridică oficială.")
    print(" Verificați întotdeauna legea în vigoare sau consultați un avocat.")
    print("!" * 60)
    
    print("\nScrie 'exit' sau 'q' pentru a închide.")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nTu: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("La revedere! Drum bun! 🚗")
                break
            
            response = process_query(vector_db_collection, user_input)
            
            print("\nAgent Rutier:")
            print(response)
            print("-" * 60)
            
        except KeyboardInterrupt:
            print("\nLa revedere!")
            break
        except Exception as e:
            print(f"\n[Eroare]: {e}")

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("EROARE: Variabila de mediu GEMINI_API_KEY nu este setată!")
    else:
        start_interactive_chat()