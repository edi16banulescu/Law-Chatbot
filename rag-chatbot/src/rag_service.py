import sys
from google import genai
from src.data_processor import load_and_chunk_data
from src.vector_db_manager import create_or_update_db, retrieve_chunks, clear_db
import ollama
from dotenv import load_dotenv

load_dotenv()

# --- Configuratii LLM Local ---
GENERATION_MODEL = 'gemma2:9b' # Modelul de chat

def optimize_query_with_llm(user_query: str) -> str:
    """
    Reformulează întrebarea folosind LLM local.
    """
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
        response = ollama.chat(model=GENERATION_MODEL, messages=[
            {'role': 'user', 'content': optimization_prompt}
        ])
        return response['message']['content'].strip()
    except Exception as e:
        print(f"[WARN] Ollama error: {e}")
        return user_query

def generate_response_with_llm(retrieved_chunks: list[dict], user_query: str) -> str:
    """
    Generează răspunsul final folosind Llama 3.2 local.
    """
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
    
    user_message = f"CONTEXT LEGISLATIV:\n---\n{context_text}\n---\n\nÎNTREBARE UTILIZATOR: {user_query}\n\nRĂSPUNS:"
    
    try:
        response = ollama.chat(model=GENERATION_MODEL, messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message}
        ])
        
        final_answer = response['message']['content']
        if final_answer:
             final_answer += f"\n\n(Surse: {citations_str})"
        return final_answer
    except Exception as e:
        return f"Eroare generare locală: {e}"

# --- FUNCTII MODUL INTERACTIV ---

def initialize_rag_system():
    print("\n" + "="*60)
    print(" 🦙  INITIALIZARE AGENT RUTIER (LOCAL - OLLAMA)... ")
    print("="*60)
    
    reindex = input("Dorești re-indexarea completă a bazei de date? (da/nu) [nu]: ").lower().strip()
    
    if reindex in ['da', 'y', 'yes']:
        print("... Ștergerea bazei de date vechi ...")
        try:
            from vector_db_manager import clear_db
            clear_db()
        except ImportError:
            pass

    print("... Încărcare date ...")
    chunks_list, metadata_list, document_ids = load_and_chunk_data()
    
    if not chunks_list:
        print("[EROARE] Nu s-au putut încărca datele.")
        sys.exit(1)

    print(f"... Conectare la ChromaDB si Vectorizare Locală...")
    collection = create_or_update_db(chunks_list, metadata_list, document_ids)
    
    print("\n✅ Sistem local pregătit!")
    return collection

def process_query(collection, user_input, k_results=10): # K=10 e suficient pt Llama
    print(" 🦙 (Gândesc...) Reformulez întrebarea...")
    enhanced_query = optimize_query_with_llm(user_input)
    
    print(" 🔍 (Caut...) Analizez legislația...")
    retrieved_chunks = retrieve_chunks(collection, enhanced_query, k=k_results)
    
    if not retrieved_chunks:
        return "Nu am găsit articole relevante."

    print(" ✍️  (Scriu...) Generez răspunsul...")
    answer = generate_response_with_llm(retrieved_chunks, user_input)
    return answer

def start_interactive_chat():
    try:
        collection = initialize_rag_system()
    except Exception as e:
        print(f"[FATAL] {e}")
        return

    print("\n" + "!" * 60)
    print(" MOD LOCAL ACTIVAT (Llama 3.2 + Nomic Embed)")
    print("!" * 60)
    
    while True:
        try:
            user_input = input("\nTu: ").strip()
            if user_input.lower() in ['exit', 'q']: break
            print("\nAgent:", process_query(collection, user_input))
            print("-" * 60)
        except KeyboardInterrupt: break

if __name__ == "__main__":
    start_interactive_chat()