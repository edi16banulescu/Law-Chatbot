import os
from google import genai
from google.genai.errors import APIError
from data_processor import load_and_chunk_data
# Importam functiile necesare din vector_db_manager

# --- Configuratii LLM ---
GENERATION_MODEL = 'gemini-2.5-flash'

def get_gemini_client():
    """Initializeaza clientul Gemini si verifica existenta API Key."""
    # Cheia API este citita direct din variabilele de mediu
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY nu este setată. Verificați variabila de mediu.")
    return genai.Client(api_key=api_key)

def generate_response_with_llm(retrieved_chunks: list[dict], user_query: str) -> str:
    """Generează răspunsul final, fundamentat pe context, folosind Gemini API."""
    
    client = get_gemini_client()
    
    # 1. Construim Contextul Sursă pentru LLM
    context_text = "\n".join([f"[{chunk['articol']}]: {chunk['text']}" for chunk in retrieved_chunks])
    citations = ", ".join([chunk['articol'] for chunk in retrieved_chunks])

    # 2. Prompt Engineering (Asigură acuratețea juridică)
    system_prompt = (
        "Ești un expert în Codul Rutier Român. Răspunde concis și în limba română la întrebarea utilizatorului "
        "bazându-te **doar** pe CONTEXTUL FURNIZAT. Asigură-te că citezi articolele legale relevante furnizate în context. "
        "Dacă contextul nu conține informația, răspunde politicos că nu poți oferi un răspuns fundamentat."
    )
    
    prompt = f"CONTEXT SURSĂ:\n---\n{context_text}\n---\n\nÎNTREBARE: {user_query}\n\nRĂSPUNS:"
    
    try:
        response = client.models.generate_content(
            model=GENERATION_MODEL,
            contents=[prompt],
            config={
                'system_instruction': system_prompt
            }
        )
        
        # Adaugam citarile in raspunsul generat
        final_answer = response.text
        if final_answer and citations:
             final_answer += f"\n\n(Fundamentat pe legislația: {citations})"
        
        return final_answer
    
    except APIError as e:
        return f"Eroare API la generarea răspunsului LLM: {e}"
    except Exception as e:
        return f"Eroare neașteptată: {e}"


# --- FUNCTIA PRINCIPALA (MAIN) ---

def run_rag_pipeline(user_input: str, k_results: int = 2):
    """Ruleaza intregul pipeline RAG."""
    print("=" * 60)
    print(f"Agent Conversațional (RAG) - Procesare întrebare: '{user_input}'")
    print("=" * 60)

    try:
        # 1. DATA ENGINEERING
        chunks_list, metadata_list, document_ids = load_and_chunk_data()

        if not chunks_list:
            print("\n[INFO] Nu s-au putut încărca datele din fișier. Oprire pipeline.")
            return

        # 2. INDEXARE (Creeaza sau actualizeaza ChromaDB)
        from vector_db_manager import create_or_update_db
        vector_db_collection = create_or_update_db(chunks_list, metadata_list, document_ids)
        
        # 3. RETRIEVAL (Regasirea Articolelor)
        from vector_db_manager import retrieve_chunks
        retrieved_chunks = retrieve_chunks(vector_db_collection, user_input, k=k_results)

        if not retrieved_chunks:
            print("\n[INFO] Nu am putut regăsi articole relevante din Codul Rutier. Vă rugăm să reformulați întrebarea.")
            return

        # 4. GENERATION (Generarea Raspunsului real de catre LLM)
        final_answer = generate_response_with_llm(retrieved_chunks, user_input)

        print("\n" + "#" * 60)
        print("RĂSPUNS FINAL FURNIZAT DE GEMINI (Fundamentat RAG)")
        print("#" * 60)
        print(final_answer)
        
    except ValueError as e:
        print(f"\n[FATAL ERROR]: Eroare de configurare sau cheie API: {e}")
    except Exception as e:
        print(f"\n[FATAL ERROR]: O eroare a întrerupt pipeline-ul: {e}")

if __name__ == "__main__":
    # USER QUERY
    query1 = "Care sunt sancțiunile pentru ITP expirat?"
    query2 = "Ce amenda iau daca merg fara placute de inmatriculare?"
    
    # Rulati prima interogare cu K mai mare pentru a include contextul necesar
    run_rag_pipeline(query1, k_results=3) 
    print("\n\n" + "-"*60 + "\n\n")
    # Rulati a doua interogare cu K normal
    # run_rag_pipeline(query2, k_results=2)