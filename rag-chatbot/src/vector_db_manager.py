import chromadb
from google import genai
from google.genai.errors import APIError
import os
import time 
import shutil # NOU: Necesara pentru stergerea directorului ChromaDB

# --- Configuratii ChromaDB si API ---
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "CodRutier_RAG"
EMBEDDING_MODEL = 'text-embedding-004' 
# Limita maxima de articole pe lot (impusa de API)
BATCH_SIZE = 100 

def get_gemini_client():
    """Initializeaza clientul Gemini si verifica existenta API Key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY nu este setată. Setați variabila de mediu.")
    return genai.Client(api_key=api_key)

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """
    Generează vectori de embedding reali folosind Gemini API, 
    împărțind lista de texte în loturi (batches) de maximum 100 de elemente.
    """
    client = get_gemini_client()
    all_embeddings = []
    
    # Impartirea in loturi de 100
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        print(f"  > Procesează lotul {i // BATCH_SIZE + 1} de {len(batch)} articole...")
        
        try:
            response = client.models.embed_content(
                model=EMBEDDING_MODEL,
                contents=batch, 
                config={'task_type': "RETRIEVAL_DOCUMENT"} 
            )
            
            # CORECȚIE CRITICĂ: Extragem lista de valori (float) din fiecare obiect ContentEmbedding
            for embedding_obj in response.embeddings:
                all_embeddings.append(embedding_obj.values)
            
            # Pauza de 1 secunda intre loturi pentru a preveni throttling (optimizare)
            time.sleep(1) 
            
        except APIError as e:
            print(f"Eroare API la generarea embedding-urilor (Lotul {i // BATCH_SIZE + 1}): {e}")
            return []
        except Exception as e:
            # Tipareste eroarea completa pentru debugging
            import traceback
            print(f"Eroare neașteptată la embedding: {e}")
            print(traceback.format_exc())
            return []

    return all_embeddings

def create_or_update_db(chunks_list: list[str], metadata_list: list[dict], document_ids: list[str]):
    """Creează clientul ChromaDB și adaugă documentele."""
    
    # 1. Client ChromaDB in modul local/persistenta
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # 2. Populare DB
    if collection.count() < 1:
        print("[DB Manager] Generare embedding-uri reale... Așteptați...")
        vectors = generate_embeddings(chunks_list)
        
        if not vectors:
             # Eroarea critica este acum aruncata in generate_embeddings
             raise Exception("Nu s-au putut genera vectorii din cauza erorii API.")
        
        # Inserare in ChromaDB
        collection.add(
            embeddings=vectors,
            documents=chunks_list,
            metadatas=metadata_list,
            ids=document_ids
        )
        print(f"[DB Manager] Baza de date ChromaDB '{COLLECTION_NAME}' populată cu {collection.count()} articole.")
    else:
        print(f"[DB Manager] Baza de date ChromaDB a fost deja populată ({collection.count()} articole).")
        
    return collection

def retrieve_chunks(collection, user_query: str, k: int = 2) -> list[dict]:
    """Interogheaza ChromaDB pentru a gasi cele mai relevante articole."""
    client = get_gemini_client()
    
    # 1. Vectorizeaza Intrebarea (Query Vector)
    query_vector_response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=[user_query], 
        config={'task_type': "RETRIEVAL_QUERY"} 
    )
    # Accesarea vectorului se face cu .embeddings[0].values
    query_vector = query_vector_response.embeddings[0].values
    
    # 2. Cauta cei mai apropiati k vectori
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=k,
        include=['documents', 'metadatas']
    )
    
    retrieved_chunks = []
    if results['documents'] and results['metadatas']:
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            retrieved_chunks.append({
                "text": doc,
                "articol": f"{meta['sursa']} - Articolul {meta['articol']}"
            })
            
    return retrieved_chunks

def clear_db():
    """Șterge directorul ChromaDB pentru a forța o nouă indexare completă."""
    if os.path.exists(CHROMA_PATH):
        try:
            shutil.rmtree(CHROMA_PATH)
            print(f"[DB Manager] Directorul ChromaDB '{CHROMA_PATH}' a fost șters cu succes.")
        except OSError as e:
            print(f"Eroare la ștergerea directorului ChromaDB: {e}")
    else:
        print(f"[DB Manager] Directorul ChromaDB '{CHROMA_PATH}' nu există.")

if __name__ == '__main__':
    # Acest test necesita ca data_processor sa fie importabil
    # Deoarece este un modul, presupunem ca este importat dintr-un script principal
    pass