import chromadb
from google import genai
from google.genai.errors import APIError
import os

# --- Configuratii ChromaDB si API ---
CHROMA_PATH = "chroma_db"
COLLECTION_NAME = "CodRutier_RAG"
EMBEDDING_MODEL = 'text-embedding-004' # Modelul Google pentru embedding

def get_gemini_client():
    """Initializeaza clientul Gemini si verifica existenta API Key."""
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY nu este setată în variabilele de mediu.")
    return genai.Client(api_key=api_key)

def generate_embeddings(texts: list[str]) -> list[list[float]]:
    """Generează vectori de embedding reali folosind Gemini API."""
    client = get_gemini_client()
    try:
        response = client.models.embed_content(
            model=EMBEDDING_MODEL,
            content=texts,
            task_type="RETRIEVAL_DOCUMENT"
        )
        # API-ul returneaza un obiect, extragem listele de embeddings
        return response['embedding']
    except APIError as e:
        print(f"Eroare API la generarea embedding-urilor: {e}")
        return []
    except Exception as e:
        print(f"Eroare neașteptată la embedding: {e}")
        return []

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
        content=[user_query],
        task_type="RETRIEVAL_QUERY" # Task type diferit pentru query
    )
    query_vector = query_vector_response['embedding'][0]
    
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

if __name__ == '__main__':
    # Rularea acestui script va incerca sa creeze/actualizeze ChromaDB
    from data_processor import load_and_chunk_data
    chunks, metadata, ids = load_and_chunk_data()
    db_collection = create_or_update_db(chunks, metadata, ids)
    
    # Testare retrieval
    test_query = "Amenda pentru ITP expirat"
    retrieved = retrieve_chunks(db_collection, test_query, k=1)
    print(f"\n[DB Manager] Test Retrieval pentru '{test_query}':")
    if retrieved:
        print(f"Articol regăsit: {retrieved[0]['articol']}")