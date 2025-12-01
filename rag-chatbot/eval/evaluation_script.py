import json
import os
import time
from google import genai
from rag_service import initialize_rag_system, process_query

# Configuratii
BENCHMARK_FILE = "benchmark_data.json"
REPORT_FILE = "raport_evaluare.txt"
JUDGE_MODEL = 'gemini-2.5-flash'

def get_judge_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY nu este setată.")
    return genai.Client(api_key=api_key)

def evaluate_retrieval(retrieved_text, expected_article_id, expected_source):
    """
    Metrică Deterministică: Verifică dacă ID-ul articolului corect apare în textul regăsit.
    """
    # Cautam ceva de genul "Art. 102" sau "Articolul 102"
    search_terms = [
        f"Art. {expected_article_id}",
        f"Articolul {expected_article_id}",
        f"Art. {expected_article_id}.", 
        f"Articolul {expected_article_id}."
    ]
    
    found = False
    for term in search_terms:
        if term.lower() in retrieved_text.lower():
            found = True
            break
            
    # Verificam si sursa (OUG vs HG vs Penal)
    source_match = False
    if expected_source.lower() in retrieved_text.lower():
        source_match = True
        
    if found and source_match:
        return 1.0 # Perfect
    elif found:
        return 0.5 # A gasit articolul dar poate sursa e ambigua
    else:
        return 0.0 # Fail

def evaluate_answer_quality_with_llm(user_question, rag_answer, key_facts):
    """
    Folosește LLM-ul ca Judecător pentru a compara răspunsul RAG cu faptele cheie.
    Returnează un scor de la 0 la 10.
    """
    client = get_judge_client()
    facts_str = ", ".join(key_facts)
    
    prompt = (
        "Ești un evaluator obiectiv pentru un sistem juridic AI. "
        f"Întrebare Utilizator: '{user_question}'\n"
        f"Fapte Cheie Obligatorii (Ground Truth): [{facts_str}]\n"
        f"Răspunsul Generat de AI: '{rag_answer}'\n\n"
        "Sarcina ta: Evaluează dacă Răspunsul Generat conține corect faptele cheie.\n"
        "Dă o notă de la 0 la 10, unde:\n"
        "0 = Răspuns complet greșit sau halucinație.\n"
        "5 = Răspuns parțial corect, lipsește un fapt cheie sau este vag.\n"
        "10 = Răspuns perfect, conține toate faptele cheie juridice.\n\n"
        "Returnează DOAR nota (un singur număr)."
    )
    
    try:
        response = client.models.generate_content(
            model=JUDGE_MODEL,
            contents=[prompt]
        )
        score = int(response.text.strip())
        return score
    except:
        return 0

def run_evaluation():
    print("="*60)
    print(" 🧪 PORNIRE EVALUARE AUTOMATĂ (PIPELINE DE TESTARE)")
    print("="*60)
    
    # 1. Incarcam datele de test
    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
        
    # 2. Initializam sistemul RAG (o singura data)
    # Nota: Cand te intreaba de re-indexare, alege 'nu' daca ai indexat deja corect
    collection = initialize_rag_system()
    
    total_retrieval_score = 0
    total_answer_score = 0
    results_log = []
    
    print(f"\nÎncep evaluarea pentru {len(test_cases)} cazuri de test...\n")
    
    for case in test_cases:
        print(f"Testare ID {case['id']}: {case['question']}...")
        
        # Rulam RAG-ul (ne intereseaza raspunsul text, dar trebuie sa capturam si contextul intern)
        # Pentru evaluare precisa, ar trebui sa modificam process_query sa returneze si contextul,
        # dar aici vom analiza raspunsul final care contine (Surse: ...)
        
        # Nota: process_query printeaza in consola, dar returneaza string-ul raspunsului
        # Pentru a nu polua consola, poti comenta print-urile din rag_service.py sau le ignori
        rag_output = process_query(collection, case['question'], k_results=15)
        
        # 1. Evaluare Retrieval (verificam daca sursa apare in raspunsul text la final)
        # Deoarece rag_service pune la final "(Surse: ...)", putem verifica acolo
        retrieval_score = evaluate_retrieval(rag_output, case['expected_article_id'], case['expected_source'])
        
        # 2. Evaluare Calitate Raspuns (LLM Judge)
        answer_score = evaluate_answer_quality_with_llm(case['question'], rag_output, case['key_facts'])
        
        # Logging
        log_entry = (
            f"ID: {case['id']}\n"
            f"Întrebare: {case['question']}\n"
            f"Răspuns RAG: {rag_output}\n"
            f"Așteptat (Articol): {case['expected_source']} Art. {case['expected_article_id']}\n"
            f"Scor Retrieval: {retrieval_score * 100}%\n"
            f"Scor Răspuns (Judge): {answer_score}/10\n"
            "--------------------------------------------------\n"
        )
        results_log.append(log_entry)
        
        total_retrieval_score += retrieval_score
        total_answer_score += answer_score
        
        print(f"  -> Retrieval: {retrieval_score}, Answer Quality: {answer_score}/10")
        time.sleep(2) # Pauza sa nu lovim limita de rate a API-ului
        
    # Calcul Statistici Finale
    avg_retrieval = (total_retrieval_score / len(test_cases)) * 100
    avg_answer = total_answer_score / len(test_cases)
    
    final_report = (
        "==================================================\n"
        "RAPORT FINAL DE EVALUARE AUTOMATĂ\n"
        "==================================================\n"
        f"Număr teste: {len(test_cases)}\n"
        f"Acuratețe Regăsire (Retrieval Accuracy): {avg_retrieval:.2f}%\n"
        f"Calitate Medie Răspuns (LLM Grade): {avg_answer:.2f}/10\n\n"
        "DETALII PE CAZURI:\n" + 
        "".join(results_log)
    )
    
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(final_report)
        
    print("\n" + "="*60)
    print(f"EVALUARE COMPLETĂ. Raport salvat în '{REPORT_FILE}'")
    print(f"Retrieval Accuracy: {avg_retrieval:.2f}%")
    print(f"Average Quality: {avg_answer:.2f}/10")
    print("="*60)

if __name__ == "__main__":
    if not os.getenv("GEMINI_API_KEY"):
        print("EROARE: Setează GEMINI_API_KEY!")
    else:
        run_evaluation()