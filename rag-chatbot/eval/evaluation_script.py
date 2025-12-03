import json
import os
from src.rag_service import process_query, initialize_rag_system

# Configuratii
BENCHMARK_FILE = "eval/benchmark_data.json"
REPORT_FILE = "raport_evaluare.txt"
JUDGE_MODEL = 'llama3.2'

def evaluate_retrieval(retrieved_text, expected_article_id, expected_source):
    """
    Metrică Deterministică: Verifică dacă ID-ul articolului corect apare în textul regăsit.
    """
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
            
    source_match = False
    if expected_source.lower() in retrieved_text.lower():
        source_match = True
        
    if found and source_match:
        return 1.0 
    elif found:
        return 0.5 
    else:
        return 0.0

def evaluate_answer_quality_with_llm(user_question, rag_answer, key_facts):
    """
    Folosește LLM-ul Local (Ollama) ca Judecător.
    """
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
        "Răspunde DOAR cu nota numerică (un singur număr, de exemplu: 8). Nu scrie alt text."
    )
    
    try:
        response = ollama.chat(model=JUDGE_MODEL, messages=[
            {'role': 'user', 'content': prompt}
        ])
        
        # Curățăm răspunsul pentru a extrage doar numărul
        content = response['message']['content'].strip()
        # Extragem doar cifrele din raspuns (in caz ca modelul e verbos)
        import re
        match = re.search(r'\b(10|[0-9])\b', content)
        if match:
            return int(match.group(1))
        return 0
    except Exception as e:
        print(f"[WARN] Eroare la evaluarea cu LLM: {e}")
        return 0

def run_evaluation():
    print("="*60)
    print(" 🦙 PORNIRE EVALUARE AUTOMATĂ (LOCAL - OLLAMA)")
    print("="*60)
    
    if not os.path.exists(BENCHMARK_FILE):
        print(f"[EROARE] Nu găsesc fișierul {BENCHMARK_FILE}.")
        return

    with open(BENCHMARK_FILE, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)
        
    # Initializam sistemul RAG (care acum foloseste si el Ollama din rag_service.py)
    original_cwd = os.getcwd()
    os.chdir(parent_dir) 
    try:
        # Nota: Alege 'nu' la re-indexare daca ai deja baza de date locala facuta
        collection = initialize_rag_system()
    except Exception as e:
        print(f"[FATAL] Eroare la inițializare: {e}")
        os.chdir(original_cwd)
        return
    
    total_retrieval_score = 0
    total_answer_score = 0
    results_log = []
    
    print(f"\nÎncep evaluarea pentru {len(test_cases)} cazuri de test...\n")
    
    # Schimbam directorul pentru a rula procesele
    os.chdir(parent_dir)

    for case in test_cases:
        print(f"Testare ID {case['id']}: {case['question']}...")
        
        # Rulam RAG-ul
        rag_output = process_query(collection, case['question'], k_results=10)
        
        # 1. Evaluare Retrieval
        retrieval_score = evaluate_retrieval(rag_output, case['expected_article_id'], case['expected_source'])
        
        # 2. Evaluare Calitate Raspuns (Local Judge)
        answer_score = evaluate_answer_quality_with_llm(case['question'], rag_output, case['key_facts'])
        
        log_entry = (
            f"ID: {case['id']}\n"
            f"Întrebare: {case['question']}\n"
            f"Răspuns RAG: {rag_output}\n"
            f"Așteptat: {case['expected_source']} Art. {case['expected_article_id']}\n"
            f"Scor Retrieval: {retrieval_score * 100}%\n"
            f"Scor Răspuns (Llama Judge): {answer_score}/10\n"
            "--------------------------------------------------\n"
        )
        results_log.append(log_entry)
        
        total_retrieval_score += retrieval_score
        total_answer_score += answer_score
        
        print(f"  -> Rezultat: Retrieval={retrieval_score*100}%, Calitate={answer_score}/10")
        # Nu e nevoie de sleep la Ollama local, dar poate ajuta la output curat
        
    # Revenim la directorul original
    os.chdir(original_cwd)

    avg_retrieval = (total_retrieval_score / len(test_cases)) * 100
    avg_answer = total_answer_score / len(test_cases)
    
    final_report = (
        "==================================================\n"
        "RAPORT FINAL DE EVALUARE AUTOMATĂ (LOCAL)\n"
        "==================================================\n"
        f"Număr teste: {len(test_cases)}\n"
        f"Judecător: {JUDGE_MODEL} (Ollama)\n"
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
    run_evaluation()