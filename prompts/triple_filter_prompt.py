def build_fact_selection_prompt(question: str, triples: list, top_k: int = 4) -> str:
    """
    Builds the prompt for selecting top-K relevant triples for a given question.
    
    Parameters:
        question (str): The natural language question.
        triples (list): List of triples, where each triple is a list or tuple of (subject, predicate, object).
        top_k (int): Maximum number of relevant triples to select.

    Returns:
        str: The formatted prompt string to be sent to the LLM.
    """
    import json

    prompt = f"""
You are a reasoning assistant. You are given a question and a list of candidate facts (triples). 
Each triple is in the form of subject-predicate-object.

Your task is to select the most relevant facts (maximum {top_k}) that may help answer or partially support the question.

Even if the facts do not directly answer the question, include them if they are likely to help infer the answer.

Return the selected triples in the JSON format: {{"fact": [["s", "p", "o"], ...]}}

If none are relevant at all, return: {{"fact": []}}

Question: {question}

Triples:
{json.dumps(triples, indent=2)}
"""

    return prompt.strip()
