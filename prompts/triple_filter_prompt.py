def build_fact_selection_prompt(
    question: str,
    triples: list,
    top_k: int = 4,
    max_candidates_in_prompt: int = 60,
) -> str:
    """
    Build a detailed, instruction-first prompt for selecting relevant fact triples.

    This function produces a plain text instruction that guides a language model
    to select at most top_k triples that are most helpful for answering the
    given question. The output format is constrained to a single JSON object
    with one key named fact that maps to a list of triples. Each triple must be
    a three element list of strings in the order subject, predicate, object.

    The prompt is written in simple present tense and avoids decorative symbols.
    It explains the task, defines what relevance means, and sets strict output
    rules so the downstream parser can safely load the result.

    Parameters
    ----------
    question : str
        The natural language question.
    triples : list
        A list of candidate triples. Each triple should be a list or tuple
        of length three in the order subject, predicate, object. Non string
        elements will be coerced to strings in the prompt rendering.
    top_k : int, optional
        The maximum number of triples to select. Defaults to 4.
    max_candidates_in_prompt : int, optional
        The maximum number of candidate triples to include in the prompt body
        to control token usage. If the input list is longer, it is truncated
        to the first max_candidates_in_prompt items. Defaults to 60.

    Returns
    -------
    str
        A formatted instruction string to send to a language model. The model
        must return a single line JSON object of the form
        {"fact": [["s","p","o"], ...]} with no additional text.
    """
    import json

    # Prepare a safe, bounded view of candidate triples for the prompt.
    # Coerce all triple fields to strings to avoid serialization surprises.
    candidates = []
    for t in triples[: max(0, int(max_candidates_in_prompt))]:
        try:
            s, p, o = t
        except Exception:
            # Skip malformed entries defensively
            continue
        s = "" if s is None else str(s)
        p = "" if p is None else str(p)
        o = "" if o is None else str(o)
        candidates.append([s, p, o])

    # The instruction below is deliberately explicit. It defines the goal,
    # relevance criteria, tie breaking, constraints, and the required output.
    # The last lines clearly state that the model must return only JSON.
    instruction = f"""
You are a careful reasoning assistant. You receive one question and a finite set of candidate facts expressed as subject, predicate, object triples. Your task is to select at most {top_k} triples that are most useful for answering the question or for supporting a short chain of reasoning that leads to an answer.

Follow these guidelines.

Goal
Select a compact set of triples that helps answer the question with high precision. Prefer triples that connect named entities in the question to other entities or attributes that are likely to resolve the information need.

Relevance
A triple is relevant when any of the following signals hold.
One or more triple terms match entities, aliases, titles, dates, or numeric quantities present in the question.
The triple links two or more concepts that together reduce ambiguity in the question.
The triple forms part of a short reasoning path that connects the main entity in the question to a required attribute such as a date, membership, location, affiliation, or comparison.
The triple helps disambiguate homonyms or resolve alternate names.

Selection policy
Prioritize precision over breadth. Avoid adding loosely related triples that are not on the shortest path to the answer.
Prefer triples that directly support a likely answer. Include bridge triples only when they are necessary to connect otherwise disjoint but important entities.
If multiple triples express the same relation or fact, keep only one representative that is clearest and most specific.
Resolve aliases by treating obvious alternate names as the same entity. Do not duplicate the same fact with different surface forms.
If a triple conflicts with another triple, choose the one that aligns with more reliable or direct evidence for the question.

Temporal and numeric cues
If the question mentions a specific time, year, or range, prefer triples whose time markers agree with that period.
If the question implies a first, earliest, latest, or most recent property, prefer triples that include ordering or date information.

Length and count limits
Return at most {top_k} triples.
If no triple appears relevant, return an empty list.

Constraints
Do not invent new facts. Only choose from the supplied candidates.
Keep the original surface forms for subject, predicate, and object as they appear in the candidates. Do not rewrite, reformat, or expand them.
Do not include explanations, comments, or additional keys in the output.

Output format
Return a single JSON object with exactly one key named fact.
The value must be a JSON array of triples, where each triple is a three element JSON array of strings in the order subject, predicate, object.
Do not wrap the JSON in code fences. Do not add any text before or after the JSON.

Question
{question}

Candidates
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Return only JSON in this exact shape.
{{"fact": [["s","p","o"], ...]}}
If no triple is relevant, return
{{"fact": []}}
"""

    # Strip leading and trailing whitespace so the prompt is clean in logs.
    return instruction.strip()
