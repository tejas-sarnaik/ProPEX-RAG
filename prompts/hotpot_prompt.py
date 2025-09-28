def build_hotpot_prompt(
    question: str,
    passages: list,
    top_entities: list,
    top_triples: list | None = None,
    max_passages: int = 12,
    max_entities: int = 8,
    max_triples: int = 8,
    include_passage_indices: bool = True,
    truncate_chars: int = 1200,
) -> str:
    """
    Build a clear instruction-first prompt for multi hop question answering over retrieved passages,
    augmented by a small set of entity hints and optional structured fact triples.

    The prompt is written in plain language and is designed to guide a language model to
    use the passages as the primary evidence source. If the passages do not contain the
    answer, the model may consult the structured triples. The prompt requests a precise
    answer with minimal text and defines specific fallback behavior when the answer is
    not present. It avoids decorative markup and focuses on clarity and constraints.

    Parameters
    ----------
    question : str
        The natural language question to answer.
    passages : list
        A list of passage strings. Each string is a passage body or an already
        concatenated title and body. Passages are trimmed and optionally truncated.
    top_entities : list
        A list of entity names as strings. A small subset is shown to hint which
        entities are most relevant. The model should not fabricate content based
        solely on these names.
    top_triples : list or None, optional
        A list of structured fact triples, each triple as a three element list or
        tuple in the order subject, predicate, object. This section is optional
        and is used as a secondary source if the passages do not contain the answer.
    max_passages : int, optional
        Maximum number of passages to include in the prompt. Defaults to twelve.
    max_entities : int, optional
        Maximum number of entity names to expose in the hint line. Defaults to eight.
    max_triples : int, optional
        Maximum number of triples to include. Defaults to eight.
    include_passage_indices : bool, optional
        When true, each passage is labeled with a stable line starting with the
        term Passage and an ordinal. This improves the model ability to reference
        specific snippets internally without requiring citations in the answer.
        Defaults to true.
    truncate_chars : int, optional
        Maximum number of characters to include from each passage to control
        token usage. Defaults to one thousand two hundred characters.

    Returns
    -------
    str
        A formatted instruction string suitable for sending to a language model.
        The returned instruction asks for a concise final answer grounded in the
        passages, with defined fallback to structured facts, and a final fallback
        message if the answer is not present.
    """

    def _clean_text(s: str) -> str:
        if s is None:
            return ""
        s = str(s).replace("\r", " ").replace("\n", " ").strip()
        # Collapse internal whitespace
        return " ".join(s.split())

    def _truncate(s: str, limit: int) -> str:
        if limit is None or limit <= 0:
            return s
        if len(s) <= limit:
            return s
        return s[:limit].rstrip()

    # Prepare passages
    prepped_passages = []
    for idx, p in enumerate(passages[: max(0, int(max_passages))], start=1):
        body = _truncate(_clean_text(p), int(truncate_chars))
        if not body:
            continue
        if include_passage_indices:
            block = f"Passage {idx}\n{body}"
        else:
            block = body
        prepped_passages.append(block)

    rendered_passages = "\n\n".join(prepped_passages) if prepped_passages else "No passages available."

    # Prepare entity hint line
    shown_entities = [e for e in (top_entities or []) if isinstance(e, str)]
    entity_hint = ", ".join(shown_entities[: max(0, int(max_entities))]) if shown_entities else ""

    # Prepare structured triples
    rendered_triples = ""
    if top_triples:
        triples_buf = []
        for t_idx, t in enumerate(top_triples[: max(0, int(max_triples))], start=1):
            try:
                s, p, o = t
            except Exception:
                # Skip malformed entries defensively
                continue
            s = _clean_text(s)
            p = _clean_text(p)
            o = _clean_text(o)
            # Each triple is rendered in a simple human readable line without special symbols
            triples_buf.append(f"Triple {t_idx}\nSubject: {s}\nPredicate: {p}\nObject: {o}")
        if triples_buf:
            rendered_triples = "\n\nStructured facts\n" + "\n\n".join(triples_buf)

    # Assemble instruction
    # The instruction enforces evidence use, conflict handling, precision preference,
    # and explicit fallbacks. It asks for a single concise answer without citations.
    instruction = f"""
You answer the question using only the passages that follow. Use the passages as the primary source of evidence. If the answer does not appear in any passage, you may use the structured facts that follow the entity hint. If neither the passages nor the structured facts contain the answer, respond with the exact sentence Not found in retrieved context.

When multiple passages discuss related content, prefer the passage that states the fact most directly and precisely. When the passages disagree, choose the statement that is most specific and that names concrete entities, dates, places, or quantities that match the question. Do not use knowledge that is not present in the passages or the structured facts. Do not provide citations or references. Do not copy long spans of text. Keep the final answer concise and factual.

If the question requires a yes or no decision, begin the answer with Yes or No, then include a short supporting sentence that appears in a passage. If no passage contains a supporting sentence, you may include a short supporting line derived from a single structured fact. If neither source supports a decision, respond with Not found in retrieved context.

Important entities
{entity_hint}

Passages
{rendered_passages}
""".rstrip()

    if rendered_triples:
        instruction = f"{instruction}\n\n{rendered_triples}"

    instruction = f"{instruction}\n\nQuestion\n{_clean_text(question)}\n\nAnswer\n".strip()

    return instruction
