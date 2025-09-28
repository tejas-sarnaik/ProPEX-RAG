def build_sampleqa_prompt(
    question: str,
    passages: list,
    max_passages: int = 10,
    include_text_fields: tuple = ("text", "content", "body"),
    truncate_chars: int = 800,
) -> str:
    """
    Build a clear, instruction-first prompt for answer synthesis over retrieved passages.

    This function creates a plain text instruction that asks a language model to answer
    the given question using only the supplied passages. The prompt is written in a
    simple, abstract style and avoids decorative formatting. It defines the task,
    the evidence constraint, the selection policy when passages disagree, and the
    required behavior when the answer is not present in the retrieved context.

    Parameters
    ----------
    question : str
        The user question in natural language.
    passages : list
        A list of passage dictionaries. Each item may contain keys such as
        title, text, content, or body. If only a title is present, the title
        is used as the evidence string. If a text-like field is present, a
        short excerpt is included to provide more substance to the model.
    max_passages : int, optional
        The maximum number of passages to include in the prompt. Defaults to ten.
    include_text_fields : tuple, optional
        A tuple of candidate keys that may hold the passage body. The function
        uses the first matching key it finds in each passage. Defaults to the
        tuple text, content, body.
    truncate_chars : int, optional
        The maximum character count to include from any single passage body.
        Long passages are truncated to control token usage. Defaults to eight
        hundred characters.

    Returns
    -------
    str
        A formatted instruction for a language model. The model is expected to
        produce one concise answer. If the answer is not present in the supplied
        passages, the model must return the sentence Not found in retrieved context.
    """

    def _clean(s: str) -> str:
        if s is None:
            return ""
        s = str(s)
        s = s.replace("\r", " ").replace("\n", " ").strip()
        return " ".join(s.split())

    def _first_text_field(p: dict) -> str:
        for k in include_text_fields:
            if k in p and isinstance(p[k], str) and p[k].strip():
                return p[k]
        return ""

    # Prepare a bounded and sanitized set of passages for the prompt.
    # Each passage is rendered with a stable index, a title line, and an optional excerpt line.
    blocks = []
    for idx, p in enumerate(passages[: max(0, int(max_passages))], start=1):
        title = _clean(p.get("title", "")) or f"Passage {idx}"
        body = _clean(_first_text_field(p))
        if truncate_chars and body:
            body = body[:truncate_chars].rstrip()
        if body:
            block = f"Passage {idx}\nTitle: {title}\nExcerpt: {body}"
        else:
            block = f"Passage {idx}\nTitle: {title}"
        blocks.append(block)

    rendered_passages = "\n\n".join(blocks) if blocks else "No passages available."

    # The instruction below is explicit and minimal. It defines evidence use,
    # conflict handling, precision preference, and the required fallback when
    # the answer is not present in the retrieved context. It requests a single,
    # concise answer and forbids unsupported content.
    instruction = f"""
You answer the question using only the passages that follow. Use the passages as the sole evidence source. Prefer precision over speculation. If the passages do not contain the answer, respond with the exact sentence Not found in retrieved context.

When passages disagree, choose the answer that is most specific and most directly supported by the text. Prefer passages that name exact entities, dates, quantities, or definitions that align with the question. If several passages support the same answer, keep it concise. Do not copy long passages. Do not include citations or explanations. Do not use outside knowledge beyond what appears in the passages.

Passages
{rendered_passages}

Question
{_clean(question)}

Answer
""".strip()

    return instruction
