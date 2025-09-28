def build_hotpot_prompt(question, passages, top_entities, top_triples=None):
    entity_hint = ", ".join(top_entities[:5])
    numbered_passages = "\n\n".join([f"({i+1}) {p.strip()}" for i, p in enumerate(passages)])

    triple_text = ""
    if top_triples:
        triple_text = "\n\nStructured Knowledge Triples:\n" + "\n".join(
            [f"[T{i+1}] {s} {p} {o}" for i, (s, p, o) in enumerate(top_triples[:5])]
        )

    prompt = f"""
You are an intelligent fact-extraction system. Answer the user's question using only the numbered passages and the provided structured triples if needed. The goal is to return high-precision factual answers by selecting only direct evidence.

Passages:
{numbered_passages}

Important Entities: {entity_hint}{triple_text}

Question: {question}

### Instructions:
- Your response should **only contain exact sentence(s)** from the passages.
- If **no passage** contains the answer, fall back to **using the structured triples**.
- For Yes/No questions, start with "Yes" or "No", followed by the sentence or triple that supports it.
- Do **not add any explanation or paraphrasing**. Only copy directly.
- If the answer is obvious from a single sentence or triple, return only that.
- Prefer **precision** over coverage. If unsure, leave it blank.

Final Answer:"""

    print("\n🟢 FINAL PROMPT:\n" + "="*40)
    print(prompt)
    print("="*40)
    return prompt.strip()