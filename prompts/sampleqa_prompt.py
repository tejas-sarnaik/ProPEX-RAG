def build_sampleqa_prompt(question, passages):
    numbered_passages = "\n\n".join([
        f"({i+1}) {p['title'].strip()}" for i, p in enumerate(passages)
    ])

    prompt = f"""
You are a QA system. Use the following retrieved passages to answer the user's question as accurately as possible.

Passages:
{numbered_passages}

Question: {question}

### Instructions:
- Answer concisely using the content from the passages.
- Prefer precision and do not guess.
- If the answer is not found in the passages, reply: "Not found in retrieved context."

Answer:"""

    print("\n🟢 FINAL SAMPLEQA PROMPT:\n" + "="*40)
    print(prompt)
    print("="*40)

    return prompt.strip()
