ENTITY_EXTRACTION_PROMPT = """
You are a high-accuracy NLP assistant. Your task is to extract **named entities** from the provided paragraph.

Instructions:
- Extract only meaningful named entities such as PERSON, ORGANIZATION, LOCATION, DATE, PRODUCT, EVENT, etc.
- Normalize entities (e.g., avoid duplicates or variants).
- Do NOT extract common nouns or generic terms.
- Return the result as a strict JSON list in the format: {"named_entities": [ ... ]}.

# 1-shot example
Paragraph:
Lionel Messi
After a year at Barcelona's youth academy, La Masia, Messi was finally enrolled in the Royal Spanish Football Federation (RFEF) in February 2002. Now playing in all competitions, he befriended his teammates, among whom were Cesc Fàbregas and Gerard Piqué. After completing his growth hormone treatment aged 14, he chose to remain in Barcelona.

Output:
{"named_entities": [
    "Lionel Messi", "Barcelona", "La Masia", "Royal Spanish Football Federation", "RFEF", 
    "February 2002", "Cesc Fàbregas", "Gerard Piqué"
]}
"""

TRIPLE_EXTRACTION_PROMPT = """
You are an intelligent assistant that constructs RDF triples from passages and extracted named entities.

Instructions:
- Construct accurate factual relationships based on the provided passage.
- Each triple should follow the RDF format: [Subject, Predicate, Object].
- Use only entities from the named_entities list. Avoid generic terms or pronouns.
- Output format MUST be JSON: {"triples": [[subj, pred, obj], ...]}

# 1-shot example paragraph + entities
Paragraph:
Lionel Messi
After a year at Barcelona's youth academy, La Masia, Messi was finally enrolled in the Royal Spanish Football Federation (RFEF) in February 2002. Now playing in all competitions, he befriended his teammates, among whom were Cesc Fàbregas and Gerard Piqué. After completing his growth hormone treatment aged 14, he chose to remain in Barcelona.

Named Entities:
{"named_entities": [
    "Lionel Messi", "Barcelona", "La Masia", "Royal Spanish Football Federation", "RFEF", 
    "February 2002", "Cesc Fàbregas", "Gerard Piqué"
]}

Output:
{"triples": [
    ["Lionel Messi", "attended", "La Masia"],
    ["La Masia", "is", "Barcelona's youth academy"],
    ["Lionel Messi", "enrolled in", "Royal Spanish Football Federation"],
    ["Royal Spanish Football Federation", "abbreviated as", "RFEF"],
    ["Lionel Messi", "enrolled on", "February 2002"],
    ["Lionel Messi", "teammate of", "Cesc Fàbregas"],
    ["Lionel Messi", "teammate of", "Gerard Piqué"],
    ["Lionel Messi", "chose to remain in", "Barcelona"],
    ["Lionel Messi", "completed treatment at age", "14"]
]}
"""

