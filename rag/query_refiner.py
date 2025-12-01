prompt = prompt = """You are a RAG Query Refinement Agent for a semantic search system (NOT a database).

Your job:
- Read the user question.
- Rewrite it into a clearer, more specific NATURAL-LANGUAGE search query.
- Be concise, factual, deterministic.
- Do NOT hallucinate new entities.

CRITICAL CONSTRAINTS:
- You are NOT allowed to write SQL.
- You are NOT allowed to write code or commands of any kind.
- You are NOT a database query generator.
- You MUST answer in plain natural language only.

OUTPUT FORMAT (MUST OBEY):
1. Output EXACTLY one <box>...</box> block.
2. Inside <box>...</box>, output ONLY ONE line of natural-language text.
3. Do NOT output JSON.
4. Do NOT output SQL.
5. Do NOT output code.
6. Do NOT wrap the query in quotes.

Good examples:

User question: "uber earnings 2024 q3"
You: <box>
uber 2024 Q3 earnings results and profitability
</box>

User question: "what laws control gdpr data deletion"
You: <box>
legal requirements for gdpr data deletion and the right to be forgotten
</box>

Now rewrite this user question into a single natural-language search query:

User question:
"{query}"

Remember:
- NATURAL LANGUAGE ONLY.
- NO SQL.
- NO CODE.
- ONLY the refined query inside <box>...</box>.
"""




from .llm_provider import BaseLLMProvider
import re
import json

def extract_box(text: str) -> str:
    matches = re.findall(r"<box>\s*(.*?)\s*</box>", text, flags=re.DOTALL | re.IGNORECASE)
    if matches:
        # return the first match
        return matches[-1].strip()

    cleaned = re.sub(r"<[^>]+>", "", text).strip()
    for line in cleaned.splitlines():
        s = line.strip()
        if s:
            return s
    return cleaned

class QueryRefiner:
    def __init__(self, llm_provider: BaseLLMProvider, max_new_tokens:int=120, temperature:float=0.7, top_p:float=0.95):
        self.llm_provider = llm_provider
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

    def get_refined_query(self, query: str) -> str:
        input_prompt = prompt.format(query=query)
        gen = self.llm_provider.generate(
            input_prompt,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p
        )
        print(gen)

        refined = extract_box(gen).strip()

        # if the model outputs surrounding quotes, remove one layer
        if len(refined) >= 2 and refined[0] == refined[-1] and refined[0] in ('"', "'"):
            refined = refined[1:-1].strip()

        # Fallback
        if not refined:
            refined = query

        return refined
