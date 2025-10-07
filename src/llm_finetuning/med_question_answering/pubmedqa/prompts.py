SYSTEM_PROMPT = (
    "You are a biomedical research assistant. "
    "Carefully read the given PubMed abstract and answer the question. "
    "Provide step-by-step reasoning using only the provided context. "
    "Conclude with a structured answer in XML tags."
)

PUBMEDQA_USER_PROMPT = """Question:
{question}

Context (PubMed Abstract):
{context}

Please explain your reasoning, and then provide the final decision strictly in one of the following formats:
<answer>Yes</answer>
<answer>No</answer>
<answer>Maybe</answer>
"""
