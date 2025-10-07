SYSTEM_PROMPT = (
    "You are a biomedical research assistant. "
    "Carefully read the given biomedical context and answer the question. "
    "Provide step-by-step reasoning using only the provided context. "
    "Conclude with a structured answer in XML tags."
)

BIOASQ_USER_PROMPT = """Question:
{question}

Context (Biomedical Text):
{context}

Please explain your reasoning, and then provide the final answer strictly in the following format:
<answer>
<item>Answer 1</item>
<item>Answer 2</item>
...
</answer>
"""
