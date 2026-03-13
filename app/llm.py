from typing import List

from openai import OpenAI


def answer_with_context(question: str, contexts: List[str]) -> str:
    """
    Call an LLM to answer a question given supporting context chunks.

    Expects OPENAI_API_KEY to be set in the environment.
    """
    client = OpenAI()

    system_prompt = (
        "You are an AI notes assistant. "
        "Answer the user's question using ONLY the provided context from their documents. "
        "If the answer is not in the context, say you are not sure."
    )

    joined_context = "\n\n---\n\n".join(contexts)

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    "Context from notes:\n"
                    f"{joined_context}\n\n"
                    "Question:\n"
                    f"{question}"
                ),
            },
        ],
    )

    return response.choices[0].message.content or ""


__all__ = ["answer_with_context"]

