
import os
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed(text: str):
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding

def classify(query: str) -> str:
    prompt = f"""Classify query into VECTOR, SQL, or HYBRID.
Query: {query}
Return one word."""
    r = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
    return r.output_text.strip()

def generate_sql(prompt: str) -> str:
    r = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt
    )
    return r.output_text.strip()

def answer(prompt: str) -> str:
    r = client.responses.create(
        model="gpt-4.1",
        input=prompt
    )
    return r.output_text.strip()
