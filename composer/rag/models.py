from sentence_transformers import SentenceTransformer #type: ignore

def get_model() -> SentenceTransformer:
    return SentenceTransformer('nomic-ai/nomic-embed-text-v1', trust_remote_code=True)
