from typing import TYPE_CHECKING

# claim we always import ST
if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer #type: ignore
    def get_model() -> SentenceTransformer:
        ...
else:
    try:
        from sentence_transformers import SentenceTransformer #type: ignore

        def get_model() -> SentenceTransformer:
            return SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)
    except ImportError:
        # for tests (no ST dependency)
        def get_model() -> "SentenceTransformer":
            raise NotImplementedError("Sentence transformers not available")

