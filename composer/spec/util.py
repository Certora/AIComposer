import hashlib

def string_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]
