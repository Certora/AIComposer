"""Generate human-readable mnemonic identifiers themed around formal verification and blockchain."""

import random

# ~80 adjectives: formal logic / math / crypto flavored
_ADJECTIVES = [
    # logic & proof
    "sound", "complete", "valid", "proven", "decidable",
    "modal", "temporal", "bounded", "abstract", "concrete",
    "reflexive", "transitive", "symmetric", "total", "partial",
    "monotone", "inductive", "coinductive", "constructive", "classical",
    "linear", "affine", "strict", "eager", "lazy",
    "atomic", "consistent", "stable", "convergent", "terminal",
    "closed", "open", "dense", "compact", "finite",
    "infinite", "maximal", "minimal", "optimal", "tight",
    "recursive", "continuous", "homomorphic", "convex",
    "transfinite", "temporal", "reduced",
    # crypto & blockchain
    "staked", "minted", "bonded", "bridged", "wrapped",
    "frozen", "liquid", "vested", "slashed", "forked",
    "sharded", "pruned", "anchored", "pegged", "settled",
    "signed", "hashed", "salted", "masked", "blinded",
    # general color
    "golden", "silver", "crimson", "cobalt", "amber",
    "ivory", "onyx", "jade", "scarlet", "obsidian",
    "azure", "violet", "copper", "iron", "crystal",
]

# ~80 nouns: verification / math / crypto / blockchain
_NOUNS = [
    # logic & verification
    "lemma", "axiom", "proof", "oracle", "witness",
    "predicate", "invariant", "theorem", "conjecture", "corollary",
    "lattice", "fixpoint", "closure", "domain", "model",
    "guard", "assert", "assume", "require", "satisfy",
    "trace", "reduct", "normal", "kernel", "image",
    "functor", "morphism", "monad", "topos", "sheaf", "monoid",
    "sieve", "filter", "ideal", "ring", "field",
    "analysis", "widening", "narrowing", "join", "meet", "heap",
    "stack", "rule", "codomain", "abstraction",
    # crypto & blockchain
    "ledger", "epoch", "block", "shard", "vault",
    "stake", "beacon", "bridge", "relay", "anchor",
    "cipher", "nonce", "merkle", "trie", "bloom",
    "genesis", "finality", "receipt", "calldata", "opcode",
    "delegator", "proposer", "attester", "sequencer", "prover",
    "verifier", "relayer", "keeper", "solver", "miner",
    # general evocative
    "summit", "forge", "nexus", "prism", "sigil",
    "glyph", "rune", "codex", "vertex", "helix",
    "pulse", "flux", "arc", "zenith", "core",
]

# ~40 verbs (present tense / gerund-ish, used as a third slot)
_VERBS = [
    "proving", "staking", "bridging", "minting", "forging",
    "hashing", "signing", "sealing", "binding", "folding",
    "reducing", "lifting", "mapping", "chaining", "linking",
    "merging", "forking", "syncing", "polling", "settling",
    "guarding", "draining", "locking", "burning", "wrapping",
    "slicing", "pruning", "tracing", "solving", "seeking",
    "anchoring", "relaying", "attesting", "proposing", "sequencing",
    "verifying", "yielding", "claiming", "indexing", "compiling",
]

# ~40 adverbs / manner words (fourth slot)
_ADVERBS = [
    "soundly", "fairly", "swiftly", "deeply", "tightly",
    "broadly", "cleanly", "safely", "firmly", "sharply",
    "purely", "deftly", "boldly", "keenly", "briskly",
    "calmly", "wholly", "justly", "truly", "fully",
    "onchain", "offchain", "upstream", "downstream", "inward",
    "outward", "forward", "upward", "always", "finally",
    "exactly", "densely", "finitely", "locally", "globally",
    "totally", "strictly", "lazily", "eagerly", "greedily",
    "unsoundly", "optimistically", "recursively", "iteratively", "infinitely"
]


def mnemonic_id(sep: str = "-") -> str:
    """Generate a 4-word mnemonic identifier.

    Format: adjective-noun-verb-adverb
    Example: "proven-lattice-forging-deeply"

    With ~75×75×40×40 combinations ≈ 9 million unique IDs.
    """
    return sep.join([
        random.choice(_ADJECTIVES),
        random.choice(_NOUNS),
        random.choice(_VERBS),
        random.choice(_ADVERBS),
    ])
