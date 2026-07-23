"""
Lightweight contextual-bandit-style adaptation.

This is a deliberately simple analog to "the system improves with every
interaction" — an epsilon-greedy bandit picks between prompt variants per
agent based on thumbs-up rate collected from real usage. It is NOT full
RLHF/policy-gradient training; it's the smallest honest version of a
feedback loop that a POC can demonstrate in a day. The README documents
the path from this to a real RL/fine-tuning pipeline.
"""
import random

from . import config
from . import db


def choose_variant(agent: str, variants: list) -> str:
    """Epsilon-greedy: explore a random variant with probability EPSILON,
    otherwise exploit the variant with the highest observed thumbs-up rate."""
    if random.random() < config.EPSILON:
        return random.choice(variants)

    stats = db.get_variant_stats(agent)
    best_variant = variants[0]
    best_rate = -1.0
    for v in variants:
        uses, ups = stats.get(v, (0, 0))
        # Optimistic default for untried variants so they still get a chance
        rate = (ups / uses) if uses > 0 else 0.5
        if rate > best_rate:
            best_rate = rate
            best_variant = v
    return best_variant
