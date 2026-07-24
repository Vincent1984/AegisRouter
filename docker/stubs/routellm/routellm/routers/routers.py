"""Minimal router registry stub for aegis_router model_classifier."""


# Stub router classes
class MFRouter:
    """Matrix factorization router stub."""
    pass


class SWRankingRouter:
    """SW ranking router stub."""
    pass


class CausalLLMRouter:
    """Causal LLM router stub."""
    pass


class BERTRouter:
    """BERT router stub."""
    pass


ROUTER_CLS = {
    "mf": MFRouter,
    "sw_ranking": SWRankingRouter,
    "causal_llm": CausalLLMRouter,
    "bert": BERTRouter,
}
