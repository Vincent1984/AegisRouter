"""Minimal tokenizers stub for offline development.

This provides just enough to satisfy litellm's import requirement.
Full tokenizers functionality requires the huggingface tokenizers package.
"""


class Tokenizer:
    """Stub Tokenizer class."""

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        raise NotImplementedError("tokenizers stub: install 'tokenizers' for full functionality")

    @classmethod
    def from_file(cls, *args, **kwargs):
        raise NotImplementedError("tokenizers stub: install 'tokenizers' for full functionality")
