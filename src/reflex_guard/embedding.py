"""Dense semantic embedding stage (Section III-B).

BAAI/bge-small-en-v1.5: 33M parameters, 384 dimensions, CLS-token pooling as configured by the model, L2
normalised onto the unit hypersphere so that downstream similarity is angular.

A note on the instruction prefix. The paper prefixes inputs with ``"query:"``.
That is the E5 convention; BGE's own retrieval convention is
``"Represent this sentence for searching relevant passages: "``. For a symmetric
*classification* task neither prefix is obviously correct, and the empty prefix
is a defensible third option. The prefix is therefore configurable, defaulting to
the paper's value so results reproduce, and ``scripts/train.py --prefix ''``
lets you measure how much it actually matters.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BGEEmbedder", "DEFAULT_MODEL", "PAPER_PREFIX"]

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
PAPER_PREFIX = "query:"
BGE_RETRIEVAL_PREFIX = "Represent this sentence for searching relevant passages: "


class BGEEmbedder:
    """Sentence-transformer encoder with L2-normalised output.

    Parameters
    ----------
    model_name:
        HuggingFace model id.
    prefix:
        Instruction prefix prepended to every input. Pass ``""`` to disable.
    device:
        ``"cuda"``, ``"cpu"`` or ``None`` to auto-select.
    batch_size:
        Encoding batch size. Note that the latency figures in the paper are
        measured at batch size 1, which is the worst case; see
        ``scripts/latency_bench.py``.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        prefix: str = PAPER_PREFIX,
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: heavy import

        if device is None:
            try:
                import torch

                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:  # pragma: no cover
                device = "cpu"

        self.model_name = model_name
        self.prefix = prefix
        self.device = device
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)

    @property
    def dim(self) -> int:
        """Embedding size (384 for the default model)."""
        return int(self.model.get_sentence_embedding_dimension())

    def _format(self, texts: list[str]) -> list[str]:
        """Prepend the instruction prefix (e.g. ``"query:"``) to every input."""
        if not self.prefix:
            return list(texts)
        return [f"{self.prefix}{t}" for t in texts]

    def encode(
        self,
        texts: list[str] | str,
        batch_size: int | None = None,
        show_progress: bool = False,
    ) -> np.ndarray:
        """Return an ``(n, dim)`` float32 array of L2-normalised embeddings."""
        single = isinstance(texts, str)
        items = [texts] if single else list(texts)
        vecs = self.model.encode(
            self._format(items),
            batch_size=batch_size or self.batch_size,
            normalize_embeddings=True,          # unit hypersphere (Section III-B)
            convert_to_numpy=True,
            show_progress_bar=show_progress,
        ).astype(np.float32)
        return vecs[0] if single else vecs

    def __call__(self, texts: list[str] | str) -> np.ndarray:
        return self.encode(texts)
