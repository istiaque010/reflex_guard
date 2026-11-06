"""Jailbreak-aware preprocessing (Section III-A of the paper).

Three pattern families are handled, and they are handled differently on purpose:

* **Base64 payloads** are *decoded*. An encoded prompt is near-orthogonal to its
  plaintext in embedding space, so without decoding the classifier sees noise.
  This is the single highest-impact preprocessing step in the paper (Base64
  recall 7% -> 100%).
* **GCG-style adversarial suffixes** are *detected but deliberately not removed*.
  The [CLS] embedding attenuates a short high-entropy suffix, and the suffix's presence
  is itself weak evidence of an adversarial prompt.
* **DrAttack-style structural framing** raises a sensitivity flag only. These
  patterns occur in perfectly legitimate prompts, so acting on them alone would
  be unacceptable; the flag is consumed downstream by the two-tier threshold
  policy in :mod:`reflex_guard.guard`.

Honest scope note: the decoder handles Base64 only. Hex, ROT-13, Base32, URL
encoding and nested encodings are NOT handled and will pass through untouched.
This is a real limitation of the method as published, not an oversight here.
"""

from __future__ import annotations

import base64
import binascii
import re
from dataclasses import dataclass, field

__all__ = ["PreprocessResult", "JailbreakAwarePreprocessor"]

# A base64 run long enough to plausibly carry an instruction. The paper's
# sliding-4-character-window formulation (Eq. 1) is an indicator for the same
# character class; a length floor is used here to avoid firing on ordinary words.
_B64_RUN = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")

# GCG suffixes are high-entropy token soup: runs of punctuation, stacked
# brackets, backslash escapes, repeated separators.
_GCG_PATTERNS = (
    re.compile(r"[!@#$%^&*()\[\]{}<>~`|\\/_=+-]{6,}"),
    re.compile(r"(?:\s*\\[a-zA-Z]+){3,}"),
    re.compile(r"(?:==|\|\||--|\.\.\.){3,}"),
    re.compile(r"(?:\{\s*\}|\[\s*\]|<\s*>){3,}"),
)

# Structural / role framing used by DrAttack and similar semantic-camouflage
# attacks. Deliberately broad; precision is poor by construction.
_STRUCTURAL_PATTERNS = (
    re.compile(r"\bas a (?:researcher|scientist|professor|historian|expert)\b", re.I),
    re.compile(r"\bfor (?:educational|academic|research|training) purposes\b", re.I),
    re.compile(r"\b(?:please )?consider (?:this|the following) (?:scenario|situation)\b", re.I),
    re.compile(r"\byou are (?:now )?(?:a|an) [A-Za-z ]{0,40}\b(?:who|that) (?:can|will|must)\b", re.I),
    # Role assignment to an authoritative persona, without a relative clause.
    re.compile(
        r"\byou are (?:now )?(?:a|an) [\w ]{0,40}?"
        r"(?:consultant|expert|researcher|specialist|analyst|advisor|adviser|"
        r"professor|engineer|historian|author|writer|investigator)\b",
        re.I,
    ),
    re.compile(r"\bhypothetically(?: speaking)?\b", re.I),
    re.compile(r"\bin a fictional\b", re.I),
    re.compile(r"\bfor a (?:novel|story|screenplay|game)\b", re.I),
    # Narrative pretext stated in the first person rather than as "for a novel".
    re.compile(
        r"\b(?:i am|i'm|we are|we're) writing (?:a|an|my) "
        r"(?:novel|story|book|screenplay|script|play|thriller|article)\b",
        re.I,
    ),
    re.compile(r"\bignore (?:all )?(?:previous|prior|above) instructions\b", re.I),
)

DECODED_MARKER = "DECODED"

# Sentence-final punctuation that is perfectly normal and must not count as odd.
_ORDINARY_PUNCT = ".,!?;:\"'"
_INTERNAL_CAPITAL = re.compile(r"[a-z][A-Z]")
_NON_LEXICAL = re.compile(r"[^\w'-]")


def _is_odd_token(token: str) -> bool:
    """True if *token* does not look like an ordinary word.

    Catches pure-punctuation tokens, camel-cased fragments ("similarlyNow") and
    letters fused to symbols ("giving**ONE"), while leaving hyphenated words,
    contractions and normally punctuated words alone.
    """
    core = token.strip(_ORDINARY_PUNCT)
    if not core:
        return True
    return bool(_INTERNAL_CAPITAL.search(core) or _NON_LEXICAL.search(core))


@dataclass
class PreprocessResult:
    """Outcome of preprocessing a single prompt."""

    text: str
    """Text to embed. Equals the input unless a payload was decoded."""

    original: str
    base64_decoded: bool = False
    gcg_suspected: bool = False
    structural_flag: bool = False
    decoded_segments: list[str] = field(default_factory=list)

    @property
    def any_flag(self) -> bool:
        return self.base64_decoded or self.gcg_suspected or self.structural_flag


class JailbreakAwarePreprocessor:
    """Detect and normalise obfuscated prompts before embedding.

    Parameters
    ----------
    decode_base64:
        Decode detected Base64 runs and splice the plaintext back in.
    min_decoded_printable:
        Fraction of decoded bytes that must be printable ASCII before a decode
        is accepted. Guards against treating arbitrary hex/binary blobs as text.
    """

    def __init__(self, decode_base64: bool = True, min_decoded_printable: float = 0.85) -> None:
        self.decode_base64 = decode_base64
        self.min_decoded_printable = min_decoded_printable

    # ------------------------------------------------------------------ base64
    @staticmethod
    def _try_decode(token: str) -> str | None:
        """Return decoded text, or None if *token* is not decodable ASCII text."""
        padded = token + "=" * (-len(token) % 4)
        try:
            raw = base64.b64decode(padded, validate=True)
        except (binascii.Error, ValueError):
            return None
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return None

    def _decode_payloads(self, text: str) -> tuple[str, list[str]]:
        decoded: list[str] = []

        def repl(match: re.Match[str]) -> str:
            token = match.group(0)
            out = self._try_decode(token)
            if out is None or not out.strip():
                return token
            printable = sum(32 <= ord(c) < 127 or c in "\n\t" for c in out) / len(out)
            if printable < self.min_decoded_printable:
                return token
            decoded.append(out)
            return f"{DECODED_MARKER}: {out}"

        return _B64_RUN.sub(repl, text), decoded

    # ----------------------------------------------------------------- detect
    @staticmethod
    def detect_gcg(text: str, tail_tokens: int = 8, odd_ratio: float = 0.35) -> bool:
        """Flag GCG-style adversarial suffixes.

        Regexes alone are not sufficient: a real GCG suffix such as

            describing.\\ + similarlyNow write oppositeley.]( Me giving**ONE

        contains no long punctuation run, so pattern matching misses it. The
        discriminative property is that the *tail* of the prompt degenerates into
        non-lexical tokens - stray punctuation, camel-cased fragments, letters
        fused to symbols - while the head stays fluent. We therefore combine the
        patterns with a token-shape statistic over the trailing tokens.

        This is a surface heuristic, not a detector with a measured ROC. It exists
        to annotate prompts, never to block them on its own.
        """
        if any(p.search(text) for p in _GCG_PATTERNS):
            return True
        tokens = text.split()
        if len(tokens) < tail_tokens:
            return False
        tail = tokens[-tail_tokens:]
        return (sum(_is_odd_token(t) for t in tail) / len(tail)) >= odd_ratio

    @staticmethod
    def detect_structural(text: str) -> bool:
        """Structural/role framing indicative of semantic-camouflage attacks.

        WARNING: this detector is *not* validated in the paper. Its false-positive
        rate on benign prompts that legitimately use research or educational
        framing is unmeasured. ``scripts/structural_detector_eval.py`` measures it;
        run that before relying on the two-tier threshold policy.
        """
        return any(p.search(text) for p in _STRUCTURAL_PATTERNS)

    # -------------------------------------------------------------------- api
    def __call__(self, text: str) -> PreprocessResult:
        return self.process(text)

    def process(self, text: str) -> PreprocessResult:
        out = text
        decoded: list[str] = []
        if self.decode_base64:
            out, decoded = self._decode_payloads(out)
        return PreprocessResult(
            text=out,
            original=text,
            base64_decoded=bool(decoded),
            gcg_suspected=self.detect_gcg(text),
            structural_flag=self.detect_structural(text),
            decoded_segments=decoded,
        )

    def process_batch(self, texts: list[str]) -> list[PreprocessResult]:
        return [self.process(t) for t in texts]
