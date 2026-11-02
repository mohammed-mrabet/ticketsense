"""Nettoyage du texte des tickets avant vectorisation."""

import re
import unicodedata

_SPACES = re.compile(r"\s+")
_NOISE = re.compile(r"[^a-z0-9%\s]")


def strip_accents(text: str) -> str:
    """« Écran bloqué » -> « Ecran bloque » : les tickets sont souvent tapés sans accents."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(c for c in normalized if unicodedata.category(c) != "Mn")


def clean_text(text: str | None) -> str:
    """Minuscules, sans accents ni ponctuation, espaces normalisés."""
    if not text:
        return ""
    text = strip_accents(text.lower())
    text = _NOISE.sub(" ", text)
    return _SPACES.sub(" ", text).strip()


def ticket_text(subject: str | None, body: str | None) -> str:
    """Objet + corps : l'objet est court mais souvent très informatif."""
    return clean_text(f"{subject or ''} . {body or ''}")
