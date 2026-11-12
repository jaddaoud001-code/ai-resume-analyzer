"""Anonymisation des données personnelles d'un CV avant tout envoi à un fournisseur."""

import re

# Extensions acceptées pour un domaine écrit sans http(s):// ni www. : liste restreinte,
# en minuscules, pour ne pas prendre "Node.js", "ASP.NET", "Socket.IO" ou "README.md".
_BARE_TLDS = "com|fr|dev|io|me|net|org|app|ai|co|eu|tech|page|site|xyz|info|be|ch|ca|uk|de"
_URL_END = r"(?<![.,;:!?])"  # la ponctuation finale n'appartient pas à l'URL

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(
    rf"(?P<explicit>https?://|www\.)[^\s<>()\"']+{_URL_END}"
    rf"|\b(?:[a-z0-9-]+\.)+(?P<tld>{_BARE_TLDS})\b(?:/[^\s<>()\"']*)?{_URL_END}",
    re.IGNORECASE,
)

# Français : 06 12 34 56 78, 06.12.34.56.78, 0612345678…
_PHONE_FR = re.compile(r"(?<![\w+])0[1-9](?:[ .-]?\d{2}){4}(?!\d)")
# International : +33 6 12 34 56 78, +33 (0)6…, 0033…, +1 (415) 555-2671, +961 3 123 456.
_PHONE_INTL = re.compile(r"(?<![\w+])(?:\+|00)\d{1,3}(?:[ .-]?\(0\))?(?:[ .-]?\(?\d{1,4}\)?){2,6}(?!\d)")
_MIN_PHONE_DIGITS, _MAX_PHONE_DIGITS = 8, 15


def redact_pii(text: str) -> str:
    """Remplace emails, URLs et numéros de téléphone par [EMAIL], [URL] et [PHONE]."""
    text = _EMAIL.sub("[EMAIL]", text)
    text = _URL.sub(_redact_url, text)
    text = _PHONE_INTL.sub(_redact_phone, text)
    return _PHONE_FR.sub("[PHONE]", text)


def _redact_url(match: re.Match[str]) -> str:
    # IGNORECASE sert à "HTTPS://" et "WWW." ; un domaine nu exige une extension en minuscules.
    if match.group("explicit") or match.group("tld").islower():
        return "[URL]"
    return match.group(0)


def _redact_phone(match: re.Match[str]) -> str:
    # "(0)" est la forme française du préfixe national : il ne compte pas.
    digits = re.sub(r"\D", "", match.group(0).replace("(0)", ""))
    if _MIN_PHONE_DIGITS <= len(digits) <= _MAX_PHONE_DIGITS:
        return "[PHONE]"
    return match.group(0)
