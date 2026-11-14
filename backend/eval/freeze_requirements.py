"""Fige les exigences d'une offre de référence pour l'évaluation du matching (hors pytest).

Appelle une fois l'extraction (vrai LLM, clé dans .env, limiteur de l'application) et écrit
`offers/<offre>.requirements.json` : exigences, modèle, version du prompt et date. La liste
est ensuite relue à la main ; eval_matching.py l'utilise telle quelle, sans réextraire,
pour que les variations de l'extraction ne se mélangent pas à celles du matching.

Usage (depuis backend/) :
    .venv/bin/python eval/freeze_requirements.py eval/offers/sii.txt [--model X] [--force]
"""

import argparse
import asyncio
import json
import sys
from datetime import date
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app.config import Settings  # noqa: E402
from app.llm import get_llm, llm_model  # noqa: E402
from app.llm_calls import RateLimiter  # noqa: E402
from app.requirements import EXTRACTION_PROMPT_VERSION, extract_requirements  # noqa: E402


def frozen_path(offer: Path) -> Path:
    return offer.with_name(f"{offer.stem}.requirements.json")


async def freeze(offer: Path, model: str | None = None) -> dict[str, object]:
    settings = Settings()
    if model:
        settings = settings.model_copy(update={"llm_model": model})
    result = await extract_requirements(
        offer.read_text(encoding="utf-8"), get_llm(settings), settings, RateLimiter.from_settings(settings)
    )
    return {
        "offer": offer.name,
        "extraction_model": llm_model(settings),
        "extraction_prompt_version": EXTRACTION_PROMPT_VERSION,
        "frozen_on": date.today().isoformat(),
        **result.model_dump(mode="json"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("offer", type=Path)
    parser.add_argument("--model", help="modèle d'extraction ; défaut : LLM_MODEL du .env")
    parser.add_argument("--force", action="store_true", help="remplace une liste déjà figée")
    args = parser.parse_args()
    target = frozen_path(args.offer)
    if target.exists() and not args.force:
        parser.error(f"{target} existe déjà (--force pour la remplacer)")
    data = asyncio.run(freeze(args.offer, args.model))
    target.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{target} : {len(data['requirements'])} exigences")  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
