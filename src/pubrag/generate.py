"""Génération de la réponse avec Claude, sous contrainte de citation."""

from __future__ import annotations

import os

from anthropic import Anthropic

from .retrieve import Hit

SYSTEM_PROMPT = """Tu réponds à des questions sur un corpus de publications \
scientifiques en génétique humaine, à partir des extraits fournis et d'eux seuls.

Règles impératives :
1. N'utilise que les extraits fournis. Ne complète jamais avec tes connaissances \
générales, même si tu connais le sujet.
2. Cite systématiquement tes sources au format [PMID:12345678], immédiatement après \
l'affirmation concernée. Une affirmation sans citation est une erreur.
3. Si les extraits ne permettent pas de répondre, dis-le explicitement. Ne comble pas.
4. Si les extraits se contredisent, signale la contradiction plutôt que de trancher.
5. Réponds dans la langue de la question, de façon concise et factuelle."""


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f'<extrait id="{index}" pmid="{hit.chunk.pmid}">\n'
            f"Source : {hit.chunk.citation}\n\n"
            f"{hit.chunk.text}\n"
            f"</extrait>"
        )
    return "\n\n".join(blocks)


class MissingAPIKeyError(RuntimeError):
    """Levée quand aucune clé Anthropic n'est configurée.

    Exception dédiée pour que l'API puisse la distinguer d'une vraie panne et
    répondre « fonction indisponible » plutôt que « erreur serveur » : la
    recherche, elle, fonctionne sans clé.
    """


class Generator:
    def __init__(
        self,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1200,
        temperature: float = 0.0,
        api_key: str | None = None,
    ) -> None:
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise MissingAPIKeyError(
                "ANTHROPIC_API_KEY absent. Copiez .env.example en .env et renseignez la clé."
            )
        self._client = Anthropic(api_key=key)
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature

    def answer(self, question: str, hits: list[Hit]) -> str:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=SYSTEM_PROMPT,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Extraits du corpus :\n\n{format_context(hits)}\n\nQuestion : {question}"
                    ),
                }
            ],
        )
        return "".join(block.text for block in message.content if block.type == "text")
