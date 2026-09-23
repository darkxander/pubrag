FROM python:3.12-slim

# uv : installation des dépendances nettement plus rapide que pip.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Les dépendances d'abord : cette couche est mise en cache tant que
# pyproject.toml ne change pas.
COPY pyproject.toml README.md ./
COPY src/ src/
RUN uv pip install --system --no-cache '.[embed,api]'

COPY config/ config/
COPY eval/ eval/

# Pré-téléchargement des poids du modèle d'encodage, pour que le premier
# démarrage du conteneur ne dépende pas du réseau.
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('BAAI/bge-small-en-v1.5')"

ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "pubrag.api:app", "--host", "0.0.0.0", "--port", "8000"]
