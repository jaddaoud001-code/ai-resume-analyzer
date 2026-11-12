# Backend FastAPI. Le modèle d'embeddings local (~2,2 Go) n'est PAS dans l'image : il est
# téléchargé au premier démarrage dans le volume monté sur /models (voir docker-compose.yml),
# puis réutilisé aux démarrages suivants.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    EMBEDDINGS_CACHE_DIR=/models

WORKDIR /app

# Utilisateur non root ; /models lui appartient (un volume nommé neuf hérite de ces droits).
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /models \
    && chown app:app /models

# Dépendances dans une couche séparée : reconstruite seulement si requirements.txt change.
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

USER app
EXPOSE 8000

# /health répond pendant le chargement du modèle (embeddings_ready=false) : le conteneur est
# sain dès que l'API répond ; une analyse arrivée pendant le chargement l'attend.
HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--forwarded-allow-ips", "*"]
