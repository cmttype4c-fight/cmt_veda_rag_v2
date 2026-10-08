# RAG v2 image. One image serves three roles (selected by the command):
#   api      uvicorn main:app          (default CMD)
#   worker   python worker.py          (exactly ONE replica; see deploy/README.md)
#   migrate  deploy/migrate.sh         (one-shot, idempotent)
#
# No secrets, DSNs, or API keys are baked in. All configuration arrives at
# run time through the environment (see deploy/rag-v2.env.example).

# ---------- build stage: compilers live here only ----------
FROM python:3.11-slim AS build
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential cmake git \
 && rm -rf /var/lib/apt/lists/*
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# CPU-only torch first, so sentence-transformers does not pull the CUDA build.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
# Fail the BUILD (not the first request) if a required module is missing.
RUN python -c "import fastapi, uvicorn, psycopg2, redis, rq, numpy, sklearn, pypdf, faiss, sentence_transformers, llama_cpp, multipart; print('dependency check ok')"
# Bake the embedding model into the image so start-up never needs the network.
ENV HF_HOME=/opt/hf
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# ---------- runtime stage ----------
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# libgomp1: OpenMP runtime needed by faiss / llama.cpp. postgresql-client: psql for migrate.sh.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 postgresql-client \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/venv /opt/venv
COPY --from=build /opt/hf /opt/hf

# uid 10001 must be able to write the mounted runtime dir (see deploy/README.md).
RUN useradd --system --uid 10001 --home-dir /app --shell /usr/sbin/nologin rag
WORKDIR /app
COPY --chown=rag:rag . /app
RUN chmod +x /app/deploy/*.sh
USER rag
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
