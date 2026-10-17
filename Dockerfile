# FinSight on Hugging Face Spaces (Docker, free CPU: 2 vCPU, 16 GB RAM)
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/home/user/.cache/huggingface TOKENIZERS_PARALLELISM=false
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

# CPU-only PyTorch keeps the image small
COPY requirements.txt requirements-full.txt ./
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install -r requirements-full.txt

USER user
# download the embedding model and the reranker at build time, so start-up is fast
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('BAAI/bge-reranker-base')"

COPY --chown=user . .
EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
