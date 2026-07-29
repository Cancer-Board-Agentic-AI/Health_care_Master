FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN mkdir -p knowledge/uploads knowledge/vectorstore
# Pre-cache embedding weights at build time (network available here) so the
# runtime container can load them with local_files_only=True while offline.
RUN python -c "from config.settings import get_settings; from huggingface_hub import snapshot_download; snapshot_download(repo_id=get_settings().models['embedding'])"
EXPOSE 8000
CMD ["uvicorn", "backend.fastapi:app", "--host", "0.0.0.0", "--port", "8000"]
