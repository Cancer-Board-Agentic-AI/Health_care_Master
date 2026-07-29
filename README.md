# Local Medical AI Multi-Agent System

A fully local clinical decision-support application. It uses the unified MedGemma medical multimodal model family for inference, LangGraph for a collaborating agent workflow, ChromaDB + SentenceTransformers for PDF retrieval, SQLite for history, FastAPI for the API, and Streamlit for the user interface. No API key or cloud inference is used by the application.

## Safety

This is educational decision support, not a medical device. It does not diagnose or prescribe. Emergency-warning wording triggers an immediate recommendation to seek emergency care; it cannot assess severity. Validate all output with a licensed clinician and local protocols.

## Architecture

`Supervisor → Triage / Symptoms / Diagnosis / Drug / Lab / RAG (fan-out) → Evidence aggregation → Consensus → One report`

Each specialist records reasoning, confidence, evidence, citations, recommendations, and uncertainty in LangGraph state. The report generator exposes one unified answer while Streamlit can show the individual evidence trails.

## Local installation

1. Install Python 3.12 and [Ollama](https://ollama.com).
2. In a terminal at this folder, create and activate a virtual environment.
3. Run `pip install -r requirements.txt`.
4. Pull the default unified medical model: `ollama pull medgemma:27b`. On lower-memory systems, pull `medgemma:4b` instead.
5. Start the API: `uvicorn backend.fastapi:app --reload`.
6. In another terminal start the UI: `streamlit run ui/streamlit_app.py`.
7. Open `http://localhost:8501` and upload PDFs in the sidebar. Uploads are chunked and stored in the local Chroma collection.
8. install ollama `curl -fsSL https://ollama.com/install.sh | sh`
9. 

Embeddings are loaded with `local_files_only=True`, so the running service never downloads models or sends data remotely. Before an air-gapped deployment, pre-cache `BAAI/bge-large-en-v1.5` in the SentenceTransformers/Hugging Face cache and pull the Ollama models while provisioning the machine. Missing embedding weights now surface as a clear RAG-unavailable condition rather than silently creating a weak fallback encoder.

## Docker deployment

Run `docker compose up --build -d`, then load the model once: `docker compose exec ollama ollama pull medgemma:27b`. The UI is on port 8501 and API on port 8000. Knowledge, Chroma persistence, and SQLite are bind-mounted under `knowledge/`.

For NVIDIA, install NVIDIA Container Toolkit and add the appropriate Compose GPU device reservation to the `ollama` service. Without it Ollama and embeddings use CPU automatically.

## Operations

- Edit `config/config.yaml` to select models, Ollama endpoint, chunking, and retrieval count.
- Place curated PDFs in `knowledge/pdfs/` to make them selectable as sample documents in the UI sidebar (for users without a local PDF to upload), or upload your own through the UI/API; all citations point to locally indexed documents.
- API endpoints: `GET /health`, `POST /ask`, `POST /documents`, `GET /documents/samples`, `POST /documents/samples/{filename}`, `GET /history`.
- Multimodal requests: use the Streamlit image uploader or `POST /ask-with-image` as multipart form data (`question`, `model`, `vision_model`, `image`). JPEG, PNG, and WEBP images up to 20 MB are accepted and retained only in `knowledge/uploads/`.
- Pull `medgemma:27b` before use (or `medgemma:4b` for lower-resource hardware). The same MedGemma model performs medical-image feature extraction and all text-based clinical reasoning; the LangGraph agents still provide independent roles, evidence aggregation, and one consensus report. Do not use image output as a diagnosis.
- Run `pytest` for the included offline unit tests.

## Project map

- `agents/`: LangGraph specialists, evidence aggregation, consensus, report formatting.
- `models/`: local Ollama and sentence-transformer providers.
- `rag/`: PDF ingestion, persistent Chroma store, local retriever.
- `backend/`: FastAPI and SQLite history.
- `ui/`: Streamlit chat interface.
