import os
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector

load_dotenv()

ROOT_DIR = Path(__file__).resolve().parent.parent
PDF_PATH = os.getenv("PDF_PATH") or str(ROOT_DIR / "document.pdf")
DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documentos")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# Ingestão em lotes pequenos com backoff, para respeitar o rate limit
# gratuito da API de embeddings do Gemini.
EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "5"))
BATCH_PAUSE_SECONDS = float(os.getenv("EMBED_BATCH_PAUSE", "1"))
MAX_RETRIES = 6


def _add_with_backoff(store, batch, batch_num, total_batches):
    delay = 20.0
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            store.add_documents(batch)
            print(f"  lote {batch_num}/{total_batches} gravado ({len(batch)} chunks).")
            return
        except Exception as exc:
            is_quota = "429" in str(exc) or "quota" in str(exc).lower()
            if not is_quota or attempt == MAX_RETRIES:
                raise
            print(
                f"  lote {batch_num}/{total_batches}: rate limit atingido "
                f"(tentativa {attempt}/{MAX_RETRIES}), aguardando {delay:.0f}s..."
            )
            time.sleep(delay)
            delay *= 2


def ingest_pdf():
    if not DATABASE_URL:
        raise RuntimeError("Variável DATABASE_URL não configurada. Copie .env.example para .env.")
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Variável GOOGLE_API_KEY não configurada. Copie .env.example para .env.")

    pdf_path = Path(PDF_PATH)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF não encontrado em: {pdf_path}")

    print(f"Lendo PDF: {pdf_path}")
    documents = PyPDFLoader(str(pdf_path)).load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=False,
    )
    chunks = splitter.split_documents(documents)
    print(f"{len(documents)} página(s) divididas em {len(chunks)} chunk(s).")

    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    # pre_delete_collection recria a coleção, evitando duplicar dados a cada execução.
    store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
        pre_delete_collection=True,
    )

    total_batches = (len(chunks) + EMBED_BATCH_SIZE - 1) // EMBED_BATCH_SIZE
    print(
        f"Gerando embeddings e gravando na coleção '{COLLECTION_NAME}' "
        f"em {total_batches} lote(s) de até {EMBED_BATCH_SIZE}..."
    )
    for i in range(0, len(chunks), EMBED_BATCH_SIZE):
        batch = chunks[i : i + EMBED_BATCH_SIZE]
        batch_num = i // EMBED_BATCH_SIZE + 1
        _add_with_backoff(store, batch, batch_num, total_batches)
        if batch_num < total_batches:
            time.sleep(BATCH_PAUSE_SECONDS)

    print("Ingestão concluída com sucesso.")


if __name__ == "__main__":
    ingest_pdf()
