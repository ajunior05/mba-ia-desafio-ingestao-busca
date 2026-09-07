PROMPT_TEMPLATE = """
CONTEXTO:
{contexto}

REGRAS:
- Responda somente com base no CONTEXTO.
- Se a informação não estiver explicitamente no CONTEXTO, responda:
  "Não tenho informações necessárias para responder sua pergunta."
- Nunca invente ou use conhecimento externo.
- Nunca produza opiniões ou interpretações além do que está escrito.

EXEMPLOS DE PERGUNTAS FORA DO CONTEXTO:
Pergunta: "Qual é a capital da França?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Quantos clientes temos em 2024?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

Pergunta: "Você acha isso bom ou ruim?"
Resposta: "Não tenho informações necessárias para responder sua pergunta."

PERGUNTA DO USUÁRIO:
{pergunta}

RESPONDA A "PERGUNTA DO USUÁRIO"
"""

import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_postgres import PGVector

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
COLLECTION_NAME = os.getenv("PG_VECTOR_COLLECTION_NAME", "documentos")
EMBEDDING_MODEL = os.getenv("GOOGLE_EMBEDDING_MODEL", "models/gemini-embedding-001")
CHAT_MODEL = os.getenv("GOOGLE_CHAT_MODEL", "gemini-3.5-flash-lite")

TOP_K = 10


def _get_store():
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)
    return PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=DATABASE_URL,
        use_jsonb=True,
    )


def search_prompt(question=None):
    if not DATABASE_URL:
        print("Variável DATABASE_URL não configurada. Copie .env.example para .env.")
        return None
    if not os.getenv("GOOGLE_API_KEY"):
        print("Variável GOOGLE_API_KEY não configurada. Copie .env.example para .env.")
        return None

    store = _get_store()
    llm = ChatGoogleGenerativeAI(model=CHAT_MODEL, temperature=0)
    prompt = PromptTemplate(
        input_variables=["contexto", "pergunta"],
        template=PROMPT_TEMPLATE,
    )

    def answer(user_question: str) -> str:
        results = store.similarity_search_with_score(user_question, k=TOP_K)
        contexto = "\n\n".join(doc.page_content for doc, _score in results)
        mensagem = prompt.format(contexto=contexto, pergunta=user_question)
        resposta = llm.invoke(mensagem)
        return resposta.content.strip()

    # Uso direto (ex.: python src/search.py) devolve a resposta;
    # sem argumento, devolve a função para o CLI reutilizar a conexão.
    if question:
        return answer(question)
    return answer


if __name__ == "__main__":
    resultado = search_prompt("Qual o faturamento da Empresa SuperTechIABrazil?")
    print(resultado)
