# Ingestão e Busca Semântica com LangChain e Postgres

Software lê um PDF, guarda seus
trechos como vetores em uma base no PostgreSQL com pgVector e responde perguntas via CLI.

## Tecnologias

- Python + LangChain
- PostgreSQL + pgVector (via Docker Compose)
- Gemini (Google) — embeddings `models/gemini-embedding-001` e LLM `gemini-3.5-flash-lite`

## Pré-requisitos

- Docker e Docker Compose
- Python 3.12+
- Uma API Key do Google (Gemini)

## Configuração

1. Crie e ative o ambiente virtual:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

2. Instale as dependências:

   ```bash
   pip install -r requirements.txt
   ```

3. Copie o arquivo de variáveis de ambiente e preencha a sua chave:

   ```bash
   cp .env.example .env
   ```

   Edite o `.env` e informe `GOOGLE_API_KEY`. Os demais valores já vêm com
   padrões compatíveis com o `docker-compose.yml`.

## Execução

1. Suba o banco de dados (Postgres + extensão vector):

   ```bash
   docker compose up -d
   ```

2. Faça a ingestão do PDF (`document.pdf` na raiz do projeto):

   ```bash
   python src/ingest.py
   ```

   O PDF é dividido em chunks de 1000 caracteres com overlap de 150, convertido
   em embeddings e gravado na coleção configurada. A cada execução a coleção é
   recriada, evitando duplicação de dados.

3. Rode o chat no terminal:

   ```bash
   python src/chat.py
   ```

## Exemplo de uso

```
PERGUNTA: Qual o faturamento da Empresa SuperTechIABrazil?
RESPOSTA: O faturamento foi de 10 milhões de reais.

----------------------------------------

PERGUNTA: Quantos clientes temos em 2024?
RESPOSTA: Não tenho informações necessárias para responder sua pergunta.
```

Perguntas cuja resposta não esteja explicitamente no PDF retornam sempre:
`Não tenho informações necessárias para responder sua pergunta.`

## Estrutura

```
├── docker-compose.yml     # Postgres + pgVector
├── requirements.txt       # Dependências
├── .env.example           # Template das variáveis de ambiente
├── src/
│   ├── ingest.py          # Ingestão do PDF no banco vetorial
│   ├── search.py          # Busca (k=10) + montagem do prompt + LLM
│   └── chat.py            # CLI de perguntas e respostas
├── document.pdf           # PDF de origem
└── README.md
```

## Como funciona

- **Ingestão** (`ingest.py`): `PyPDFLoader` lê o PDF, o
  `RecursiveCharacterTextSplitter` gera os chunks, o modelo de embeddings do
  Gemini os vetoriza e o `PGVector` persiste tudo no Postgres.
- **Busca** (`search.py`): a pergunta é vetorizada e comparada aos vetores
  armazenados via `similarity_search_with_score(query, k=10)`. Os 10 trechos
  mais relevantes são concatenados como CONTEXTO e enviados ao LLM dentro de um
  prompt que o obriga a responder somente com base nesse contexto.
- **CLI** (`chat.py`): laço de perguntas e respostas no terminal. Digite `sair`
  para encerrar.
