from search import search_prompt


def main():
    chain = search_prompt()

    if not chain:
        print("Não foi possível iniciar o chat. Verifique os erros de inicialização.")
        return

    print("Chat iniciado. Digite sua pergunta (ou 'sair' para encerrar).\n")

    while True:
        try:
            pergunta = input("PERGUNTA: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEncerrando.")
            break

        if not pergunta:
            continue
        if pergunta.lower() in {"sair", "exit", "quit"}:
            print("Encerrando.")
            break

        try:
            resposta = chain(pergunta)
        except Exception as exc:
            print(f"Ocorreu um erro ao processar a pergunta: {exc}\n")
            continue

        print(f"RESPOSTA: {resposta}\n")
        print("-" * 40 + "\n")


if __name__ == "__main__":
    main()
