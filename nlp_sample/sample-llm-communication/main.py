from agent import ChatAgent


def main() -> None:
    model_name = "MBZUAI/LaMini-T5-61M"

    agent = ChatAgent(model_name)

    while True:
        print("you  : ", end="", flush=True)
        question = input().strip()
        if question in ["bye", "exit", "quit"]:
            break
        answer = agent.chat(question)
        print(f"agent: {answer}", flush=True)

    print("DONE")


if __name__ == "__main__":
    main()
