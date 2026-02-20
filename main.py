from dotenv import load_dotenv

load_dotenv()

from langchain import hub
from langchain_classic.agents import AgentExecutor
from langchain_classic.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4", temperature=0)


def main():
    print("Hello from langchain-session!")


if __name__ == "__main__":
    main()
