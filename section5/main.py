from typing import List, Union

from dotenv import load_dotenv

# LangChain 패키지 버전에 따라 import 경로가 다를 수 있습니다.
from langchain.agents.format_scratchpad import format_log_to_str
from langchain.agents.output_parsers import ReActSingleInputOutputParser
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool, render_text_description, tool
from langchain_openai import ChatOpenAI

# callback.py에서 생성한 클래스 로드
from callback import AgentCallbackHandler

load_dotenv()


@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"\n[Tool Execution] get_text_length enter with text='{text}'")
    # LLM이 전달할 수 있는 불필요한 따옴표나 줄바꿈 제거 (방어 로직)
    text = text.strip("'\n\"")
    return len(text)


def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    """AgentAction에서 넘겨준 tool 이름과 일치하는 실제 Tool 객체를 찾아 리턴"""
    for t in tools:
        if t.name == tool_name:
            return t
    raise ValueError(f"Tool with name {tool_name} not found")


if __name__ == "__main__":
    tools = [get_text_length]

    # ReAct 프롬프트 템플릿 (hwchase17/react 규격)
    template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

    # 도구 정보(tools, tool_names)는 프롬프트에 미리 바인딩(partial)
    prompt = PromptTemplate.from_template(template=template).partial(
        tools=render_text_description(tools),
        tool_names=", ".join([t.name for t in tools]),
    )

    # LLM 객체 생성 - 할루시네이션 방지용 stop 토큰과 로깅용 callback 주입
    llm = ChatOpenAI(
        temperature=0,
        stop=["\nObservation", "Observation"],
        callbacks=[AgentCallbackHandler()],
    )

    # 1. 입력 변수 매핑(scratchpad 변환 포함) -> 2. 프롬프트 -> 3. LLM -> 4. 출력 파서
    agent = (
        {
            "input": lambda x: x["input"],
            "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
        }
        | prompt
        | llm
        | ReActSingleInputOutputParser()
    )

    # --- Agent 루프(AgentExecutor) 로직 ---
    intermediate_steps = []
    agent_step = ""

    # 1. 파싱 결과가 AgentFinish(최종 답변 도달)가 아닐 동안 무한 반복
    while not isinstance(agent_step, AgentFinish):
        # 에이전트 연쇄(체인) 실행!
        # - 처음엔 intermediate_steps가 비어 있음.
        # - 두 번째부터는 Action과 Observation 기록이 텍스트로 치환되어 주입됨.
        agent_step: Union[AgentAction, AgentFinish] = agent.invoke(
            {
                "input": "What is the length of the word: DOG",
                "agent_scratchpad": intermediate_steps,
            }
        )

        print(f"\n[Agent Step 파싱 결과]\n{agent_step}")

        # 2. 결과가 Action(도구 사용 필요)인 경우
        if isinstance(agent_step, AgentAction):
            tool_name = agent_step.tool
            tool_to_use = find_tool_by_name(tools, tool_name)
            tool_input = agent_step.tool_input

            # 도구 실제 실행
            observation = tool_to_use.func(str(tool_input))
            print(f"[Observation 결과] {observation=}")

            # 3. 이번 루프의 행동과 결과를 히스토리에 기록하여 다음 루프에 전달
            intermediate_steps.append((agent_step, str(observation)))

    # 4. 루프 종료 (AgentFinish 도달 시)
    if isinstance(agent_step, AgentFinish):
        print("\n🎉 최종 답변 도달!")
        print(agent_step.return_values)
