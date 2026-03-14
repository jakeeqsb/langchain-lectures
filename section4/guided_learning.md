# Section 4 — Guided 학습 노트: ReAct Agent + 구조화 출력

> **학습 방식**: 개념 이해 → 예시 코드 확인 → 직접 실습 → 확인 순으로 진행한다.
> **참고 노트**: `section4/1.lecture_note.md` (ReAct Agent), `section4/2.lecture_note.md` (Output Parsing)
> **실제 예시 코드**: `main.py` (Agent Loop 완성본)
> **사용 패키지**: `langchain`, `langchain-openai`, `langchain-tavily`, `langsmith`, `pydantic`

---

## STAGE 1. LangChain Tool — LLM이 외부 API를 사용하는 방법

### 개념

LLM은 외부 API를 직접 호출할 수 없다.
**Tool**은 "어떤 외부 기능이 있고, 언제·어떻게 호출하면 되는지"를 LLM이 이해할 수 있는 스키마로 감싸는 래퍼 객체다.

| Tool 구성 요소 | 역할 |
|--------------|------|
| `name` | LLM이 Tool을 식별하는 문자열 |
| `description` | LLM이 **언제** 이 Tool을 쓸지 판단하는 기준 |
| `args schema` | LLM이 **어떤 인자**로 호출할지 결정하는 JSON 스키마 |

### 예시 코드 — 공식 통합 Tool (TavilySearch)

```python
from langchain_tavily import TavilySearch

# TavilySearch는 LangChain Tool 인터페이스를 구현한 공식 통합 객체
tools = [TavilySearch()]

# Tool 메타데이터 확인
t = tools[0]
print(t.name)         # "tavily_search_results_json"
print(t.description)  # "A search engine optimized for..."
print(t.args)         # {"query": {...}, "include_domains": {...}, ...}
```

### 예시 코드 — 커스텀 Tool (@tool)

```python
from langchain.tools import tool

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog.
    제품 카탈로그에서 가격을 조회한다."""
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)
```

> 🔍 **내부 동작 원리**: Tool이 LLM에 전달하는 정보 흐름
>
> ```text
> Tool 객체 메타데이터
>   ├─ name        → LLM이 호출할 Tool 식별자 (함수명에서 추출)
>   ├─ description → LLM이 "언제" 사용할지 판단 (docstring에서 추출)
>   └─ args schema → LLM이 "어떤 인자"로 호출할지 결정 (타입 힌팅에서 추출)
>
> 실행 흐름:
>   LLM → Tool 이름 + 인자 "결정"만 함 (추론)
>   AgentExecutor → 실제 tool._run(args) 호출 (실행)
>   ⚠️ LLM은 API를 직접 호출하지 않는다
> ```

> 💡 **실무 포인트**: `description`과 `args` 설명이 명확할수록 LLM Tool 선택 정확도가 높아진다.
> 공식 통합 패키지가 있다면 직접 만들기 전에 먼저 확인하라 — 해당 팀이 LLM 친화적으로 설명을 작성해두었다.

> ⚠️ `@tool`로 커스텀 Tool을 만들 때 타입 힌팅과 docstring은 **필수**다.
> 없으면 LLM이 받는 스키마가 불완전해져 잘못된 인자로 호출한다.

🔗 [LangChain Tools 개념 문서](https://python.langchain.com/docs/concepts/tools/)
🔗 [TavilySearch 통합 패키지](https://python.langchain.com/docs/integrations/tools/tavily_search/)

### 🛠️ 실습 1

```python
# section4/practice/stage1_tools.py
from langchain_tavily import TavilySearch

t = TavilySearch()

# TODO: Tool의 3가지 핵심 속성을 출력하라
print("name:", t.___)
print("description:", t.___)
print("args:", t.___)
```

✅ **확인 사항**: `description`에 검색엔진 설명, `args`에 `query` 필드가 있어야 함

---

## STAGE 2. LLM 설정 — `ChatOpenAI` + `temperature=0`

### 개념

에이전트에서 LLM 설정 시 `temperature=0`이 거의 필수인 이유:

- 랜덤성이 있으면 동일한 질문에 다른 Tool을 선택하거나 다른 순서로 실행
- **재현 불가능한 버그** 발생 → 디버깅 극도로 어려움

```python
from langchain_openai import ChatOpenAI

# temperature=0: 매 실행마다 동일한 결과 보장 (결정론적)
llm = ChatOpenAI(model="gpt-4o", temperature=0)
```

또는 `init_chat_model`로 모델 교체를 유연하게:

```python
from langchain.chat_models import init_chat_model

# 개발 중: Ollama 로컬 모델 (무료)
llm = init_chat_model("ollama:qwen2.5:7b", temperature=0)

# 프로덕션: 이 한 줄만 바꾸면 됨
# llm = init_chat_model("openai:gpt-4o", temperature=0)
```

> ⚠️ **GPT-5 주의**: `ChatOpenAI(model="gpt-5")`로 ReAct Agent 실행 시
> `stop` 파라미터 미지원 → `400 Bad Request` 에러 발생.
> ReAct 알고리즘이 `stop` 인자를 사용하기 때문이다. **GPT-4/4o 사용 권장.**

🔗 [ChatOpenAI API 레퍼런스](https://python.langchain.com/api_reference/openai/chat_models/langchain_openai.chat_models.base.ChatOpenAI.html)

### 🛠️ 실습 2

```python
# stage2_llm.py
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage

llm = init_chat_model("ollama:qwen2.5:7b", temperature=0)

# TODO: 동일한 질문을 3번 반복 호출해 응답이 동일한지 확인
for i in range(3):
    response = llm.invoke([HumanMessage(content="1 + 1 = ?")])
    print(f"Run {i+1}: {response.___}")   # 응답 텍스트 속성?
```

✅ **확인 사항**: 3번 모두 동일한 답변이 나와야 함 (`temperature=0` 효과)

---

## STAGE 3. ReAct 프롬프트 — 에이전트 추론의 핵심

### 개념

ReAct(**Re**asoning + **Act**ing) 프롬프트는 LLM에게 "Thought → Action → Observation" 사이클로 추론하도록 강제하는 프롬프트 템플릿이다.

```python
from langchain import hub

# LangChain Hub에서 공개 ReAct 프롬프트 로드
react_prompt = hub.pull("hwchase17/react")

# 입력 변수 확인
print(react_prompt.input_variables)
# ['input', 'tools', 'tool_names', 'agent_scratchpad']
```

### ReAct 프롬프트 구조

```text
Answer the following questions as best you can.
You have access to the following tools:
{tools}                          ← Tool 목록 자동 삽입

Use the following format:
Question: the input question
Thought: you should think about what to do
Action: the action to take, one of [{tool_names}]  ← Tool 이름 목록
Action Input: the input to the action
Observation: the result of the action
... (Thought/Action/Observation 반복)
Thought: I now know the final answer
Final Answer: the final answer

Question: {input}                ← 사용자 질문
Thought: {agent_scratchpad}      ← Tool 호출 기록 누적
```

| 변수 | 채우는 주체 | 내용 |
|------|-----------|------|
| `{input}` | 사용자 | 원래 질문 |
| `{tools}` | LangChain 자동 추출 | Tool 이름 + 설명 전체 |
| `{tool_names}` | LangChain 자동 추출 | Tool 이름 목록만 |
| `{agent_scratchpad}` | AgentExecutor 관리 | Thought/Action/Observation **누적 기록** |

> 🔍 **내부 동작 원리**: `agent_scratchpad`가 에이전트 "기억"의 원천
>
> ```text
> 1차 LLM 호출 시 scratchpad: "" (비어있음)
>
> Tool 실행 후 scratchpad:
>   "Thought: 검색이 필요하다
>    Action: tavily_search_results_json
>    Action Input: AI engineer LangChain Bay Area
>    Observation: [{'url': '...', 'content': 'Job 1: ...'}]"
>
> 2차 LLM 호출 시 이 scratchpad 전체가 프롬프트에 삽입됨
> → LLM이 "나는 이미 검색했고 결과가 있다"는 것을 기억함
> → "Final Answer:" 생성 가능
>
> Tool 호출이 많을수록 scratchpad가 길어짐 → 토큰 비용 급증
> ```

> ⚠️ **EU 엔드포인트 사용 시 `hub.pull` 실패**:
>
> ```python
> # 해결 방법 1: 환경변수 일시 제거
> import os
> os.environ.pop("LANGCHAIN_ENDPOINT", None)
>
> # 해결 방법 2: 프롬프트 직접 생성 (위 템플릿 복붙)
> from langchain_core.prompts import PromptTemplate
> react_prompt = PromptTemplate.from_template("...")
> ```

> 💡 **실무 포인트**: ReAct는 Cursor, Claude Code, Devin 등 현대 AI 에이전트 기술의 근간이다.
> 이 메커니즘을 이해하면 어떤 에이전트 프레임워크든 내부를 읽을 수 있다.

🔗 [ReAct 논문 (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)
🔗 [LangChain Hub - hwchase17/react](https://smith.langchain.com/hub/hwchase17/react)

### 🛠️ 실습 3

```python
# stage3_react_prompt.py
from langchain import hub

react_prompt = hub.pull("hwchase17/react")

# TODO: 입력 변수 목록 출력
print(react_prompt.___)

# TODO: 프롬프트 템플릿 전문 출력
print(react_prompt.___)
```

✅ **확인 사항**: `input_variables`에 4개 변수, 템플릿에 `{agent_scratchpad}` 포함

---

## STAGE 4. `create_react_agent`와 `AgentExecutor` — 추론과 실행의 분리

### 개념

| | `create_react_agent` 반환값 | `AgentExecutor` |
|--|----------------------------|--------------------|
| **타입** | Runnable (Chain) | AgentExecutor 객체 |
| **역할** | 추론만 담당 ("다음에 뭘 해야 해?") | 실행 담당 (Tool 실제 호출, 루프) |
| **Tool 실행** | ❌ 하지 않음 | ✅ 직접 실행 |
| **while 루프** | ❌ 없음 | ✅ 있음 |

> ⚠️ **흔한 오해**: `create_react_agent`가 반환하는 `agent` 변수는 이름과 달리 단순한 **Chain**이다.
> "에이전트"다운 지능적 동작(루프, Tool 실행)은 전부 `AgentExecutor`가 처리한다.

### 예시 코드

```python
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o", temperature=0)
react_prompt = hub.pull("hwchase17/react")

# create_react_agent: 추론 엔진 (Chain)
# 내부: react_prompt | llm | OutputParser
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

# AgentExecutor: 실행 오케스트레이터
agent_executor = AgentExecutor(
    agent=agent,      # 추론 체인
    tools=tools,      # 실제 실행할 Tool 목록
    verbose=True,     # Thought/Action/Observation 로그 출력
)
chain = agent_executor
```

> 🔍 **내부 동작 원리**: `AgentExecutor`의 while 루프 의사코드
>
> ```text
> while True:
>     # 추론 엔진 호출 → 다음 행동 결정
>     output = agent.invoke({
>         "input": question,
>         "agent_scratchpad": scratchpad  # 지금까지의 기록
>     })
>
>     if isinstance(output, AgentFinish):   # "Final Answer:" 감지
>         return output.return_values["output"]
>
>     if isinstance(output, AgentAction):   # Tool 호출 필요
>         observation = tools_dict[output.tool].run(output.tool_input)
>
>         # scratchpad에 기록 추가 → 다음 반복에서 LLM이 기억
>         scratchpad += f"\nAction: {output.tool}\n..."
> ```

> 💡 **실무 포인트**: `chain.invoke({...})`는 dict를 **위치 인자**로 전달해야 한다.
>
> ```python
> chain.invoke(input={"input": "..."})   # ❌ 키워드 인자 불가
> chain.invoke({"input": "..."})         # ✅
> ```

🔗 [create_react_agent API 레퍼런스](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.react.agent.create_react_agent.html)
🔗 [AgentExecutor API 레퍼런스](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html)

### 🛠️ 실습 4-A: 실행하고 verbose 로그 분석

```python
# stage4_react_agent.py
from dotenv import load_dotenv
load_dotenv()

from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o", temperature=0)
react_prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

result = agent_executor.invoke({"input": "What is the capital of South Korea?"})
print("\n최종 결과:", result["output"])  # dict의 어떤 키에 답이?
```

✅ **확인 사항**:

- verbose 로그에 `Thought:` → `Action:` → `Observation:` → `Final Answer:` 순서 확인
- `result`는 dict이고 `result["output"]`에 최종 답 있음

### 🛠️ 실습 4-B: 실험

```python
# 실험 1: verbose=False로 변경
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)
# → 중간 로그는 사라지고 result["output"]만 확인 가능

# 실험 2: 잘못된 호출 방식 확인
result = agent_executor.invoke(input={"input": "..."})  # ❌
# → 어떤 에러가 발생하는가?

# 실험 3: max_iterations=1 설정
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=1)
result = agent_executor.invoke({"input": "What's the latest AI news? Give 3 examples."})
# → Tool 한 번만 호출하고 중단되는 현상 관찰
```

---

## STAGE 5. Pydantic 스키마 설계 — 구조화 출력의 설계도

### 개념

LLM의 텍스트 출력을 Python 객체로 받으려면 **어떤 형식의 데이터를 원하는가**를 먼저 Pydantic으로 정의해야 한다.

### 예시 코드

```python
# schemas.py
from typing import List
from pydantic import BaseModel, Field

class Source(BaseModel):
    """에이전트가 사용한 출처 소스 스키마"""
    url: str = Field(description="출처 URL")

class AgentResponse(BaseModel):
    """에이전트의 최종 응답 스키마 (답변 + 출처 목록)"""
    answer: str = Field(description="에이전트의 최종 답변")
    sources: List[Source] = Field(
        description="답변 생성에 사용한 출처 목록",
        default_factory=list  # ⚠️ default=[] 아님 (공유 참조 버그 방지)
    )
```

> 🔍 **내부 동작 원리**: Pydantic이 LLM에 전달하는 JSON 스키마
>
> ```json
> {
>   "title": "AgentResponse",
>   "properties": {
>     "answer": {"type": "string", "description": "에이전트의 최종 답변"},
>     "sources": {
>       "type": "array",
>       "items": {"$ref": "#/definitions/Source"},
>       "description": "답변 생성에 사용한 출처 목록"
>     }
>   }
> }
> ```
>
> 이 스키마가 프롬프트 또는 Function Calling API를 통해 LLM에 전달됨
> → LLM이 이 형식에 맞는 JSON을 생성

> 💡 **실무 포인트**: `Field(description="...")` 품질이 LLM 출력 정확도에 직결된다.
> 특히 **중첩 구조(nested)**에서 각 필드의 역할을 명확히 기술해야 LLM이 올바른 위치에 데이터를 채운다.

> ⚠️ **Python mutable 기본값 버그**:
> `default=[]`를 쓰면 모든 인스턴스가 동일한 리스트를 공유 → `default_factory=list` 사용.

🔗 [Pydantic 공식 문서](https://docs.pydantic.dev/latest/)
🔗 [LangChain Structured Outputs](https://python.langchain.com/docs/concepts/structured_outputs/)

### 🛠️ 실습 5

```python
# stage5_pydantic.py
from typing import List
from pydantic import BaseModel, Field

# TODO: 직접 JobPosting 스키마를 설계해보라
# 포함할 필드: title(str), company(str), location(str), url(str)
class JobPosting(BaseModel):
    """채용 공고 스키마"""
    title: str = Field(description="___")
    company: str = Field(description="___")
    location: str = Field(description="___")
    url: str = Field(description="___")

class JobSearchResponse(BaseModel):
    """채용 공고 검색 결과 스키마"""
    query: str = Field(description="검색 쿼리")
    jobs: List[JobPosting] = Field(description="___", default_factory=___)

# TODO: 스키마 확인
print(JobSearchResponse.model_json_schema())
```

✅ **확인 사항**: `model_json_schema()`에 `jobs` 필드가 array 타입으로 표시

---

## STAGE 6. 구조화 출력 방법 1 — `PydanticOutputParser` (구방식)

### 개념

프롬프트에 "이 JSON 형식으로 답해줘"라는 지침을 직접 삽입해 구조화 출력을 유도한다.

### 예시 코드

```python
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda

# 1. 출력 파서 생성
output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

# format_instructions 출력해서 LLM에 전달되는 텍스트 확인
print(output_parser.get_format_instructions())

# 2. 커스텀 ReAct 프롬프트에 {format_instructions} 플레이스홀더 삽입
REACT_WITH_FORMAT = """...
Final Answer: the final answer formatted according to the format instructions
{format_instructions}

Question: {input}
Thought: {agent_scratchpad}"""

react_prompt_with_format = PromptTemplate(
    template=REACT_WITH_FORMAT,
    input_variables=["input", "agent_scratchpad", "tool_names"],
).partial(format_instructions=output_parser.get_format_instructions())

# 3. LCEL 파이프라인 조합
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt_with_format)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

extract_output = RunnableLambda(lambda x: x["output"])            # dict → 문자열
parse_output   = RunnableLambda(lambda x: output_parser.parse(x)) # 문자열 → Pydantic

chain = agent_executor | extract_output | parse_output

result: AgentResponse = chain.invoke({"input": "..."})
print(result.answer)           # 구조화된 답변
print(result.sources[0].url)   # 출처 URL
```

> 🔍 **내부 동작 원리**: LCEL `|` 파이프 실행 흐름
>
> ```text
> chain.invoke({"input": "query"})
>
>   Step 1: agent_executor.invoke(...)
>     → {"input": "query", "output": '{"answer": "...", "sources": [...]}'}
>
>   Step 2: extract_output.invoke(step1_result)
>     → '{"answer": "...", "sources": [...]}'   (JSON 문자열)
>
>   Step 3: parse_output.invoke(step2_result)
>     → AgentResponse(answer="...", sources=[Source(url="...")])
>
> 핵심: 각 Runnable은 이전 출력을 입력으로 받음
> RunnableLambda: 일반 Python 함수를 Runnable로 래핑하는 가장 간단한 방법
> ```

> ⚠️ **구방식의 단점**:
>
> - Format Instructions가 **모든 반복 프롬프트**에 포함됨 → 불필요한 토큰 낭비
> - LLM 모델 능력에 따라 JSON 형식을 어겨 `parse()` 에러 가능

🔗 [PydanticOutputParser API](https://python.langchain.com/api_reference/langchain/output_parsers/langchain.output_parsers.pydantic.PydanticOutputParser.html)
🔗 [RunnableLambda API](https://python.langchain.com/api_reference/core/runnables/langchain_core.runnables.base.RunnableLambda.html)
🔗 [LCEL 개념 문서](https://python.langchain.com/docs/concepts/lcel/)

### 🛠️ 실습 6

```python
# stage6_pydantic_parser.py
from langchain_core.output_parsers.pydantic import PydanticOutputParser

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)

# TODO: format_instructions 내용을 출력해 LLM에 전달되는 텍스트 확인
print(output_parser.get_format_instructions())

# TODO: RunnableLambda 안에 print를 넣어 각 단계 타입 확인
from langchain_core.runnables import RunnableLambda

extract = RunnableLambda(lambda x: (print(f"[extract] type={type(x)}"), x["output"])[1])
parse   = RunnableLambda(lambda x: (print(f"[parse] type={type(x)}"), output_parser.parse(x))[1])
```

✅ **확인 사항**: `extract` 단계는 `dict` 입력, `parse` 단계는 `str` 입력 확인

---

## STAGE 7. 구조화 출력 방법 2 — `with_structured_output` (신방식, 권장)

### 개념

Function Calling API를 활용해 LLM이 반드시 스키마에 맞는 JSON을 생성하도록 **API 레벨에서** 강제한다.

```python
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# structured_llm: AgentResponse 형식만 반환하는 특수 모델 인스턴스
structured_llm = llm.with_structured_output(AgentResponse)

# ⚠️ 중요: 추론 체인(create_react_agent)에는 일반 llm 사용
#          마지막 파이프 단계에서만 structured_llm 사용
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

extract_output = RunnableLambda(lambda x: x["output"])

# parse_output 대신 structured_llm이 마지막에 구조화
chain = agent_executor | extract_output | structured_llm

result: AgentResponse = chain.invoke({"input": "..."})
print(result.answer)
print(result.sources[0].url)
```

> 🔍 **내부 동작 원리**: `with_structured_output` 동작 방식
>
> ```text
> llm.with_structured_output(AgentResponse):
>   1. AgentResponse 스키마를 Function Calling용 tool 정의로 변환
>   2. invoke() 시 API payload의 "tools" 파라미터에 스키마 자동 포함
>   3. LLM은 Function Calling으로 스키마에 맞는 JSON 반드시 생성
>      (프롬프트 지침이 아닌 API 레벨 강제)
>   4. LangChain이 자동으로 PydanticOutputParser 적용
>      → AgentResponse 인스턴스 반환
>
> LangSmith 트레이스 차이:
>   구방식: 모든 반복 프롬프트에 format_instructions 텍스트 포함 ↑
>   신방식: 마지막 structured_llm 호출에서만 "tools"에 스키마 전송 ↓ (토큰 절약)
> ```

> ⚠️ `create_react_agent`에 `structured_llm`을 실수로 넘기면?
> → 모든 반복에서 AgentResponse 형식을 강제 적용하려 해 Tool 선택 단계가 망가짐.
> **추론에는 일반 `llm`, 마지막 파이프에만 `structured_llm`.**

🔗 [with_structured_output 공식 문서](https://python.langchain.com/docs/how_to/structured_output/)
🔗 [Function Calling 개념 문서](https://python.langchain.com/docs/concepts/tool_calling/)

---

## STAGE 8. 두 방식 비교 + LangSmith로 검증

### 두 방식 전체 비교

| 항목 | PydanticOutputParser (구방식) | with_structured_output (신방식) |
|-----|----------------------------|---------------------------------|
| **구조화 메커니즘** | 프롬프트 텍스트 지침 주입 | Function Calling API 레벨 강제 |
| **신뢰성** | 낮음 (파싱 에러 가능) | 높음 (API가 형식 보장) |
| **토큰 효율** | 낮음 (모든 반복에 스키마) | 높음 (마지막에만 스키마) |
| **모델 지원** | 모든 LLM | Function Calling 지원 모델 |
| **코드 복잡도** | 높음 (파서+프롬프트 수정) | 낮음 (한 줄) |
| **현재 권장** | ❌ 레거시 (학습 목적) | ✅ **프로덕션 권장** |

### 🛠️ 실습 8 — 두 방식을 직접 실행하고 LangSmith에서 비교

```python
# stage8_comparison.py
from dotenv import load_dotenv
load_dotenv()

# --- 공통 설정 ---
from langchain import hub
from langchain.agents import AgentExecutor
from langchain.agents.react.agent import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.runnables import RunnableLambda
from schemas import AgentResponse  # Source, AgentResponse 정의

tools = [TavilySearch()]
llm = ChatOpenAI(model="gpt-4o", temperature=0)
react_prompt = hub.pull("hwchase17/react")
question = {"input": "Find 2 LangChain AI engineer jobs in the Bay Area."}

# --- 방식 1: PydanticOutputParser ---
from langchain_core.output_parsers.pydantic import PydanticOutputParser

output_parser = PydanticOutputParser(pydantic_object=AgentResponse)
agent1 = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
executor1 = AgentExecutor(agent=agent1, tools=tools, verbose=True)
chain1 = executor1 | RunnableLambda(lambda x: x["output"]) | RunnableLambda(output_parser.parse)

# --- 방식 2: with_structured_output ---
structured_llm = llm.with_structured_output(AgentResponse)
agent2 = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)
executor2 = AgentExecutor(agent=agent2, tools=tools, verbose=True)
chain2 = executor2 | RunnableLambda(lambda x: x["output"]) | structured_llm

# 실행 (LangSmith 활성화 후 트레이스 비교)
result1 = chain1.invoke(question)
result2 = chain2.invoke(question)
```

✅ **LangSmith에서 확인할 것**:

- 방식 1: 모든 LLM 호출 프롬프트에 `format_instructions` 텍스트 포함 여부
- 방식 2: 마지막 `structured_llm` 호출에서만 `tools` 파라미터에 스키마 전송
- 두 방식의 총 토큰 수 차이 비교

---

## 최종 실습 — Section 4 전체를 처음부터 구현하기

아래 스펙을 보고 **노트 참고 없이** 처음부터 구현하라.

```text
스펙:
- Tool: TavilySearch (공식 통합 패키지 사용)
- LLM: ChatOpenAI(model="gpt-4o", temperature=0)
- 프롬프트: hub.pull("hwchase17/react")
- 구조화 출력 스키마:
    JobPosting(title, company, location, url)
    JobResponse(query: str, jobs: List[JobPosting])
- 출력 방식: with_structured_output (신방식)
- 질문: "Find 3 LangChain AI engineer job postings in the Bay Area"
- LangSmith 트레이싱 적용
```

```python
# section4/practice/final_agent.py
# TODO: 처음부터 직접 구현
```

✅ **확인 사항**:

- `result.jobs`가 `List[JobPosting]`으로 반환
- `result.jobs[0].url`로 첫 번째 공고 URL 접근 가능
- LangSmith에서 Tool 실행 → Final Answer → structured_llm 흐름 확인

---

## 핵심 개념 요약

```text
[ReAct Agent 구조]
✅ Tool = name + description + args schema (LLM의 도구 사용 설명서)
✅ LLM은 Tool 선택·인자 결정만 (추론), AgentExecutor가 실제 실행
✅ react_prompt = Thought/Action/Observation 사이클 강제하는 프롬프트 템플릿
✅ agent_scratchpad = Tool 호출 기록 누적 → LLM 기억의 원천
✅ create_react_agent 반환값 = Chain (추론만), AgentExecutor = 실행 오케스트레이터
✅ chain.invoke({...}) → dict를 위치 인자로 전달 (키워드 인자 불가)
✅ verbose=True + LangSmith 동시 사용으로 실행 흐름 완전 파악

[구조화 출력]
✅ Pydantic BaseModel = LLM 구조화 출력의 설계도
✅ Field(description="...") 품질 = LLM 출력 정확도에 직결
✅ default_factory=list → mutable 기본값 공유 버그 방지

✅ [구방식] PydanticOutputParser + LCEL 파이프 조립 → 모든 반복에 스키마, 파싱 에러 가능
✅ [신방식] with_structured_output → Function Calling, 마지막 단계만 스키마, 프로덕션 권장

✅ structured_llm은 마지막 파이프에만 — create_react_agent에 넘기면 오동작

[주의]
✅ temperature=0 → 에이전트 결정론적 동작 필수
✅ GPT-5는 stop 파라미터 미지원 → ReAct 방식 호환 불가
✅ Tool 호출 많을수록 scratchpad 길어짐 → 토큰 비용 급증 → max_iterations 설정 필수
```

---

## 실습 체크리스트

- [ ] `TavilySearch()` Tool의 `name`, `description`, `args` 직접 출력해 확인
- [ ] `hub.pull("hwchase17/react")`의 `input_variables`와 템플릿 전문 출력
- [ ] `verbose=True`로 실행 후 Thought/Action/Observation 사이클 개수 세기
- [ ] `max_iterations=1`로 설정 → 복잡한 질문에서 중간 중단 관찰
- [ ] `output_parser.get_format_instructions()` 출력 → LLM에 실제 전달되는 텍스트 확인
- [ ] PydanticOutputParser 방식 실행 → LangSmith에서 모든 반복에 스키마 포함 확인
- [ ] `with_structured_output` 방식으로 교체 → LangSmith에서 마지막에만 스키마 확인
- [ ] `create_react_agent`에 실수로 `structured_llm` 전달 → 오동작 현상 관찰
- [ ] RunnableLambda 안에 `print(type(x))`를 넣어 각 단계 입출력 타입 확인
- [ ] `default_factory=list` 대신 `default=[]` 사용 → Python 공유 참조 버그 재현

---

*참고: `section4/1.lecture_note.md` (ReAct Agent), `section4/2.lecture_note.md` (Output Parsing)*
*패키지: `pip install langchain langchain-openai langchain-tavily langsmith pydantic python-dotenv`*
