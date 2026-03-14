# LangChain 강의 노트 — Section 3: Agent 내부 구현 (인트로)

> **강의 주제**: AI 에이전트 내부 동작을 레이어별로 직접 구현하며 완전히 이해하기
> **난이도**: 고급
> **목표**: "에이전트가 어떻게 작동하는가"를 가장 깊은 레벨에서 이해
> **사용 패키지**: `langchain`, `langchain-ollama`, `langchain-openai`, `langsmith`
> **권장 모델**: Ollama + `qwen2.5:7b` (Function Calling 지원 필수)

---

## 1. 이 섹션의 핵심 목표

> "라이브러리 **사용법**은 문서를 보면 된다.  
> **원리를 이해**하려면 직접 만들어봐야 한다."

섹션이 끝나면:

- 에이전트가 내부에서 어떻게 동작하는지 **정확히** 알게 된다
- 어떤 모델, 어떤 프레임워크든 **유연하게 적용**할 수 있게 된다
- 프로덕션에서 에이전트를 **디버깅·커스터마이징**할 수 있게 된다

> 🔍 **내부 동작 원리**: 에이전트의 본질은 단순한 `while` 루프다.
>
> ```text
> while True:
>     response = llm.invoke(messages)       # LLM에게 다음 행동 물어보기
>     if response에 tool_call이 있으면:
>         result = tool.invoke(args)        # 툴 실행
>         messages에 tool_call + result 추가    # 기억에 저장
>     else:
>         return response.content           # Final Answer → 종료
> ```
>
> `AgentExecutor`나 `create_react_agent`는 이 루프를 감싸는 **편의 래퍼**일 뿐이다.
> 내부를 직접 구현하면 이 진실을 체감할 수 있다.

> 💡 **실무 포인트**: 프로덕션에서 에이전트가 예상치 못한 동작을 할 때,
> 이 `while` 루프 관점으로 문제를 바라보면 원인을 훨씬 빠르게 찾을 수 있다.
> "LLM이 어떤 메시지를 받았고, 무엇을 결정했는가"를 역추적하면 된다.

🔗 [LangChain Agents 공식 문서](https://python.langchain.com/docs/concepts/agents/)

---

## 2. 레이어별 학습 계획

강의는 추상화 레이어를 **바깥에서 안으로** 벗겨가며 진행된다.

```text
Layer 0 (이전 섹션)
  └─ LangChain create_react_agent + AgentExecutor
     → LangChain이 모든 것을 처리, 내부 원리 모름

Layer 1 (이번 섹션 1단계)
  └─ Agent Loop 직접 구현 (LangChain 기본 요소 활용)
     → while 루프 + Function Calling
     → LangChain의 tool, bind_tools, ChatModel, ToolMessage 사용
     → 보일러플레이트 코드는 LangChain이 처리

Layer 2 (이번 섹션 2단계)
  └─ Agent Loop 직접 구현 (순수 구현, 프레임워크 없음)
     → JSON 스키마 직접 작성
     → Function Calling을 raw 방식으로 구현
     → LangChain의 가치를 직접 체감

Layer 3 (이번 섹션 3단계)
  └─ Function Calling 없이 Agent 구현 (ReAct 프롬프트 방식)
     → ReAct 프롬프트 + 정규식 + Scratchpad만 사용
     → 에이전트가 처음 등장했을 때의 구현 방식
     → 가장 깊은 수준의 이해
```

> 🔍 **내부 동작 원리**: 각 레이어가 감추는 것
>
> | 레이어 | 감추는 추상화 | 직접 다루는 것 |
> |--------|-------------|--------------|
> | Layer 0 | 루프 전체, Tool 실행, 프롬프트 | 결과(`result["output"]`)만 받음 |
> | Layer 1 | JSON 스키마, 모델별 메시지 포맷 | `while` 루프, `tool_calls` 파싱 |
> | Layer 2 | 없음 (모두 직접) | OpenAI raw API, JSON 스키마 직접 작성 |
> | Layer 3 | Function Calling 자체 없음 | 텍스트 파싱(`re`), Scratchpad 관리 |

> 💡 **실무 포인트**: Layer 0만 알고 실무에 투입되면, 에이전트가 무한 루프에 빠지거나
> 엉뚱한 Tool을 반복 호출할 때 속수무책이 된다. Layer 1~2를 직접 구현해야
> `max_iterations`, `handle_parsing_errors`, `early_stopping_method` 같은
> `AgentExecutor` 옵션들이 **왜 존재하는지** 이해된다.

🔗 [AgentExecutor API 레퍼런스](https://python.langchain.com/api_reference/langchain/agents/langchain.agents.agent.AgentExecutor.html)

---

## 3. 각 레이어 상세 설명

### Layer 1: LangChain 기본 요소로 Agent Loop 구현

```python
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

# 모델 초기화
llm = init_chat_model("ollama:qwen2.5:7b", temperature=0)

@tool
def my_tool(query: str) -> str:
    """도구 설명 — LLM이 이 docstring을 읽고 언제 사용할지 판단한다."""
    return f"결과: {query}"

tools = [my_tool]
llm_with_tools = llm.bind_tools(tools)

# 핵심: while 루프로 직접 에이전트 루프 구현
messages = [HumanMessage(content="질문")]
MAX_ITERATIONS = 10

for _ in range(MAX_ITERATIONS):
    ai_message = llm_with_tools.invoke(messages)
    if not ai_message.tool_calls:       # Final Answer
        print(ai_message.content)
        break
    # Tool 실행 후 결과를 messages에 추가
    tool_name = ai_message.tool_calls[0]["name"]
    result = my_tool.invoke(ai_message.tool_calls[0]["args"])
    messages.append(ai_message)
    messages.append(ToolMessage(content=str(result),
                                tool_call_id=ai_message.tool_calls[0]["id"]))
```

> 🔍 **내부 동작 원리**: `bind_tools()`가 하는 일
>
> ```text
> bind_tools([my_tool]) 호출 시:
>   1. @tool 데코레이터가 만든 JSON 스키마를 추출
>   2. 매 llm.invoke() 호출 시 이 스키마를 payload에 자동 삽입
>   3. 모델이 tool_call을 결정하면 → ai_message.tool_calls 에 파싱된 dict 삽입
>
> ToolMessage의 역할:
>   Tool 실행 결과를 대화 기록에 추가해야 LLM이 다음 호출에서 그 결과를 알 수 있다.
>   tool_call_id 를 반드시 일치시켜야 OpenAI 등 API가 정상 처리함.
> ```

> 💡 **실무 포인트**: `tool_call_id`를 잘못 매칭하면 일부 API(특히 OpenAI)에서
> `invalid_request_error`가 발생한다. `ai_message.tool_calls[i]["id"]`와
> `ToolMessage(tool_call_id=...)`를 반드시 1:1 매칭할 것.

> ⚠️ **주의**: `bind_tools()` 없이 `llm.invoke()`를 호출하면 모델이 Tool의 존재 자체를 모른다.
> `bind_tools()`를 빼먹는 실수는 매우 흔하다.

🔗 [bind_tools() API 레퍼런스](https://python.langchain.com/api_reference/core/language_models/langchain_core.language_models.chat_models.BaseChatModel.html#langchain_core.language_models.chat_models.BaseChatModel.bind_tools)

---

### Layer 2: 순수 구현 (프레임워크 없음)

```python
import openai
import json

client = openai.OpenAI()

# JSON 스키마 직접 작성 (LangChain @tool이 대신 해주던 것)
tools_schema = [{
    "type": "function",
    "function": {
        "name": "my_tool",
        "description": "도구 설명",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "검색어"}
            },
            "required": ["query"]
        }
    }
}]

# LangChain 없이 OpenAI API 직접 호출
response = client.chat.completions.create(
    model="gpt-4o",
    tools=tools_schema,
    messages=[{"role": "user", "content": "질문"}]
)
```

> 🔍 **내부 동작 원리**: LangChain `@tool`이 대신 해주는 것들
>
> ```text
> @tool 데코레이터 수행 작업:
>   1. 함수 이름        → "name"
>   2. docstring       → "description"
>   3. 타입 힌팅(str, float 등) → "parameters" JSON 스키마
>   4. Pydantic validation 추가
>
> bind_tools() 수행 작업:
>   1. 위 스키마를 API payload의 "tools" 필드에 자동 삽입
>   2. provider별(OpenAI, Anthropic, Ollama) 포맷 차이 추상화
> ```

> 💡 **실무 포인트**: Layer 2를 한 번이라도 직접 구현하면, 이후 LangChain이
> 왜 유용한지 체감할 수 있다. 특히 멀티 모델 전략(OpenAI ↔ Anthropic 스위칭)을
> 사용하는 프로덕션 환경에서 LangChain의 추상화 가치가 극명하게 드러난다.

> ⚠️ **주의**: OpenAI와 Anthropic은 tool_call 응답 포맷이 다르다. 프레임워크 없이
> 직접 구현 시 provider마다 파싱 코드를 별도로 작성해야 한다.

🔗 [OpenAI Function Calling 공식 문서](https://platform.openai.com/docs/guides/function-calling)

---

### Layer 3: ReAct 프롬프트 + 정규식 (Function Calling 없음)

```python
import re
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4", temperature=0)

# ReAct 형식의 프롬프트 — LLM이 텍스트로 행동을 출력하도록 유도
react_prompt = """다음 도구를 사용할 수 있습니다: {tools}

반드시 아래 형식으로 답하세요:
Thought: 무엇을 해야 할지 생각
Action: 사용할 도구 이름
Action Input: 도구에 전달할 입력
Observation: 도구 실행 결과 (시스템이 채움)
... (반복)
Thought: 이제 최종 답을 알았습니다
Final Answer: 최종 답변

질문: {input}
{agent_scratchpad}"""

# LLM의 텍스트 응답에서 Action 파싱
response_text = llm.invoke(react_prompt.format(...)).content

action = re.search(r"Action: (.+)", response_text).group(1).strip()
action_input = re.search(r"Action Input: (.+)", response_text).group(1).strip()

# Tool 실행 후 scratchpad에 추가
result = tool_map[action](action_input)
scratchpad += f"\nAction: {action}\nAction Input: {action_input}\nObservation: {result}\n"
```

> 🔍 **내부 동작 원리**: Function Calling 이전 시대의 에이전트
>
> ```text
> Function Calling이 없던 시절:
>   LLM 출력 = 순수 텍스트
>   → 개발자가 정규식으로 "Action:", "Action Input:"을 파싱
>   → 파싱된 도구 이름으로 Python 함수를 동적 호출
>   → 결과를 "Observation:"으로 프롬프트에 이어붙여 재호출
>
> Scratchpad의 역할:
>   "Thought → Action → Observation" 기록의 누적본.
>   LLM이 현재까지의 추론 과정 전체를 볼 수 있게 해줌.
>   LangChain AgentExecutor의 agent_scratchpad 변수가 이를 자동 관리함.
> ```

> 💡 **실무 포인트**: 현대 에이전트는 대부분 Function Calling을 사용하지만,
> **오픈 웨이트 모델 중 Function Calling을 지원하지 않는 모델**에 에이전트를 붙여야 할 때
> Layer 3 방식이 여전히 유효하다. 모델의 지시 이행 능력(instruction following)이
> 충분히 좋다면 ReAct 프롬프트만으로도 동작한다.

> ⚠️ **주의**: 정규식 파싱은 LLM 출력이 형식을 벗어나면 즉시 실패한다.
> `re.search(...)`가 `None`을 반환하는 케이스 처리가 필수다.
> LangChain의 `OutputParser`가 이 예외 처리를 담당한다.

🔗 [ReAct 논문 (Yao et al., 2022)](https://arxiv.org/abs/2210.03629)  
🔗 [LangChain ReAct Agent 공식 문서](https://python.langchain.com/docs/how_to/migrate_agent/)

---

## 4. 이 섹션에서 사용하는 환경

| 항목 | 내용 |
|------|------|
| **로컬 모델** | Ollama + Qwen2.5 (Function Calling 지원) |
| **클라우드 모델** | OpenAI (GPT-4o, 필요 시) |
| **조건** | Function Calling을 지원하는 모델이면 Layer 1~2 모두 가능 |

```bash
# Ollama 설치 (macOS)
brew install ollama
ollama pull qwen2.5:7b   # 또는 원하는 Function Calling 지원 모델
ollama serve
```

> ⚠️ **주의**: Ollama 모델 중 Function Calling을 지원하지 않는 모델(예: 일부 소형 모델)로
> Layer 1~2를 시도하면 `tool_calls`가 항상 빈 리스트로 반환된다.
> 반드시 [Ollama 모델 페이지](https://ollama.com/library)에서 "tools" 지원 여부를 확인하라.

> 💡 **실무 포인트**: 로컬 모델은 무료지만 GPT-4급 성능은 아니다.
> 학습 목적으로는 충분하지만, 프로덕션에서 복잡한 다단계 추론이 필요하다면
> OpenAI/Anthropic 모델을 함께 테스트해보는 것을 권장한다.

🔗 [Ollama 공식 사이트](https://ollama.com)  
🔗 [LangChain Ollama 통합 패키지](https://python.langchain.com/docs/integrations/chat/ollama/)

---

## 5. LangChain의 진짜 가치 (이 섹션 후 알게 되는 것)

Layer 2(순수 구현)를 직접 해보면 LangChain이 제공하는 핵심 가치를 체감하게 된다:

| LangChain 제공 | 직접 구현 시 필요한 것 |
|---------------|----------------------|
| `@tool` 데코레이터 | JSON 스키마 수동 작성 |
| `bind_tools()` | Provider별 Tool 포맷 직접 처리 |
| `init_chat_model()` | Provider별 클라이언트 직접 import |
| `HumanMessage` 등 | Provider별 메시지 dict 직접 작성 |
| `@tool` 자동 LangSmith 트레이싱 | 수동 트레이싱 코드 추가 |

```text
✅ 단일 인터페이스: OpenAI든 Ollama든 동일한 코드로 교체 가능
✅ Tool 추상화: JSON 스키마 직접 작성 불필요
✅ 메시지 관리: 대화 기록 관리 자동화
✅ 이식성: 모델이 바뀌어도 코드 변경 최소화
```

> 🔍 **내부 동작 원리**: `init_chat_model("ollama:qwen2.5:7b")`의 내부
>
> ```text
> "ollama:qwen2.5:7b" 문자열을 파싱:
>   provider = "ollama"  →  langchain-ollama 패키지의 ChatOllama 클래스 동적 로딩
>   model    = "qwen2.5:7b"  →  ChatOllama(model="qwen2.5:7b") 초기화
>
> "openai:gpt-4o" 로 바꾸면:
>   provider = "openai"  →  langchain-openai 패키지의 ChatOpenAI 클래스 동적 로딩
>   model    = "gpt-4o"  →  ChatOpenAI(model="gpt-4o") 초기화
> ```

🔗 [init_chat_model() API 레퍼런스](https://python.langchain.com/api_reference/langchain/chat_models/langchain.chat_models.base.init_chat_model.html)

---

## 6. 핵심 요약

```text
✅ 에이전트의 본질 = while 루프: Thought → Action → Observe → 반복
✅ Layer 1: LangChain 기본 요소(@tool, bind_tools, ToolMessage)로 루프 직접 구현
✅ Layer 2: 프레임워크 없이 순수 구현 → LangChain이 감추는 것을 체감
✅ Layer 3: ReAct 프롬프트 + 정규식 → Function Calling 이전 시대의 에이전트 원리
✅ bind_tools() = Tool JSON 스키마를 매 호출마다 자동 삽입하는 래퍼
✅ ToolMessage = Tool 결과를 대화 기록에 추가, tool_call_id 반드시 일치시킬 것
✅ Scratchpad = LLM의 추론 과정 누적 기록, AgentExecutor가 자동 관리
✅ 반드시 직접 코드를 작성하면서 LangSmith 트레이스로 검증해야 진짜 이해 가능
```

---

## 7. 실습 체크리스트

- [ ] Ollama 설치 및 모델 다운로드 (`ollama pull qwen2.5:7b`)
- [ ] Layer 1: `@tool` + `bind_tools` + `while` 루프로 에이전트 직접 구현
- [ ] Layer 1 실행 후 LangSmith 트레이스에서 각 `ToolMessage`의 `tool_call_id` 확인
- [ ] Layer 2: `openai` 패키지 직접 사용해 동일 에이전트 구현 (JSON 스키마 직접 작성)
- [ ] Layer 1과 Layer 2의 코드 라인 수 비교 → LangChain 추상화 가치 체감
- [ ] Layer 3: ReAct 프롬프트 + 정규식으로 에이전트 구현 (Function Calling 없이)
- [ ] 정규식 파싱 실패 케이스 재현 → `OutputParser`의 필요성 확인
- [ ] LangSmith에서 각 레이어의 트레이스를 열어 메시지 흐름 비교

---

*참고 패키지: `langchain`, `langchain-ollama`, `langchain-openai`, `langsmith`, `python-dotenv`*

> `pip install langchain langchain-ollama langchain-openai langsmith python-dotenv`
