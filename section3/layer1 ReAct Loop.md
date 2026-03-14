# Section 3 — Layer 1: ReAct Loop (LangChain Tool Calling) 강의 노트

> **강의 주제**: LangChain의 기본 구성요소만 사용해 ReAct 스타일의 Agent Loop를 직접 구현하기
> **난이도**: 고급
> **목표**: `AgentExecutor` 뒤에 숨겨진 실제 루프를 손으로 구현하며 Thought → Action → Observation 흐름을 체득하기
> **사용 패키지**: `langchain`, `langchain-core`, `langsmith`, `python-dotenv`
> **권장 모델**: Function Calling을 지원하는 모델 (`ollama:qwen3:1.7b`, `openai:gpt-4o` 등)

---

## 1. 왜 Layer 1을 따로 배워야 할까?

> **"왜 `create_agent()`나 `AgentExecutor`를 바로 쓰지 않고, 굳이 루프를 직접 짤까요?" (Why)**
> 고수준 API는 빠르게 데모를 만들기엔 좋지만, 에이전트가 왜 특정 Tool을 골랐는지, 왜 같은 Tool을 반복 호출하는지, 왜 중간에 무한 루프에 빠졌는지까지는 잘 안 보입니다. Layer 1은 LangChain의 편의 기능은 유지하되, 에이전트의 심장부인 루프 자체는 직접 제어하는 단계입니다.

이 레이어의 핵심은 하나다.  
**현대적인 Tool Calling 기반 에이전트도 본질적으로는 ReAct 루프다.**

```text
Thought      : LLM이 다음 행동을 결정
Action       : 어떤 Tool을 어떤 인자로 호출할지 선택
Observation  : Tool 실행 결과를 다시 모델에게 전달
Final Answer : 더 이상 Tool이 필요 없으면 종료
```

> 🔍 **내부 동작 원리**:
>
> ```text
> messages = [SystemMessage, HumanMessage]
>
> while iteration < MAX_ITERATIONS:
>     ai_message = llm_with_tools.invoke(messages)   # Thought
>
>     if ai_message.tool_calls is empty:
>         return ai_message.content                  # Final Answer
>
>     tool_call = ai_message.tool_calls[0]          # Action
>     observation = tool.invoke(tool_call.args)     # Execute
>
>     messages += [ai_message, ToolMessage(...)]    # Observation 저장
> ```
>
> 예전 ReAct는 텍스트로 `Action:`을 파싱했고, 지금은 `tool_calls`라는 구조화된 필드로 받는다는 차이만 있을 뿐, 루프의 뼈대는 동일합니다.

> 💡 **실무 포인트**: 프로덕션 디버깅은 거의 항상 Layer 1 관점에서 이뤄진다. "모델이 무엇을 보고 어떤 Tool을 고른 뒤, 그 결과를 어떻게 다음 호출에 반영했는가"를 추적할 수 있어야 한다.

🔗 [Agents 공식 문서](https://docs.langchain.com/oss/python/langchain/agents)  
🔗 [Messages 공식 문서](https://docs.langchain.com/oss/python/langchain/messages)

---

## 2. 이번 예제의 목표: 쇼핑 어시스턴트 Agent

강의 예제는 아주 단순한 이커머스 시나리오다.

- 사용자는 "노트북 가격에 골드 할인 적용하면 얼마야?"처럼 질문한다.
- Agent는 절대 가격을 추측하면 안 된다.
- 반드시 먼저 가격 조회 Tool을 호출해야 한다.
- 할인 계산도 모델이 직접 하지 말고 할인 Tool을 사용해야 한다.

이 예제가 좋은 이유는 Tool 간의 **순서 의존성**이 명확하기 때문이다.

```text
질문
  ↓
get_product_price(product)
  ↓
apply_discount(price, discount_tier)
  ↓
최종 답변
```

> 💡 **실무 포인트**: 에이전트 예제는 복잡한 도메인보다 이런 "순서 제약이 있는 단순 업무"가 학습에 더 좋다. Tool 선택 오류, 순서 오류, hallucination 방지 패턴을 훨씬 선명하게 볼 수 있다.

---

## 3. 전체 코드 먼저 보기

아래 코드는 강의 내용 기준으로 정리한 Layer 1 구현체다.

```python
from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {
        "laptop": 1299.99,
        "headphones": 149.95,
        "keyboard": 89.50,
    }
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool and a discount tool.\n\n"
                "STRICT RULES:\n"
                "1. NEVER guess or assume any product price. "
                "You MUST call get_product_price first.\n"
                "2. Only call apply_discount AFTER you receive the real price.\n"
                "3. NEVER calculate discounts yourself. Always use apply_discount.\n"
                "4. If the user does not specify a discount tier, ask first."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)
        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        )

    print("ERROR: Max iterations reached without a final answer")
    return None


if __name__ == "__main__":
    result = run_agent(
        "What is the price of a laptop after applying a gold discount?"
    )
    print(result)
```

> ⚠️ **버전 주의**: `init_chat_model`, `@tool`, 메시지 import 경로는 LangChain 버전에 따라 약간 달라질 수 있다. 강의 코드와 현재 공식 문서가 100% 동일하지 않을 수 있으니, import 에러가 나면 공식 문서를 기준으로 import 경로를 다시 확인해야 한다.

🔗 [Models 공식 문서](https://docs.langchain.com/oss/python/langchain/models)  
🔗 [Tools 공식 문서](https://docs.langchain.com/oss/python/langchain/tools)

---

## 4. `init_chat_model` — 모델 교체 비용을 1줄로 줄이는 추상화

> **"왜 provider별 클래스를 직접 import하지 않고 `init_chat_model()`을 쓸까요?" (Why)**
> Layer 1의 목적은 에이전트 루프 자체를 이해하는 것이지, OpenAI SDK와 Ollama SDK의 세부 포맷 차이를 배우는 게 아닙니다. `init_chat_model()`은 모델 초기화 부분의 잡음을 줄여서 우리가 루프와 Tool Calling에 집중하게 해줍니다.

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("ollama:qwen3:1.7b", temperature=0)

# 추후 공급자만 바꿔도 나머지 코드는 유지 가능
# llm = init_chat_model("openai:gpt-4o", temperature=0)
```

> 🔍 **내부 동작 원리**:
>
> ```text
> "ollama:qwen3:1.7b"
>   ├─ provider = "ollama"
>   └─ model    = "qwen3:1.7b"
>
> init_chat_model(...)
>   → provider 문자열에 맞는 LangChain integration 선택
>   → 해당 ChatModel 인스턴스 생성
>   → 공통 인터페이스(BaseChatModel)로 반환
> ```
>
> 이 덕분에 뒤쪽 코드는 "어느 회사 모델인가"를 거의 신경 쓰지 않아도 된다.

> 💡 **실무 포인트**: 개발 단계에서는 Ollama 같은 로컬 모델로 루프를 검증하고, 품질 검증 단계에서 OpenAI나 Anthropic으로 바꾸는 전략이 흔하다. 이때 에이전트 코드가 공급자에 종속돼 있으면 실험 속도가 급격히 떨어진다.

---

## 5. `@tool` 데코레이터 — 파이썬 함수를 LLM이 이해하는 Tool로 변환

> **"왜 함수에 docstring과 타입 힌트를 정성스럽게 써야 할까요?" (Why)**
> Tool Calling에서 LLM은 파이썬 코드를 실행하는 게 아니라, 먼저 Tool의 설명서(JSON schema)를 읽고 어떤 함수를 어떤 인자로 부를지 판단합니다. 즉, 함수 본문보다도 함수명, docstring, 타입 힌트가 Tool 선택 정확도에 더 직접적인 영향을 준다.

```python
@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    ...


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    ...
```

> 🔍 **내부 동작 원리**:
>
> ```text
> 파이썬 함수
>   ├─ 함수명           → Tool name
>   ├─ docstring        → Tool description
>   ├─ 타입 힌트        → parameters schema
>   └─ 기본값 유무      → required 여부
>
> LangChain @tool
>   → 이 메타데이터를 읽어서
>   → provider별 Tool Calling 포맷에 맞는 schema 생성
> ```

| 파이썬 요소 | LLM에게 전달되는 의미 |
|---|---|
| 함수 이름 | Tool의 고유 식별자 |
| docstring | Tool을 언제 써야 하는지에 대한 설명 |
| 타입 힌트 | 인자 타입 검증 힌트 |
| 인자 이름 | LLM이 채워 넣을 JSON key |

> 💡 **실무 포인트**: Tool이 많아질수록 docstring 품질이 에이전트 품질을 좌우한다. `search_docs()`와 `search_code()`처럼 비슷한 Tool이 공존할 때 설명이 모호하면 LLM은 의외로 자주 잘못 고른다.

> ⚠️ **주의**: 타입 힌트가 없거나 docstring이 부실하면 스키마가 약해지고, 결과적으로 잘못된 인자 생성이나 엉뚱한 Tool 호출이 늘어난다.

---

## 6. `bind_tools` — 모델에 Tool 목록을 실제로 연결하는 단계

> **"Tool 함수를 정의만 하면 모델이 바로 쓸 수 있을까요?" (Why)**
> 아니다. 파이썬에 함수를 만들어 두는 것과, 그 함수 정보를 모델 호출 payload에 실어 보내는 것은 전혀 다른 일이다. `bind_tools()`는 이 간극을 메워주는 메서드다.

```python
tools = [get_product_price, apply_discount]
tools_dict = {tool.name: tool for tool in tools}

llm_with_tools = llm.bind_tools(tools)
```

> 🔍 **내부 동작 원리**:
>
> ```text
> llm.bind_tools(tools)
>   ├─ Tool 스키마 수집
>   ├─ provider 형식으로 변환
>   └─ 이후 invoke() 시 payload에 자동 포함
>
> 결과:
>   ai_message.tool_calls
>   = [
>       {
>         "name": "get_product_price",
>         "args": {"product": "laptop"},
>         "id": "call_abc123"
>       }
>     ]
> ```

> 💡 **실무 포인트**: `tools_dict = {tool.name: tool for tool in tools}` 패턴은 거의 필수다. 모델은 문자열 이름만 반환하고, 파이썬은 그 문자열을 실제 함수 객체로 다시 매핑해야 하기 때문이다.

> ⚠️ **주의**: `bind_tools()`를 빼먹으면 모델은 Tool의 존재를 모르기 때문에, 아무리 좋은 Tool을 정의해도 `tool_calls`가 비어 있는 final answer만 뱉을 가능성이 높다.

🔗 [Tool Calling 관련 메시지 문서](https://docs.langchain.com/oss/python/langchain/messages)

---

## 7. `messages` 리스트 — 에이전트의 메모리이자 상태 저장소

> **"왜 Tool 결과를 변수에만 저장하지 않고, 다시 `messages`에 넣어야 할까요?" (Why)**
> Tool을 실행한 건 파이썬이지 LLM이 아니다. 모델은 자신이 방금 요청한 Tool이 어떤 결과를 돌려줬는지 직접 보지 못한다. 따라서 실행 결과를 `ToolMessage` 형태로 다시 대화 기록에 넣어줘야 다음 iteration에서 그 결과를 바탕으로 추론을 이어갈 수 있다.

```python
messages = [
    SystemMessage(content="...STRICT RULES..."),
    HumanMessage(content=question),
]
```

iteration이 진행될수록 `messages`는 이렇게 자란다.

```text
Iteration 1 시작
  [SystemMessage, HumanMessage]

Iteration 2 시작
  [SystemMessage, HumanMessage, AIMessage(tool_call=get_product_price), ToolMessage("1299.99")]

Iteration 3 시작
  [SystemMessage, HumanMessage, AIMessage(...), ToolMessage(...),
   AIMessage(tool_call=apply_discount), ToolMessage("1000.99")]
```

> 🔍 **내부 동작 원리**:
>
> ```text
> SystemMessage  : 규칙과 역할
> HumanMessage   : 사용자 질문
> AIMessage      : 모델의 판단(도구 호출 또는 최종 답변)
> ToolMessage    : 파이썬이 실행한 도구 결과
> ```
>
> 에이전트는 사실 "messages 리스트를 매 iteration 업데이트하면서 다시 모델에 넣는 상태 머신"에 가깝다.

> 💡 **실무 포인트**: 프롬프트 품질만큼이나 `messages` 관리가 중요하다. tool result를 잘못 넣거나 순서를 바꾸면 모델은 이전 상태를 오해하고 이상한 행동을 한다.

🔗 [Messages 공식 문서](https://docs.langchain.com/oss/python/langchain/messages)

---

## 8. ReAct Loop 핵심: Thought → Action → Observation

이 강의의 본질은 아래 `for` 루프에 있다.

```python
for iteration in range(1, MAX_ITERATIONS + 1):
    ai_message = llm_with_tools.invoke(messages)
    tool_calls = ai_message.tool_calls

    if not tool_calls:
        return ai_message.content

    tool_call = tool_calls[0]
    tool_name = tool_call.get("name")
    tool_args = tool_call.get("args", {})
    tool_call_id = tool_call.get("id")

    tool_to_use = tools_dict.get(tool_name)
    observation = tool_to_use.invoke(tool_args)

    messages.append(ai_message)
    messages.append(
        ToolMessage(content=str(observation), tool_call_id=tool_call_id)
    )
```

### ReAct 개념과 코드 대응

| ReAct 단계 | 코드 위치 | 의미 |
|---|---|---|
| Thought | `ai_message = llm_with_tools.invoke(messages)` | 모델이 다음 행동을 판단 |
| Action | `tool_call = tool_calls[0]` | 어떤 Tool을 어떤 인자로 쓸지 결정 |
| Execute | `tool_to_use.invoke(tool_args)` | 파이썬이 실제 함수 실행 |
| Observation | `ToolMessage(...)` 추가 | 결과를 다시 모델 문맥에 주입 |
| Final Answer | `if not tool_calls` | 더 이상 Tool이 필요 없을 때 종료 |

> 🔍 **내부 동작 원리**:
>
> ```text
> [Iteration N]
>   messages ──▶ LLM
>                ├─ tool_calls 있음  → Tool 실행 후 messages 갱신 → Iteration N+1
>                └─ tool_calls 없음  → Final Answer 반환 후 종료
> ```

> 💡 **실무 포인트**: 이 구조를 정확히 이해해야 나중에 `AgentExecutor`, `create_react_agent`, LangGraph 노드 그래프를 볼 때도 "결국 내부에서는 같은 루프가 더 잘 포장되어 있을 뿐"이라는 감각이 생긴다.

---

## 9. `tool_call_id` — 생각보다 훨씬 중요한 프로토콜 키

> **"왜 Tool 결과에 `tool_call_id`를 꼭 넣어야 할까요?" (Why)**
> 모델이 한 번에 여러 Tool을 요청할 수 있기 때문이다. 결과값만 던져주면 LLM은 어떤 결과가 어떤 호출에 대응되는지 알 수 없다. `tool_call_id`는 이 매핑을 보장하는 요청 번호표다.

```python
tool_call_id = tool_call.get("id")

messages.append(
    ToolMessage(content=str(observation), tool_call_id=tool_call_id)
)
```

> 🔍 **내부 동작 원리**:
>
> ```text
> AIMessage.tool_calls
>   = [{"name": "get_product_price", "id": "call_abc123", ...}]
>
> ToolMessage
>   = ToolMessage(content="1299.99", tool_call_id="call_abc123")
> ```
>
> 이 둘이 정확히 대응돼야 다음 모델 호출에서 protocol이 유지된다.

> 💡 **실무 포인트**: 병렬 Tool Calling을 확장할 계획이 있다면 이 필드를 지금부터 엄격하게 다루는 습관이 중요하다. 나중에 `for tool_call in tool_calls:`로 바꿔도 설계가 흔들리지 않는다.

> ⚠️ **주의**: `tool_call_id`가 누락되거나 잘못 연결되면 provider에 따라 요청 자체가 실패한다. 이건 모델 품질 문제가 아니라 **API protocol 위반**이다.

---

## 10. 왜 `MAX_ITERATIONS`가 필요한가?

> **"모델이 똑똑하면 언젠가 끝내지 않을까요?" (Why)**
> 실제로는 그렇지 않다. 잘못된 Tool 설명, 약한 모델, 부정확한 System Prompt, 실패한 Tool 결과 처리 때문에 같은 Tool을 반복 호출하는 루프가 쉽게 생긴다.

```python
MAX_ITERATIONS = 10

for iteration in range(1, MAX_ITERATIONS + 1):
    ...

print("ERROR: Max iterations reached without a final answer")
```

> 🔍 **내부 동작 원리**:
>
> ```text
> iteration 1 → 가격 조회
> iteration 2 → 할인 적용
> iteration 3 → 최종 답변
>
> 정상이라면 2~4회 안에 끝난다.
> 하지만 실패 시:
>   가격 조회 → 다시 가격 조회 → 다시 가격 조회 ...
> ```

> 💡 **실무 포인트**: `MAX_ITERATIONS`는 성능 최적화가 아니라 안전장치다. 특히 오픈 웨이트 소형 모델은 지시를 어기고 같은 Action을 반복하는 경우가 있어 guardrail이 꼭 필요하다.

---

## 11. System Prompt — Tool 오남용을 막는 방어적 설계

강의 코드의 System Prompt는 단순 안내문이 아니라 일종의 **운영 정책 문서**다.

```python
SystemMessage(
    content=(
        "You are a helpful shopping assistant. "
        "You have access to a product catalog tool and a discount tool.\n\n"
        "STRICT RULES:\n"
        "1. NEVER guess or assume any product price. "
        "You MUST call get_product_price first.\n"
        "2. Only call apply_discount AFTER you receive the real price.\n"
        "3. NEVER calculate discounts yourself. Always use apply_discount.\n"
        "4. If the user does not specify a discount tier, ask first."
    )
)
```

> 🔍 **내부 동작 원리**:
>
> ```text
> 약한 모델은 "대충 계산해도 되겠지"라고 자주 생각한다.
> System Prompt가 강할수록
>   - Tool 사용 순서를 고정하고
>   - 추측 답변을 줄이고
>   - 누락된 파라미터가 있을 때 재질문을 유도한다.
> ```

> 💡 **실무 포인트**: Tool Calling이 있다고 해서 prompt engineering이 사라지지 않는다. 오히려 Tool 사용 조건, 금지 규칙, 재질문 조건을 더 명시적으로 적어야 안정성이 올라간다.

> ⚠️ **주의**: "할인은 계산하지 말고 Tool을 써라" 같은 규칙이 없으면 모델이 자체 계산을 시도할 수 있다. 특히 저비용 로컬 모델일수록 이런 현상이 더 자주 나온다.

---

## 12. LangSmith 트레이싱 — 루프를 눈으로 보는 도구

> **"왜 `@traceable`을 붙일까요?" (Why)**
> Layer 1의 가장 큰 장점은 내부를 직접 구현한다는 점인데, 트레이싱이 없으면 그 내부 상태를 한눈에 보기 어렵다. `@traceable`은 에이전트 루프 전체를 LangSmith 안에서 하나의 실행 단위로 묶어 보여준다.

```python
from langsmith import traceable


@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    ...
```

```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=ls__...
LANGCHAIN_PROJECT=section3-layer1-react-loop
```

> 🔍 **내부 동작 원리**:
>
> ```text
> run_agent() 시작
>   ├─ trace root 생성
>   ├─ llm.invoke() 호출 기록
>   ├─ tool.invoke() 호출 기록
>   └─ 최종 답변/에러/시간/토큰 정보 저장
> ```

> 💡 **실무 포인트**: Agent는 보통 "왜 그렇게 행동했는지"가 중요하다. LangSmith 없이는 툴 선택 실수, prompt 문제, 중간 observation 오류를 재현하기 어렵다.

🔗 [LangSmith 공식 문서](https://docs.smith.langchain.com/)

---

## 13. 현재 구현의 의도적인 단순화와 다음 단계

이 Layer 1 구현은 학습을 위해 일부러 단순화되어 있다.

1. `tool_calls[0]`만 처리한다. 즉, 한 iteration에 하나의 Tool만 실행한다.
2. Tool 실패 처리, retry, timeout, fallback이 없다.
3. 메모리는 `messages` 리스트 하나만 사용한다.
4. 상태 분기가 단순해서 LangGraph 같은 명시적 그래프는 아직 쓰지 않는다.

> 💡 **실무 포인트**: 이 단순화 덕분에 핵심 루프가 또렷하게 보인다. 하지만 프로덕션에 가면 병렬 Tool Calling, 예외 처리, 상태 분기, 사용자 세션 메모리, structured output 검증이 바로 추가된다.

다음 레이어로 이어지는 방향은 보통 이렇다.

```text
Layer 1
  └─ LangChain helper는 쓰되 루프는 직접 작성

Layer 2
  └─ Tool schema와 provider payload까지 직접 작성

Layer 3
  └─ Function Calling 없이 ReAct prompt + parsing으로 구현
```

---

## 14. 핵심 요약

```text
✅ Layer 1의 목표는 Agent Loop의 실체를 직접 구현하며 이해하는 것이다.
✅ Tool Calling 기반 Agent도 본질적으로는 ReAct 루프다.
✅ @tool은 함수명, docstring, 타입 힌트를 읽어 Tool schema를 만든다.
✅ bind_tools()를 해야 모델이 Tool의 존재를 인식한다.
✅ messages 리스트는 에이전트의 메모리이자 상태 저장소다.
✅ AIMessage + ToolMessage를 차곡차곡 쌓아야 다음 iteration이 성립한다.
✅ tool_call_id는 단순 부가정보가 아니라 protocol 매핑 키다.
✅ MAX_ITERATIONS는 무한 루프 방지용 필수 안전장치다.
✅ System Prompt는 Tool 사용 순서와 금지 규칙을 강제하는 운영 정책이다.
✅ LangSmith 트레이싱이 있어야 에이전트 디버깅이 실전 수준으로 가능해진다.
```

---

## 15. 실습 체크리스트

- [ ] `STRICT RULES`를 약하게 바꾸고 실행해서 모델이 가격을 추측하는지 관찰하기
- [ ] `tool_calls[0]` 대신 `for tool_call in tool_calls:`로 바꿔 병렬 Tool Calling 구조 초안 만들어보기
- [ ] `tool_call_id`를 일부러 잘못 넘겨보고 어떤 에러가 나는지 확인하기
- [ ] `MAX_ITERATIONS=2`로 낮춰보고 루프가 중간에 끊기는 상황 확인하기
- [ ] `get_product_price`에 없는 상품명을 넣어 `0`이 반환될 때 모델이 어떻게 행동하는지 살펴보기
- [ ] `discount_tier`를 빼고 질문해 모델이 재질문하는지 테스트하기
- [ ] `ollama:qwen3:1.7b` 대신 다른 Function Calling 지원 모델로 바꿔 Tool 선택 품질 비교하기
- [ ] LangSmith에서 iteration별 메시지 상태와 tool result 흐름을 직접 확인하기

