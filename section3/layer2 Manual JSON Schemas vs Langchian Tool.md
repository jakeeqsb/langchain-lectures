# Section 3 — Layer 2: Manual JSON Schemas vs LangChain Tool 강의 노트

> **강의 주제**: LangChain 추상화를 걷어내고, Ollama Python SDK로 Raw Function Calling Agent를 직접 구현하기
> **난이도**: 고급
> **목표**: `@tool`, `bind_tools`, 메시지 클래스, 자동 트레이싱이 실제로 어떤 개발 비용을 숨기고 있는지 체감하기
> **사용 패키지**: `ollama`, `langsmith`, `python-dotenv`
> **권장 모델**: `qwen3:1.7b` 또는 Tool Calling을 지원하는 Ollama 모델

---

## 1. 왜 Layer 2를 배우는가?

> **"Layer 1에서 이미 Agent Loop를 직접 짰는데, 왜 또 한 단계 더 내려가야 할까요?" (Why)**
> Layer 1에서는 루프는 직접 구현했지만, Tool schema 생성, provider별 payload 포맷, 메시지 객체, Tool invocation 인터페이스는 여전히 LangChain이 대신 처리해줬습니다. Layer 2는 그 보호막을 벗기고, Tool Calling이 실제로 얼마나 vendor-specific하고 번거로운지 드러내는 단계입니다.

Layer 2의 핵심 질문은 이거다.

```text
LangChain이 없으면
  1. Tool을 모델이 이해하는 형식으로 누가 변환하는가?
  2. 메시지 포맷은 누가 맞춰주는가?
  3. Tool 실행 결과를 어느 형식으로 되돌려주는가?
  4. LangSmith trace는 누가 남기는가?
```

답은 전부 하나다.  
**개발자가 직접 해야 한다.**

> 🔍 **내부 동작 원리**:
>
> ```text
> Layer 1
>   Python 함수
>     └─ @tool
>         └─ LangChain이 JSON schema 자동 생성
>
> Layer 2
>   Python 함수
>     └─ 개발자가 tools=[{...JSON schema...}]를 직접 작성
>         └─ ollama.chat(model=..., tools=..., messages=...)
> ```

> 💡 **실무 포인트**: Layer 2를 한 번이라도 경험해보면, 이후 LangChain의 추상화가 "편의 기능"이 아니라 "개발 비용 절감 장치"라는 점이 훨씬 선명해진다. 특히 멀티 모델 전략을 쓰는 팀일수록 이 차이가 크다.

🔗 [Ollama Tool Calling 공식 문서](https://ollama.com/blog/tool-support)  
🔗 [LangChain Tools 공식 문서](https://docs.langchain.com/oss/python/langchain/tools)

---

## 2. 이 레이어에서 바뀌는 것: LangChain 제거 목록

Layer 1과 비교하면 아래가 사라진다.

- `init_chat_model`
- `@tool`
- `llm.bind_tools(...)`
- `HumanMessage`, `SystemMessage`, `ToolMessage`
- `tool.invoke(...)`
- LangChain 내부 자동 trace

대신 아래를 직접 작성해야 한다.

- `ollama.chat(...)` 호출
- Tool JSON schema 수동 작성
- 메시지 dict 수동 관리
- tool name -> Python 함수 매핑 dict
- Tool 결과를 `{"role": "tool", ...}` 형태로 수동 append
- LangSmith `@traceable`을 함수별로 직접 부착

> 💡 **실무 포인트**: 이 레이어의 학습 포인트는 "Ollama를 잘 쓰는 법"이 아니라, **에이전트 프레임워크가 추상화해 주는 작업이 정확히 무엇인지 보이는 상태**를 만드는 데 있다.

---

## 3. 전체 코드 먼저 보기

```python
from dotenv import load_dotenv

load_dotenv()

import ollama
from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"


@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f"    >> Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@traceable(run_type="tool")
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze, silver, gold."""
    print(
        f"    >> Executing apply_discount(price={price}, discount_tier='{discount_tier}')"
    )
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name, e.g. laptop, headphones, keyboard",
                    }
                },
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": "bronze, silver, or gold",
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    tools_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool and a discount tool.\n\n"
                "STRICT RULES:\n"
                "1. NEVER guess any product price. "
                "You MUST call get_product_price first.\n"
                "2. Only call apply_discount after receiving the real price.\n"
                "3. NEVER calculate discounts yourself.\n"
                "4. If no discount tier is given, ask the user first."
            ),
        },
        {"role": "user", "content": question},
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")

        response = ollama_chat_traced(messages=messages)
        ai_message = response.message

        tool_calls = ai_message.tool_calls
        if not tool_calls:
            print(f"\nFinal Answer: {ai_message.content}")
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.function.name
        tool_args = tool_call.function.arguments

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use(**tool_args)
        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append({"role": "tool", "content": str(observation)})

    print("ERROR: Max iterations reached without a final answer")
    return None
```

> ⚠️ **버전 주의**: Ollama Python SDK의 응답 객체 구조와 Tool Calling 포맷은 버전에 따라 달라질 수 있다. `response.message.tool_calls`, `tool_call.function.name`, `tool_call.function.arguments` 같은 접근 방식은 사용 중인 SDK 버전에서 직접 확인해야 한다.

🔗 [Ollama Python 라이브러리](https://github.com/ollama/ollama-python)  
🔗 [LangSmith `traceable` 문서](https://docs.smith.langchain.com/reference/python/run_helpers/langsmith.run_helpers.traceable)

---

## 4. Layer 1과 Layer 2의 본질적 차이

### Layer 1

```python
@tool
def get_product_price(product: str) -> float:
    ...

llm = init_chat_model("ollama:qwen3:1.7b")
llm_with_tools = llm.bind_tools([get_product_price])
ai_message = llm_with_tools.invoke(messages)
observation = get_product_price.invoke(tool_args)
messages.append(ToolMessage(...))
```

### Layer 2

```python
def get_product_price(product: str) -> float:
    ...

tools_for_llm = [{...수동 JSON schema...}]
response = ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)
tool_name = response.message.tool_calls[0].function.name
observation = get_product_price(**tool_args)
messages.append({"role": "tool", "content": str(observation)})
```

| 항목 | Layer 1 | Layer 2 |
|---|---|---|
| Tool 정의 | `@tool` | 수동 JSON schema |
| 메시지 형식 | `SystemMessage`, `ToolMessage` | dict (`{"role": ...}`) |
| 모델 호출 | `llm.invoke()` | `ollama.chat()` |
| Tool 실행 | `tool.invoke(args)` | `python_fn(**args)` |
| Trace | LangChain 내부 연동 도움 | 수동 `@traceable` 부착 |
| Provider 교체 비용 | 낮음 | 높음 |

> 💡 **실무 포인트**: Layer 2는 "추상화가 사라지면 어디서부터 코드가 거칠어지는가"를 보여주는 교육용 레이어다. 실제 운영에서 raw SDK만 쓰는 팀도 있지만, provider가 둘 이상이면 금방 유지보수 비용이 커진다.

---

## 5. Manual JSON Schema — `@tool`이 뒤에서 하던 일을 직접 쓰기

> **"왜 이 JSON을 굳이 손으로 써봐야 할까요?" (Why)**
> Tool Calling의 실체를 이해하려면, 모델이 실제로 보는 것은 파이썬 함수가 아니라 **함수 설명서(JSON schema)** 라는 사실을 직접 봐야 한다. `@tool`은 이걸 자동 생성해 주지만, Layer 2에서는 그 자동화를 걷어낸다.

```python
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product": {
                        "type": "string",
                        "description": "The product name",
                    }
                },
                "required": ["product"],
            },
        },
    }
]
```

> 🔍 **내부 동작 원리**:
>
> ```text
> LLM은 함수 본문을 모른다.
> LLM이 보는 건 오직:
>   - 함수 이름(name)
>   - 언제 써야 하는지(description)
>   - 어떤 인자가 필요한지(parameters)
>   - 인자의 타입과 required 여부
> ```

### `@tool`이 숨기던 매핑

| 파이썬 함수 메타데이터 | 수동 JSON schema에서 직접 써야 하는 값 |
|---|---|
| 함수 이름 | `"name"` |
| docstring | `"description"` |
| 타입 힌트 | `"type": "string"`, `"number"` 등 |
| 인자 존재 여부 | `"required"` |
| 인자 설명 | `"properties" -> "description"` |

> 💡 **실무 포인트**: Tool이 2개일 때는 참을 만하지만, 20개가 되면 schema drift가 발생한다. 실제 파이썬 함수 시그니처는 바뀌었는데 JSON schema를 깜빡하고 안 바꾸는 순간, 모델은 틀린 인자를 생성하기 시작한다.

> ⚠️ **주의**: provider마다 schema 포맷이 미묘하게 다르다. 한 벤더에서 잘 되던 schema를 다른 벤더에 그대로 넣으면 깨질 수 있다. 이게 LangChain 추상화가 제공하는 핵심 가치 중 하나다.

🔗 [Anthropic Tool Use 공식 문서](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview)

---

## 6. Ollama SDK 직접 호출 — `llm.invoke()` 대신 `ollama.chat()`

> **"왜 Layer 2에서 모델 호출부가 갑자기 raw SDK처럼 바뀌나요?" (Why)**
> Layer 2의 목표는 LangChain의 공통 인터페이스를 벗겨내고, 실제 provider SDK가 어떤 데이터를 요구하는지 그대로 보는 데 있다. 그래서 `invoke()` 대신 `ollama.chat()`를 직접 쓴다.

```python
@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)
```

> 🔍 **내부 동작 원리**:
>
> ```text
> ollama.chat(
>   model="qwen3:1.7b",
>   tools=[...manual schema...],
>   messages=[...raw dict messages...]
> )
>   ↓
> provider가 반환한 raw-ish response 객체 수신
>   ↓
> response.message.tool_calls 분석
> ```

> 💡 **실무 포인트**: 이 시점부터 코드는 provider SDK에 강하게 결합된다. 모델을 Anthropic이나 OpenAI로 바꾸면 호출 함수, 응답 구조, Tool schema, 메시지 포맷까지 거의 다 다시 맞춰야 할 수 있다.

---

## 7. 메시지 객체 대신 dict를 직접 관리한다

> **"Layer 1에서는 `SystemMessage`, `ToolMessage`를 썼는데 왜 여기선 dict를 쓰죠?" (Why)**
> LangChain 메시지 클래스는 provider별 메시지 구조 차이를 추상화하기 위한 래퍼다. Layer 2에서는 그 래퍼를 제거했기 때문에, provider가 받는 raw role/content 구조를 우리가 직접 만든다.

```python
messages = [
    {"role": "system", "content": "..."},
    {"role": "user", "content": question},
]
```

Tool 결과를 넣을 때도 직접 dict를 추가한다.

```python
messages.append(ai_message)
messages.append({"role": "tool", "content": str(observation)})
```

> 🔍 **내부 동작 원리**:
>
> ```text
> Layer 1:
>   SystemMessage(...) / HumanMessage(...) / ToolMessage(...)
>
> Layer 2:
>   {"role": "system", "content": "..."}
>   {"role": "user", "content": "..."}
>   {"role": "tool", "content": "..."}
> ```
>
> 즉, Layer 2는 "대화 상태 관리"까지 raw protocol에 더 가깝게 내려온 상태다.

> 💡 **실무 포인트**: raw dict는 단순해 보이지만, role naming과 message ordering을 개발자가 직접 책임져야 한다. 여기서 실수하면 디버깅 난이도가 급격히 올라간다.

---

## 8. Tool 결과 실행: `tool.invoke()`가 아니라 그냥 Python 함수 호출

> **"왜 Tool 실행도 다시 평범한 함수 호출로 돌아가나요?" (Why)**
> Layer 2에서는 `@tool`이 사라졌기 때문에 LangChain Tool 객체도 없다. 따라서 `tool.invoke(...)` 같은 통일 인터페이스도 함께 사라진다.

```python
tools_dict = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}

tool_call = tool_calls[0]
tool_name = tool_call.function.name
tool_args = tool_call.function.arguments

tool_to_use = tools_dict.get(tool_name)
observation = tool_to_use(**tool_args)
```

> 🔍 **내부 동작 원리**:
>
> ```text
> 모델이 반환한 값:
>   "get_product_price", {"product": "laptop"}
>
> 파이썬이 해야 하는 일:
>   1. 이름으로 함수 찾기
>   2. kwargs 풀어서 실행
>   3. 결과를 문자열/메시지로 다시 변환
> ```

> 💡 **실무 포인트**: 이 단계부터는 함수 실행 전 validation, 예외 처리, retry, timeout, logging까지 전부 직접 설계해야 한다. Tool abstraction은 단순 문법 설탕이 아니라 실행 계층 정리 장치다.

---

## 9. `traceable`을 직접 붙이는 이유

> **"왜 Tool 함수에 `@traceable(run_type="tool")`를 붙일까요?" (Why)**
> Layer 1에서는 LangChain Tool과 ChatModel이 내부적으로 traceable ecosystem과 잘 연결돼 있다. Layer 2는 raw SDK와 평범한 Python 함수이므로, trace를 남기고 싶으면 직접 장착해야 한다.

```python
@traceable(run_type="tool")
def get_product_price(product: str) -> float:
    ...


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)
```

> 🔍 **내부 동작 원리**:
>
> ```text
> run_agent()
>   ├─ root trace 생성
>   ├─ ollama_chat_traced() → LLM run으로 기록
>   ├─ get_product_price() → Tool run으로 기록
>   └─ apply_discount() → Tool run으로 기록
> ```

> 💡 **실무 포인트**: 프레임워크를 벗어날수록 observability는 자동이 아니라 수동이 된다. 프로덕션에서 raw SDK를 쓰더라도 trace strategy를 먼저 설계하지 않으면 장애 분석이 매우 어렵다.

---

## 10. 이 구현이 보여주는 가장 중요한 차이: Provider Lock-in

> **"왜 강의에서 Anthropic 문서를 같이 언급하나요?" (Why)**
> Tool Calling이라는 개념 자체는 비슷해도, 벤더마다 schema 포맷과 메시지 구조가 다르다. Layer 2에서 raw SDK로 내려오면 이 차이가 즉시 코드 차이로 번진다.

```text
Ollama raw integration
  ├─ ollama.chat(...)
  ├─ Ollama response object parsing
  └─ Ollama-style tools payload

Anthropic raw integration
  ├─ 별도 client 호출
  ├─ Anthropic-style tool schema
  └─ Anthropic-style response parsing
```

> 🔍 **내부 동작 원리**:
>
> ```text
> 같은 "할인 계산 agent"라도
> provider가 바뀌면
>   - 요청 payload
>   - Tool schema
>   - response parsing
>   - message formatting
> 가 다시 바뀔 수 있다.
> ```

> 💡 **실무 포인트**: 멀티 벤더 전략을 고려한다면 raw SDK 위에서 바로 에이전트를 구축하는 것은 초기에 빨라도 장기 유지보수는 불리할 수 있다. 반대로 단일 벤더에 강하게 최적화할 땐 raw SDK가 세밀한 제어를 주기도 한다.

---

## 11. 현재 구현의 한계와 주의점

### 1. 첫 번째 Tool call만 처리한다

```python
tool_call = tool_calls[0]
```

병렬 Tool Calling이 오면 확장해야 한다.

### 2. Tool message에 provider가 요구하는 추가 필드가 있을 수 있다

현재 예제는 최소 형태만 사용한다. SDK 버전에 따라 더 명시적인 필드가 필요할 수 있다.

### 3. 함수와 schema가 분리되어 있어 drift가 발생하기 쉽다

함수 시그니처를 바꾸고 schema를 안 바꾸면 즉시 어긋난다.

### 4. Tool argument validation이 약하다

raw dict/kwargs 구조이므로 validation 계층을 직접 추가해야 한다.

> ⚠️ **공식 문서 확인 필요**: Ollama Python SDK의 tool schema 자동 생성 동작과 docstring 처리 방식은 SDK 구현 버전에 따라 달라질 수 있다. 관련 세부는 사용 중인 `ollama-python` 버전의 README와 source를 직접 확인하는 편이 안전하다.

---

## 12. LangChain이 실제로 제공하는 가치

이 Layer 2를 보고 나면 Layer 1의 LangChain 코드가 왜 훨씬 간결했는지 보인다.

| LangChain이 대신 해주는 것 | Layer 2에서 개발자가 직접 하는 일 |
|---|---|
| `@tool` | JSON schema 수동 작성 |
| `bind_tools()` | tools payload 직접 전달 |
| 메시지 클래스 | raw dict role/content 직접 관리 |
| `tool.invoke()` | 함수 디스패치 dict와 직접 호출 |
| 공통 ChatModel 인터페이스 | provider SDK 직접 import |
| tracing 연동 | 함수별 `@traceable` 수동 부착 |

> 💡 **실무 포인트**: Layer 2의 결론은 "raw가 무조건 나쁘다"가 아니다. 결론은 **프레임워크 추상화가 제거하는 비용이 무엇인지 정확히 이해한 뒤, 어느 수준에서 추상화를 멈출지 선택하라**는 것이다.

---

## 13. 핵심 요약

```text
✅ Layer 2는 LangChain 없이 Raw Function Calling Agent를 구현하는 단계다.
✅ @tool이 사라지면 Tool schema JSON을 직접 작성해야 한다.
✅ bind_tools()가 사라지면 tools payload를 provider SDK 호출에 직접 넣어야 한다.
✅ 메시지 클래스가 사라지면 role/content dict를 직접 관리해야 한다.
✅ tool.invoke()가 사라지면 함수 이름 기반 dispatch와 kwargs 실행을 직접 구현해야 한다.
✅ trace도 자동이 아니라 @traceable을 함수별로 수동 부착해야 한다.
✅ raw SDK 방식은 provider-specific 비용이 높고, 벤더 교체 비용도 커진다.
✅ 이 레이어를 이해해야 LangChain 추상화의 진짜 가치를 체감할 수 있다.
```

---

## 14. 실습 체크리스트

- [ ] `tools_for_llm`의 `description`을 일부러 모호하게 바꾸고 Tool 선택 품질이 어떻게 흔들리는지 관찰하기
- [ ] 함수 시그니처를 바꾼 뒤 schema를 안 바꿔 drift 상황을 재현해보기
- [ ] `tool_calls[0]` 대신 반복문으로 바꿔 병렬 Tool Calling 확장 초안 작성하기
- [ ] `ollama.chat()` 호출 부분을 다른 provider raw SDK로 바꾼다고 가정하고 어떤 부분이 깨지는지 목록화하기
- [ ] Layer 1 코드와 나란히 비교하며 어떤 줄들이 LangChain 추상화였는지 표시해보기
- [ ] `tools_for_llm`를 제거하고 함수 자체를 tools로 전달하는 실험을 해본 뒤 동작 차이를 관찰하기
- [ ] LangSmith trace에서 LLM run과 tool run이 수동 trace로 어떻게 분리되어 보이는지 확인하기

