# Section 3 — Layer 3: ReAct Prompt 강의 노트

> **강의 주제**: Function Calling 없이, ReAct 프롬프트와 텍스트 파싱만으로 Agent를 직접 구현하기
> **난이도**: 고급
> **목표**: 현대 Tool Calling Agent의 뿌리인 ReAct 패턴을 가장 원초적인 형태로 이해하기
> **사용 패키지**: `ollama`, `langsmith`, `python-dotenv`, `re`, `inspect`
> **권장 모델**: 지시 이행 능력이 괜찮은 모델 (`qwen3:1.7b` 이상 권장)

---

## 1. 왜 Layer 3까지 내려가야 할까?

> **"Layer 2에서 이미 JSON schema까지 직접 썼는데, 왜 Function Calling 자체를 버리나요?" (Why)**
> Layer 2는 Tool Calling의 raw protocol을 보여줬다. Layer 3는 그보다 더 깊다. 여기서는 아예 API 차원의 Tool Calling 지원이 없다고 가정하고, **오직 프롬프트와 문자열 파싱만으로 에이전트를 만든다.** 이 단계를 이해해야 ReAct가 왜 에이전트의 원형인지, 그리고 Function Calling이 어떤 문제를 해결해준 것인지 정확히 보인다.

Layer 3의 핵심은 이거다.

```text
LLM에게 "이 형식으로 생각하고 행동하라"고 프롬프트로 강제한다.
↓
LLM은 텍스트로 Thought / Action / Action Input 을 출력한다.
↓
파이썬이 정규식으로 그 텍스트를 파싱한다.
↓
도구를 실행한 뒤 Observation을 다시 프롬프트에 이어붙인다.
↓
이 문자열 히스토리를 반복하며 Final Answer를 유도한다.
```

> 🔍 **내부 동작 원리**:
>
> ```text
> [Prompt]
>   "Action: ..., Action Input: ... 형식으로 출력해"
>      ↓
> [LLM 텍스트 응답]
>   Thought: ...
>   Action: get_product_price
>   Action Input: laptop
>      ↓
> [Python regex parsing]
>   tool_name = "get_product_price"
>   tool_input = "laptop"
>      ↓
> [도구 실행]
>   Observation: 1299.99
>      ↓
> [Scratchpad 갱신]
>   이전 Thought/Action/Observation 기록 누적
> ```

> 💡 **실무 포인트**: 요즘 대부분의 상용 Agent는 Function Calling을 쓰지만, Layer 3를 이해하면 "에이전트의 본질은 결국 prompt-controlled loop"라는 감각이 생긴다. 이 감각이 있어야 프레임워크 디버깅도 잘 된다.

🔗 [ReAct 논문](https://arxiv.org/abs/2210.03629)  
🔗 [LangChain Agents 공식 문서](https://docs.langchain.com/oss/python/langchain/agents)

---

## 2. ReAct Prompt가 왜 중요한가?

강의에서 말하는 ReAct Prompt는 단순한 템플릿이 아니다.  
이건 **LLM을 reasoning engine처럼 보이게 만드는 운영 규약**이다.

ReAct는 이름 그대로:

- **Reason**: 지금 무엇을 해야 하는지 생각한다.
- **Act**: 필요한 Tool을 선택해 실행한다.

즉, 모델이 그냥 답변만 생성하는 게 아니라,
**생각 -> 행동 -> 결과 반영 -> 다시 생각** 구조로 움직이게 만든다.

```text
Question
Thought
Action
Action Input
Observation
Thought
...
Final Answer
```

> 💡 **실무 포인트**: `Observation`이라는 용어 자체가 ReAct 문맥에서 강하게 자리잡았다. 지금 우리가 Tool 결과를 observation이라고 부르는 습관도 여기서 왔다.

---

## 3. 전체 코드 먼저 보기

```python
import re
import inspect
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
    price = float(price)
    discount_percentages = {"bronze": 5, "silver": 12, "gold": 23}
    discount = discount_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)


tool_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES — you must follow these exactly:
1. NEVER guess or assume any product price. You MUST call get_product_price first.
2. Only call apply_discount AFTER you have received a price from get_product_price.
3. NEVER calculate discounts yourself using math. Always use the apply_discount tool.
4. If the user does not specify a discount tier, ask them which tier to use.

Answer the following questions as best you can. You have access to the following tools:

{tool_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat_traced(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


@traceable(name="Ollama Agent Loop")
def run_agent(question: str):
    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        full_prompt = prompt + scratchpad

        response = ollama_chat_traced(
            model=MODEL,
            messages=[{"role": "user", "content": full_prompt}],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        print(f"LLM Output:\n{output}")

        final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"Final Answer: {final_answer}")
            return final_answer

        action_match = re.search(r"Action:\s*(.+)", output)
        action_input_match = re.search(r"Action Input:\s*(.+)", output)

        if not action_match or not action_input_match:
            print("ERROR: Could not parse Action/Action Input")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        if tool_name not in tools:
            observation = f"Error: Tool '{tool_name}' not found."
        else:
            observation = str(tools[tool_name](*args))

        scratchpad += f"{output}\nObservation: {observation}\nThought:"

    print("ERROR: Max iterations reached without a final answer")
    return None
```

> ⚠️ **버전 주의**: 이 구현은 ReAct 개념을 설명하기 위한 학습용 코드다. 모델의 출력 형식, stop token 동작, 메시지 구조는 모델/SDK 버전에 따라 달라질 수 있다.

🔗 [Ollama Python 라이브러리](https://github.com/ollama/ollama-python)  
🔗 [LangSmith `traceable` 문서](https://docs.smith.langchain.com/reference/python/run_helpers/langsmith.run_helpers.traceable)

---

## 4. Layer 1, 2, 3의 차이

| 레이어 | Tool 선택 방식 | Tool 실행 신호 | 파싱 방식 | 안정성 |
|---|---|---|---|---|
| Layer 1 | LangChain `bind_tools()` | `ai_message.tool_calls` | 구조화된 dict | 높음 |
| Layer 2 | Raw provider Tool Calling | `response.message.tool_calls` | provider 객체 파싱 | 중간 |
| Layer 3 | ReAct prompt | 텍스트 안의 `Action:` | `regex` | 낮음 |

Layer 3는 가장 불편하지만, 동시에 가장 본질적이다.  
여기서는 모델이 "도구를 호출하고 있다"는 사실조차 API가 보장해주지 않는다.
전부 텍스트 패턴 약속으로 성립한다.

> 💡 **실무 포인트**: Layer 3는 오늘날 기준으로는 운영용 기본값이라기보다, Agent 동작의 원리를 가장 잘 보여주는 교육용 레이어다. 다만 Function Calling이 약한 모델, 특수 환경, 연구 실험에서는 여전히 유효하다.

---

## 5. `inspect`로 Tool 설명을 프롬프트에 주입하는 이유

> **"Function Calling이 없는데 모델은 Tool 정보를 어디서 알죠?" (Why)**
> Layer 3에서는 Tool schema를 API에 전달하지 않는다. 대신 프롬프트 안에 Tool 목록과 시그니처, 설명을 텍스트로 삽입해 "이런 도구들이 있다"고 알려준다.

```python
def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)
```

> 🔍 **내부 동작 원리**:
>
> ```text
> Python 함수
>   ├─ 함수 이름
>   ├─ 시그니처
>   └─ docstring
>      ↓
> 문자열로 직렬화
>      ↓
> prompt 안에 삽입
>      ↓
> LLM이 "사용 가능한 도구 설명서"처럼 읽음
> ```

### `__wrapped__`를 쓰는 이유

`@traceable` 같은 데코레이터를 쓰면 원래 함수 시그니처가 감싸질 수 있다.  
그래서 `__wrapped__`를 통해 원본 함수에 접근해 시그니처를 복원한다.

> 💡 **실무 포인트**: 이 부분은 Layer 3에서 특히 중요하다. Tool Calling API가 없기 때문에, Tool 설명을 얼마나 정확하게 prompt에 넣느냐가 모델의 행동 품질을 직접 좌우한다.

---

## 6. ReAct Prompt 자체가 에이전트의 "프로토콜"이다

> **"왜 프롬프트 안에 형식을 이렇게 빡세게 적어두나요?" (Why)**
> Layer 3에서는 API가 구조를 강제하지 않는다. 따라서 Thought, Action, Action Input, Observation, Final Answer 형식을 prompt가 대신 강제해야 한다. 즉, prompt가 곧 protocol이다.

```text
Use the following format:

Question: ...
Thought: ...
Action: ...
Action Input: ...
Observation: ...
...
Final Answer: ...
```

> 🔍 **내부 동작 원리**:
>
> ```text
> Function Calling world
>   → API가 구조를 강제
>
> ReAct Prompt world
>   → Prompt가 구조를 강제
>   → Python은 그 형식을 regex로 믿고 파싱
> ```

> 💡 **실무 포인트**: ReAct prompt의 힘은 "생각을 하라"가 아니라, **생각/행동/관찰의 인터페이스를 텍스트 규약으로 만들었다**는 데 있다. 그래서 이 패턴이 에이전트 역사에서 결정적이었다.

---

## 7. `scratchpad` — 에이전트의 텍스트 기반 메모리

> **"agent_scratchpad는 정확히 뭐죠?" (Why)**
> scratchpad는 지금까지의 Thought / Action / Observation 기록을 누적한 문자열이다. 다음 iteration에서 모델은 이 기록을 다시 읽고, 현재까지 무슨 일이 벌어졌는지 이해한 뒤 다음 행동을 정한다.

```python
prompt = react_prompt.format(question=question)
scratchpad = ""
full_prompt = prompt + scratchpad
```

Tool 실행 후에는 이렇게 누적된다.

```python
scratchpad += f"{output}\nObservation: {observation}\nThought:"
```

> 🔍 **내부 동작 원리**:
>
> ```text
> Iteration 1
>   Thought: 가격을 알아야 한다
>   Action: get_product_price
>   Action Input: laptop
>   Observation: 1299.99
>
> Iteration 2
>   위 기록 전체를 다시 prompt 뒤에 붙여서 재전송
>   → 모델은 "이제 할인 계산 Tool을 써야겠네"라고 판단
> ```

> 💡 **실무 포인트**: Layer 1과 2에서는 `messages` 리스트가 메모리였다면, Layer 3에서는 `scratchpad` 문자열이 메모리다. 즉, 상태 저장 방식만 바뀌었을 뿐 루프의 본질은 같다.

---

## 8. `stop=["\nObservation"]`이 중요한 이유

> **"왜 stop token을 굳이 걸어두나요?" (Why)**
> 모델이 자기 멋대로 `Observation:`까지 생성해버리면, 실제 도구 실행 결과와 가짜 observation이 섞여버린다. Layer 3에서는 파이썬이 진짜 도구 결과를 주입해야 하므로, 모델 출력은 `Action Input`까지만 끊는 게 안전하다.

```python
response = ollama_chat_traced(
    model=MODEL,
    messages=[{"role": "user", "content": full_prompt}],
    options={"stop": ["\nObservation"], "temperature": 0},
)
```

> 🔍 **내부 동작 원리**:
>
> ```text
> LLM 출력 목표:
>   Thought: ...
>   Action: ...
>   Action Input: ...
>
> 여기서 generation stop
>   ↓
> Python이 실제 Observation을 삽입
> ```

> 💡 **실무 포인트**: stop token은 Layer 3에서 꽤 중요한 안정화 장치다. 없으면 모델이 허구의 observation을 만들어 self-delusion 상태에 빠질 수 있다.

---

## 9. Regex 파싱 — Layer 3의 가장 취약한 지점

> **"왜 ReAct Agent가 예전엔 그렇게 불안정했나요?" (Why)**
> Action과 Action Input을 **텍스트에서 정규식으로 뜯어내야 했기 때문**이다. 모델이 형식을 조금만 어겨도 파서가 무너진다.

```python
action_match = re.search(r"Action:\s*(.+)", output)
action_input_match = re.search(r"Action Input:\s*(.+)", output)

if not action_match or not action_input_match:
    print("ERROR: Could not parse Action/Action Input")
    break
```

> 🔍 **내부 동작 원리**:
>
> ```text
> 정상 출력:
>   Action: get_product_price
>   Action Input: laptop
>
> 파싱 실패 예:
>   - Action 대신 Act: 사용
>   - Action Input 줄 누락
>   - Markdown 코드블록 안에 출력
>   - 여러 줄 포맷 어긋남
> ```

> 💡 **실무 포인트**: Layer 3는 "LLM이 텍스트 규약을 얼마나 잘 지키는가"에 크게 의존한다. 그래서 Function Calling API가 등장하자마자 업계가 빠르게 그쪽으로 이동했다.

> ⚠️ **주의**: 이 구현은 쉼표 기반 인자 분리와 단순 문자열 strip을 사용한다. 복잡한 JSON 인자, 중첩 구조, quoted comma 같은 케이스에서는 쉽게 깨질 수 있다.

---

## 10. Final Answer 판별도 텍스트 규약이다

> **"종료 조건도 Tool Calling처럼 구조화되어 있지 않나요?" (Why)**
> 맞다. Layer 3에서는 종료 신호도 텍스트 안에서 찾아야 한다. 즉, `tool_calls == []` 같은 구조화된 종료 조건이 없고, `Final Answer:` 문자열이 종료 프로토콜이다.

```python
final_answer_match = re.search(r"Final Answer:\s*(.+)", output)
if final_answer_match:
    final_answer = final_answer_match.group(1).strip()
    return final_answer
```

> 💡 **실무 포인트**: Layer 3의 Agent는 끝까지 prompt agreement 위에서만 동작한다. 형식이 약해지면 시작도, 중간도, 종료도 모두 흔들린다.

---

## 11. 왜 이 레이어가 역사적으로 중요했는가?

ReAct Prompt는 실제로 LangChain 초창기 에이전트 구현의 기반이 된 패턴이고,  
지금도 Agent 설계의 정신적 뿌리로 남아 있다.

이 레이어가 중요한 이유는 세 가지다.

1. Agent를 "툴 호출이 가능한 LLM"이 아니라, **루프를 가진 추론 시스템**으로 보게 만든다.
2. Observation과 scratchpad 개념을 명확히 정착시켰다.
3. Function Calling 이후의 Agent들도 사실상 같은 구조를 더 안정적으로 구현한 것임을 보여준다.

> 💡 **실무 포인트**: 오늘날 `create_agent`, LangGraph, Tool Calling 기반 시스템을 보더라도, 내부 사고 모델은 여전히 ReAct 관점으로 이해하는 게 제일 쉽다.

---

## 12. Layer 3의 장단점

### 장점

- Function Calling API가 없어도 에이전트를 만들 수 있다.
- 어떤 모델이든 텍스트 생성만 되면 원리상 적용 가능하다.
- 에이전트의 본질을 가장 선명하게 드러낸다.

### 단점

- 형식 이탈에 매우 취약하다.
- regex 파싱이 깨지기 쉽다.
- 복잡한 인자 구조를 다루기 어렵다.
- provider가 보장하는 structured tool call 안전성이 없다.

| 구분 | Layer 1/2 Tool Calling | Layer 3 ReAct Prompt |
|---|---|---|
| Tool 선택 신호 | 구조화된 필드 | plain text |
| Tool 인자 | typed / structured | 문자열 파싱 |
| 종료 조건 | `tool_calls` 없음 | `Final Answer:` 텍스트 |
| 안정성 | 높음 | 낮음 |
| 교육적 가치 | 높음 | 매우 높음 |

---

## 13. 핵심 요약

```text
✅ Layer 3는 Function Calling 없이 ReAct 프롬프트만으로 Agent를 구현하는 단계다.
✅ Tool 정보는 API schema가 아니라 prompt 안의 텍스트 설명으로 전달된다.
✅ scratchpad는 Thought / Action / Observation 기록을 누적하는 텍스트 메모리다.
✅ Python은 regex로 Action과 Action Input을 파싱해 Tool을 실행한다.
✅ stop token은 모델이 가짜 Observation을 생성하지 못하게 막는 장치다.
✅ Final Answer도 구조화 신호가 아니라 텍스트 프로토콜이다.
✅ 이 레이어를 이해하면 현대 Agent도 결국 ReAct 루프의 안정화 버전이라는 점이 보인다.
```

---

## 14. 실습 체크리스트

- [ ] `stop=["\nObservation"]`를 제거하고 모델이 가짜 Observation을 쓰는지 확인하기
- [ ] `Action:`이나 `Action Input:` 형식을 일부러 흐리게 만드는 prompt를 넣고 regex 파싱이 어디서 깨지는지 관찰하기
- [ ] `scratchpad`를 출력해 iteration마다 어떤 문자열이 누적되는지 직접 확인하기
- [ ] `tool_descriptions`에서 docstring을 지우고 Tool 선택 품질이 어떻게 떨어지는지 실험하기
- [ ] `apply_discount`에 전달되는 `price` 타입 변환(`float(price)`)을 제거하고 문자열 인자 처리 문제가 생기는지 확인하기
- [ ] Layer 2 코드와 비교하며 "structured tool call이 없을 때 어떤 불편이 생기는지"를 표로 정리해보기
- [ ] 같은 질문을 Layer 1, Layer 2, Layer 3 구현에 각각 넣고 안정성 차이를 비교해보기

