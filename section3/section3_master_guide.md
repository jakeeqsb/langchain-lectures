# Section 3: LangChain 에이전트 루프 마스터 가이드 (통합본)

> **강의 주제**: AI 에이전트 내부 동작을 레이어별로 파악하고, 바닥부터 직접 구현하여 완벽히 이해하기
> **목표**: 라이브러리 사용법을 넘어 "에이전트가 어떻게 작동하는가"를 가장 깊은 레벨에서 체득한다.
> **사용 패키지**: `langchain`, `langchain-ollama`, `langchain-openai`, `langsmith`

---

## 개요: 레이어별 학습 계획

문서만 보고 `create_agent` 함수 한 줄을 복사해 쓰는 것(Layer 0)은 쉽습니다. 하지만 프로덕션에서 에이전트가 무한 루프에 빠지거나 엉뚱한 행동을 할 때, 내부에서 무슨 일이 벌어지는지 모르면 디버깅은 불가능합니다. 본 가이드는 추상화 레이어를 밖에서 안으로, 그리고 안에서 밖으로 오가며 에이전트의 진화 과정을 완벽히 뜯어봅니다.

| 단계 | 레이어 구분 | 감추는 추상화 (LangChain의 가치) | 개발자가 직접 다루는 것 |
|---|---|---|---|
| **Part 3** | **Layer 0** (현재 최신) | 루프 전체, Tool 실행 파이프라인 | `create_agent` 호출 및 결과 받기 |
| **Part 2** | **Layer 1** (기본 요소) | JSON 스키마 자동 생성 (`@tool`) | `while` 루프 및 상태(`messages`) 핑퐁 관리 |
| **Part 1** | **Layer 2** (원시 API) | 통신 포맷 추상화 | 수십 줄의 JSON 스키마 하드코딩 |
| **Part 1** | **Layer 3** (가스라이팅) | Function Calling 자체 | 정규식(`re`)으로 텍스트 썰기, Scratchpad 관리 |

---

## Part 1. 에이전트의 진화 단계 (왜 날것부터 알아야 하는가?)

### 1-1. [Layer 3] 원시 타임: ReAct 프롬프트와 정규식 파싱

*Function Calling API가 없던 시절, 우리는 어떻게 AI에게 도구를 쥐어줬을까요?*

이 코드는 LangChain 생태계를 폭발시킨 Harrison Chase의 전설적인 오리지널 프롬프트(`hwchase17/react`)를 직접 구현한 것입니다. LangSmith 허브에서 700만 번 이상 다운로드된 가장 핵심적인 프롬프트입니다.

**① `inspect` 모듈로 파이썬 함수 정보 긁어오기**
LLM에게 내가 가진 도구를 설명하기 위해, 파이썬 기본 모듈인 `inspect`를 이용해 함수의 매개변수와 주석(Docstring)을 문자열로 추출합니다.

```python
import inspect

def get_tool_descriptions(tools_dict):
    descriptions = []
    for tool_name, tool_function in tools_dict.items():
        original_function = getattr(tool_function, "__wrapped__", tool_function)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(tool_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")
    return "\n".join(descriptions)
```

**② 프롬프트 엔지니어링의 극한 (`hwchase17/react`)**
위에서 만든 `tool_descriptions`를 프롬프트에 구겨 넣고, LLM에게 엄격한 포맷으로 대답하라고 가스라이팅을 시전합니다.

```text
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

Question: {question}
Thought:
```

**③ 정규표현식(Regex) 의존의 취약성**
LLM이 기특하게 "Action: get_product_price" 라고 텍스트를 뱉어주면, 파이썬에서 정규식을 돌려 이를 뜯어냅니다.

```python
# LLM이 내뱉은 평문 텍스트에서 도구 이름과 인자 파싱 (취약함)
action_match = re.search(r"Action:\s*(.+)", output)
action_input_match = re.search(r"Action Input:\s*(.+)", output)

tool_name = action_match.group(1).strip()
tool_input_raw = action_input_match.group(1).strip()
```

> 👨‍💻 **개발자 리포트 (Layer 3의 한계)**:
> LLM이 실수로 "Action :" 처럼 띄어쓰기를 한 칸 더 하거나, 마크다운 코드블록 안쪽에 답을 적어버리면 파이썬 `re.search`는 즉각 `None`을 반환하고 프로그램이 뻗어버립니다(Crash). 이 파멸적 신뢰성 때문에 상용 서비스에 적용하기가 매우 고통스러웠습니다.

### 1-2. [Layer 2] 안정화 시대: Raw Function Calling

OpenAI를 필두로 모델 벤더들이 **Function Calling API**를 전격 출시합니다. 텍스트 파싱 에러의 악몽은 사라졌지만, **어마어마한 양의 JSON 중첩 구조 하드코딩 노가다**가 시작되었습니다.

```python
# LangChain 없이 OpenAI/Ollama API에 밀어넣어야 하는 수동 JSON 스키마
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount tier to a price and return the final price.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {"type": "number", "description": "The original price"},
                    "discount_tier": {"type": "string", "description": "The discount tier"}
                },
                "required": ["price", "discount_tier"],
            },
        },
    }
]

# 통신할 때 롤(role)도 수동으로 끼워넣으며 메모리(messages)를 관리해야 함
observation = apply_discount(price=1299.99, discount_tier="gold")
messages.append({"role": "tool", "content": str(observation)})
```

> 👨‍💻 **개발자 리포트 (Layer 2의 한계)**:
> 함수가 하나 추가될 때마다 개발자는 JSON 스키마를 십수 줄씩 타이핑해야 했고, 실제 파이썬 구현체와 JSON 스키마 간의 정보 불일치 버그가 넘쳐났습니다. 또한 LLM 호출을 추적하기 위해 수동으로 `@traceable`을 덕지덕지 붙여야만 LangSmith에 기록이 남았습니다.

---

## Part 2. LangChain 요소로 Agent Loop 딥다이브 (Layer 1)

고통스러웠던 Layer 2, 3를 경험했으니, 이제 LangChain의 기본 추상화 도구들이 얼마나 혁신적인지 체감할 수 있습니다. 뼈대가 되는 핵심 로직(while 루프)과 요소(`@tool`, `bind_tools`)를 직접 조립해 에이전트를 만들어봅시다.

### 2-1. `init_chat_model` — 다형성과 공급자 종속성 탈피

문자열 하나로 어떤 벤더사의 모델이든 동일한 인터페이스(`BaseChatModel`)로 초기화하는 팩토리 패턴입니다.

```python
from langchain.chat_models import init_chat_model

# 로컬 Ollama 모델 (무료 개발)
llm = init_chat_model("ollama:qwen2.5:7b", temperature=0)

# 프로덕션 적용 시 OpenAI로 교체 (코드 수정 불필요)
# llm = init_chat_model("openai:gpt-4o", temperature=0)
```

> 🔍 **내부 동작 원리**:
> "ollama:qwen2.5:7b" 파싱 시 `langchain-ollama` 패키지를 동적 로딩하여 인스턴스를 생성합니다. 만들어진 객체는 모두 `BaseChatModel` 규격을 따르므로, 프롬프트 체인이나 에이전트 루프 로직을 단 한 줄도 고치지 않고 엔진을 통째로 갈아끼울 수 있습니다.

### 2-2. `@tool` 데코레이터의 흑마법 — 스키마 자동 생성

Layer 2에서 수백 줄 작성하던 그 끔찍한 JSON 스키마를 단 한 줄의 데코레이터로 압축합니다. **"파이썬 코드가 곧 진실(Truth)"**이 되는 순간입니다.

```python
from langchain.tools import tool

@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog.
    
    Args:
        product: 조회할 제품 이름 (예: laptop, headphones)
    """
    prices = {"laptop": 1299.99, "headphones": 149.95}
    return prices.get(product, 0)
```

**`@tool` 파이썬 ↔ JSON 매핑 원리**

| 파이썬 코드 요소 | 변환 매핑 영역 | 모델 지시어 역할 |
|---|---|---|
| **함수명** (`get_product_price`) | JSON `"name"` | 고유 식별자 |
| **Docstring 1번째 줄** | JSON `"description"` | **"어느 상황에 이 도구를 써야 하는가" (매뉴얼)** |
| **타입 힌팅 (`str`)** | JSON 파라미터 `"type"` | 인자값 타입 검증 (`"string"`으로 치환) |
| **파라미터 기본값 유무** | JSON `"required"` 배정 | 필수로 넣어야 할 인자 강제 |

> ⚠️ **개발자 치명적 실수 지점**: 타입 힌팅과 Docstring을 누락하면, LLM에게 장님 안경을 씌우는 것과 같습니다. 스키마가 뭉개져 상황에 안 맞는 이상한 매개변수를 생성해버립니다.

### 2-3. `bind_tools` — 모델과 Tool의 결합

LLM 인스턴스의 API Payload마다 Tool의 전체 JSON 스키마를 "항상 첨부"하도록 세팅합니다.

```python
tools = [get_product_price]
tools_dict = {t.name: t for t in tools} # 빠른 함수 매핑을 위한 해시맵

llm_with_tools = llm.bind_tools(tools)
```

> ⚠️ **주의**: `bind_tools()` 과정을 누락하고 `llm.invoke()`를 호출하면, 모델은 세상에 도구라는 게 존재하는지도 모른 채 무조건 Final Answer만 뱉으며 즉시 종료됩니다.

### 2-4. 메시지 구조와 방어적 프롬프팅

`messages` 리스트는 단순한 문자열 모음이 아니라, **에이전트의 뇌(Memory)이자 유한 상태 기계(State Machine)의 스냅샷**입니다.

```python
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

messages = [
    SystemMessage(content=(
        "You are a helpful shopping assistant.\n\n"
        "STRICT RULES:\n"
        "1. NEVER guess prices — always call get_product_price first.\n"
        "2. NEVER calculate discounts yourself — always use apply_discount tool."
    )),
    HumanMessage(content="What is the price of a laptop after gold discount?"),
]
```

> 💡 오픈 웨이트 소형 모델일수록 (예: Qwen 7B) 환각(가격 지어내기) 방지를 위해 `STRICT RULES` 와 같이 매우 공격적이고 구체적인 제약 사항을 `SystemMessage`에 달아주어야 합니다.

### 2-5. Agent Loop 완전 구현체 및 매핑 메커니즘

이 부분이 LangChain `AgentExecutor` 의 알맹이며, 모든 현대적 에이전트의 심장입니다.

```python
MAX_ITERATIONS = 10   # 무한 루프 방지
for iteration in range(1, MAX_ITERATIONS + 1):
    
    # 1. [Thought] LLM 호출 → "다음에 뭘 할지" 결정
    ai_message = llm_with_tools.invoke(messages)
    tool_calls = ai_message.tool_calls

    # 2. [Final Answer] 쓸 도구가 없으면 → 최종 답변 반환
    if not tool_calls:
        return ai_message.content

    # 3. [Action] Tool 이름과 인자 추출
    tool_call    = tool_calls[0]
    tool_name    = tool_call.get("name")
    tool_args    = tool_call.get("args", {})
    tool_call_id = tool_call.get("id")  # ★★★ 매우 중요 (요청 번호표)

    # 4. [Execute] 파이썬 로컬에서 Tool 실제 실행
    tool_to_use = tools_dict.get(tool_name)
    observation = tool_to_use.invoke(tool_args)

    # 5. [Observe] 결과를 messages에 추가 → LLM이 다음 번 호출에서 문맥을 이해함
    messages.append(ai_message)                      # AI가 "나 이 도구 쓸게"라고 선언한 증거
    messages.append(ToolMessage(
        content=str(observation),
        tool_call_id=tool_call_id                    # 내가 방금 실행한 결괏값이 어떤 요청 번호표에 속하는지 제출
    ))
```

> 🔍 **내부 동작 원리**: `tool_calls`와 `tool_call_id`의 엄격한 매핑 메커니즘
>
> 1. **LLM의 도구 사용 발급서 (`AIMessage`)**
>    LLM이 도구가 필요하다고 판단하면 단순 텍스트가 아닌 특수한 JSON 배열을 반환합니다.
>    `ai_message.tool_calls = [{"name": "get_product_price", "id": "call_abc123"}]`
>    여기서 **`id`(`"call_abc123"`)**가 핵심입니다. LLM이 발급한 "요청 번호표"입니다.
>
> 2. **파이썬의 도구 실행 및 결과 반환 (`ToolMessage`)**
>    파이썬은 함수를 실행한 후 결과값만 띡 하고 LLM에게 되돌려주면 안 됩니다. LLM은 이게 무슨 값인지 모릅니다. 반드시 **`ToolMessage(content="결과값", tool_call_id="call_abc123")`** 형태로, 아까 받은 "번호표"를 그대로 붙여서 LLM에게 돌려줘야 합니다.
>
> 👨‍💻 **치명적 에러 포인트**:
> 만약 `ai_message`에 적힌 `id`와 이후에 push되는 `ToolMessage`의 `tool_call_id`가 단 1글자라도 불일치하거나 누락되면, 다음 Iteration에서 `invoke()`를 호출하는 즉시 **OpenAI API 단에서 `invalid_request_error: Invalid tool message` 에러를 뱉으며 프로그램이 터져버립니다.** 병렬 도구 호출(Parallel Tool Calling) 시에는 여러 개의 ID가 나오므로 순서까지 완벽하게 맞춰서 핑퐁해줘야 합니다.

**[상태 업데이트 과정 예시(메시지 리스트)]**

- Iter 1 시작: `[System, Human]`
- Iter 2 시작: `[System, Human, AI("laptop가격조회_id"), Toolmsg("1299.99_id")]`
- Iter 3 시작: `[System, Human, AI("..."), Tool("..."), AI("할인적용_id2"), Toolmsg("1000.99_id2")]`
  - Iter 3에서 LLM은 이 전체 리스트를 보고 "아 다 끝났네" 하고 Final Answer를 도출합니다.

### 2-6. LangSmith 트레이싱 — `@traceable`

루프 함수 윗줄에 `@traceable`을 달고, `.env` 파일에 `LANGCHAIN_TRACING_V2=true`를 켜면 마법이 일어납니다.
LangChain의 내부 콜백이 llm.invoke()와 tool.invoke()를 모조리 캐치해서 LangSmith 웹 UI 트리 구조 스레드로 올려줍니다. 프로덕션 환경의 AI엔지니어에게 "왜 LLM이 환각을 일으켰는가?"를 파악하기 위한 트레이스 분석은 선택이 아닌 필수 생존 기술입니다.

---

## Part 3. LangGraph와 최신 `create_agent` 마이그레이션 (Layer 0)

우리는 위 Part 2에서 거대한 파이썬 `for` / `while` 루프를 내 손으로 짰습니다. 과거 LangChain의 `AgentExecutor`라는 클래스는 이 루프를 통째로 래핑한 객체였습니다. 이 레거시가 왜 죽고, 단 한 줄짜리 `create_agent` 인터페이스로 오게 되었는지 파악해야 합니다.

### 3-1. `AgentExecutor` (레거시 무한 루프)의 죽음

엔터프라이즈 환경에서는 단순한 Q&A봇을 넘어 복잡한 요구사항이 몰아쳤습니다.

- **Human-in-the-loop**: 결제 도구(Tool)를 실행하기 직전에 딱 멈춰놓고 사람의 승인 버튼을 넣고 싶다.
- **Time-Travel**: 5번째 루프에서 에러가 났을 때, 3번째 루프로 상태 공간을 스냅샷 떠서 돌아가고 싶다.

거대한 블랙박스 형태의 `while` 코드 덩어리(`AgentExecutor`)로는 이 정밀한 제어가 아예 불가능했습니다.

### 3-2. LangGraph 커널 채택 및 `create_agent`

그 결과, LangChain 팀은 에이전트 엔진을 통째로 뜯어고쳐 **유한 상태 기계(State Machine)** 기반의 `LangGraph` 커널로 교체합니다.

```python
# ✅ 신버전 코드 — LangChain v1 스타일
from langchain.agents import create_agent
from schemas import AgentResponse

agent = create_agent(
    model=llm,
    tools=tools,
    response_format=AgentResponse,  # Pydantic 구조화 출력 강제!
)

# 입력은 {"input": ...} 대신 메시지 리스트 직접 삽입
result = agent.invoke({
    "messages": [{"role": "user", "content": "search for AI engineer jobs"}]
})

print(result.get("structured_response"))
```

**[내부 동작 노드 맵]**

```text
[START] ─→ (Agent 노드: LLM 추론) ─→ (액션이 필요한가?)
                  ↑                          │ (Yes)
                  └──── (Tool 노드: 도구 실행) ◀─┘
                                             │ (No)
                                             ▼
                                           [END] (구조화 파싱 종결)
```

이제 에이전트는 [Agent Node]와 [Tool Node] 사이를 간선(Edge)으로 오고 가며, 매 순간 `messages` 데이터를 State 스냅샷으로 남기는 투명한 기계가 되었습니다.

### 3-3. `response_format`과 구조화 은닉

구버전에서는 Pydantic 구조화된 출력을 뽑아내기 위해, 개발자가 `체인 = agent_executor | extract_output | structured_llm` 같은 귀찮은 LCEL 파이프라인 조립(`|`)을 해야 했습니다.

최신 `create_agent`에서는 `response_format=AgentResponse` 인자 하나면 끝납니다. 내부 Graph가 `[END]` 노드 직전에 LLM에게 해당 객체의 JSON 스키마를 강제하고, 떨어진 문자열을 파이썬 Pydantic 객체로 역직렬화(Deserialize)까지 완벽하게 대신 처리해줍니다.

### 마이그레이션 핵심 이유 요약

| 구버전 요소 (`AgentExecutor`) | 신버전 대체 (`create_agent`) | 근본적인 제거/변경 이유 (👨‍💻 개발자 포인트) |
| --- | --- | --- |
| `hub.pull("hwchase17/react")` | **프롬프트 통째로 삭제** | OpenAI 네이티브 **Function Calling API**가 강제되므로, 텍스트를 파싱하려 가스라이팅하던 레거시가 완벽히 증발함. |
| `AgentExecutor` | **LangGraph 컴파일 객체** | 파이썬의 단순 `while` 루프 방식은 제어의 한계가 명확함. 노드(Node)와 간선(Edge)의 **상태 기계(LangGraph)**로 100% 교체됨. |
| LCEL 파이프 조립 (`\|`) | **자동 처리 (response_format)** | 출력값을 추출하고 Pydantic 객체로 포장하는 지루한 작업이 `[END]` 노드의 직렬화 메커니즘 속으로 숨어 개발자가 편해짐. |
| `result["output"]` | `result["structured_response"]` | Type-safe하게 떨어진 인스턴스를 즉각 빼내서 쓸 수 있게 접근 키 변경 |

> 💡 **실무 마이그레이션 팁**: 업데이트가 잦은 LangChain 환경에서는 신버전 공식 문서를 `.txt`나 마크다운으로 복사하여 Cursor나 Claude 같은 AI 에디터의 컨텍스트로 던져주는 것이 마이그레이션 시 가장 생산성이 높습니다.

---

### 최종 학습 점검 (나침반)

개발자로서 팀원에게 아래 항목들을 명확히 설명할 수 있다면, Section 3를 완벽히 마스터한 것입니다.

- [ ] ReAct 방식(정규표현식 파싱)이 왜 도태되었으며, Function Calling은 어떤 파멸적 한계를 극복했는가?
- [ ] 파이썬 `@tool`에서 타입 힌팅과 Docstring을 대충 썼을 때, 파싱 스키마 안에서는 무슨 일이 일어나는가?
- [ ] 모델 초기화 후 `bind_tools()`를 실수로 누락 시킨다면, 에이전트 루프는 1회차에 어떻게 오작동하고 종료되는가?
- [ ] 통신 과정에서 LLM이 뱉은 `ai_message`의 id와, 파이썬이 올리는 `ToolMessage`의 `tool_call_id`가 엇갈릴 때 OpenAI 서버는 왜 크래시를 뱉는가?
- [ ] `AgentExecutor`(초기 LangChain)의 `while` 구문은 어떤 복잡도로 인해 한계에 부딪혔으며, 그를 흡수한 `LangGraph`는 어떤 아키텍처적 이점을 보유하고 있는가?
