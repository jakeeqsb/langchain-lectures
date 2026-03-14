# Section 5 — Guided 학습 노트: ReAct 에이전트 밑바닥부터 구현하기

> **학습 방식**: 개념 이해 → 내부 코드 조립 → 예시 실습 → 확인 순으로 진행한다.
> **참고 노트**: `section5/1.lecture_note.md`, `section5/2.lecture_note.md`, `section5/3.lecture_note.md`
> **사용 코드**: `section5/main.py`, `section5/callback.py`
> **사용 패키지**: `langchain`, `langchain-openai`, `langchain-core`

---

## 🎯 Section 5의 핵심 학습 목표

LangChain은 `create_agent`라는 강력한 마법(추상화)을 제공하지만, 마법의 내부 동작을 모르면 오류(할루시네이션, 무한 루프, 파싱 에러)가 났을 때 원인을 찾을 수 없습니다.
이 섹션의 목적은 **에이전트라는 블랙박스를 박살 내고, 순수 파이썬 루프와 프롬프트로만 에이전트를 조립해 보는 것**입니다. 사용자가 직접 구성한 `main.py` 및 `callback.py`를 기반으로 아주 상세하게 알아봅니다.

---

## STAGE 1. 도구(Tool) 준비와 프롬프트 세팅

### 🧩 개념

파이썬 함수를 LLM이 인식할 수 있는 "도구 스펙"으로 변환시키고, 런타임에 에이전트가 어떤 도구를 호출해야 하는지 가이드합니다.

### 💻 실제 코드 해설 (`main.py` 핵심 추출)

```python
@tool
def get_text_length(text: str) -> int:
    """Returns the length of a text by characters"""
    print(f"\n[Tool Execution] get_text_length enter with text='{text}'")
    # LLM이 파라미터 양끝에 전달할 수 있는 불필요한 따옴표나 줄바꿈 제거 (방어 제어 로직 필수!)
    text = text.strip("'\n\"")
    return len(text)

def find_tool_by_name(tools: List[Tool], tool_name: str) -> Tool:
    """AgentAction에서 추출한 문자열 이름(tool_name)과 매칭되는 실제 Tool 인스턴스를 찾아 리턴합니다."""
    for t in tools:
        if t.name == tool_name:
            return t
    raise ValueError(f"Tool with name {tool_name} not found")
```

> 💡 **실무 포인트: 방어 로직의 중요성**
> `get_text_length` 안의 `strip()` 로직을 주목하세요. LLM은 종종 `"DOG"` 또는 `'DOG'` 형태로 문자열을 뱉거나 뒤에 `\n`을 붙입니다. 이런 작은 방어 코드가 없으면 파이썬 단에서 에러가 폭발하게 됩니다.

> ⚠️ **할루시네이션 방지 토큰 (Stop Sequence)**
> `llm = ChatOpenAI(..., stop=["\nObservation", "Observation"])`
> 이 설정이 누락된다면? LLM은 자기가 도구를 선택(Action) 해놓고는 혼자 북치고 장구치며 관찰 결과(Observation)까지 상상해버립니다. `stop` 파라미터는 "여기까지만 말하고 파이썬에게 제어권을 넘겨!"라는 절대적인 규칙입니다.

---

## STAGE 2. LLM 흐름 추적 및 기억 공간 구성 (`callback.py` 반영)

### 🧩 개념

에이전트가 "내가 지금 무슨 생각을 하고 있고, 무슨 행동을 했는가"를 잃어버리지 않게 하려면, 계속해서 프롬프트의 `{agent_scratchpad}` 자리에 기록을 치환해서 꼬리에 붙여줍니다.
또한 `callback.py`에 작성된 콜백 핸들러를 통해 LLM이 언제 요청을 받고 무슨 답을 뱉는지 시각적으로 트래킹합니다.

### 💻 실제 코드 해설 (`callback.py` 및 `main.py` 핵심 추출)

**1. 로깅 핸들러 (`callback.py`)**

```python
class AgentCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> Any:
        """LLM에 프롬프트가 전송되는 찰나를 가로채어 터미널에 전문 출력"""
        print(f"***Prompt to LLM was:***\n{prompts[0]}")
        print("*********")

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        """LLM이 막 답변 생성을 완료했을 때 가로채어 출력"""
        print(f"***LLM Response:***\n{response.generations[0][0].text}")
        print("*********")
```

**2. `intermediate_steps`와 Scratchpad 주입 구조 (`main.py`)**

```python
# 처음엔 빈 리스트로 시작
intermediate_steps = []

# Agent 체인에 Scratchpad 치환용 함수 등록
agent = (
    {
        "input": lambda x: x["input"],
        # 파이썬 튜플인 scratchpad를 순수 텍스트 문자열(Action: ... Observation: ...)로 묶어 변환
        "agent_scratchpad": lambda x: format_log_to_str(x["agent_scratchpad"]),
    }
    | prompt
    | llm
    | ReActSingleInputOutputParser()
)
```

> 🔍 **내부 동작 원리**
>
> 1) 최초로 `agent.invoke`가 호출될 때 `intermediate_steps`는 빈 리스트인 `[]` 입니다. 따라서 `format_log_to_str([])` 결과는 **빈 문자열("")**이 되어 프롬프트의 끝이 허전한 상태로 LLM에 던져집니다.
> 2) `AgentCallbackHandler`의 `on_llm_start` 로그를 통해, 실제로 프롬프트 끝의 `Thought:` 뒤에 아무것도 안 붙어있다는 것을 직접 눈으로 확인할 수 있습니다.

---

## STAGE 3. 에이전트 완성 본체: While 무한 루프 (`main.py` 분석)

### 🧩 개념

이제 추상화된 `AgentExecutor` 껍데기를 치워버리고, 사용자가 `main.py`에 직접 짜놓은 순수 파이썬 `while` 루프의 한 줄 한 줄의 의미를 파악합니다.

### 💻 실제 코드 해설 (`main.py` 전체 흐름 해부)

```python
# 1. 종료 객체(AgentFinish)가 나오지 않는 이상 계속 반복
agent_step = ""
while not isinstance(agent_step, AgentFinish):
    
    # 2. 에이전트 체인 실행 (LLM 호출 -> 파싱까지 한 번에 수행됨)
    agent_step: Union[AgentAction, AgentFinish] = agent.invoke({
        "input": "What is the length of the word: DOG",
        "agent_scratchpad": intermediate_steps, # 이전 턴의 기록(액션/결과) 통째로 반영
    })
    
    # 3. 만약 LLM이 "도구를 써야 해" 라고 결정했다면? (AgentAction 파싱됨)
    if isinstance(agent_step, AgentAction):
        tool_name = agent_step.tool                 # 사용할 도구 이름 (ex: "get_text_length")
        tool_to_use = find_tool_by_name(tools, tool_name) # 작성해둔 탐색 함수로 실제 Tool 확보
        tool_input = agent_step.tool_input          # 실행할 파라미터 (ex: "DOG")

        # LLM이 제시한 인자를 받아, 파이썬 레벨에서 강제로 함수(tool_to_use.func)를 실행!
        observation = tool_to_use.func(str(tool_input)) 
        
        # 4. 방금 행한 "액션"과 획득한 "관찰 결과"를 기억 저장소(히스토리)에 묶어 저장
        # -> 이게 다음 루프 때 또다시 agent_scratchpad로 들어갑니다!
        intermediate_steps.append((agent_step, str(observation)))

# 5. 파서가 "난 최종 답을 알아(Final Answer)" 포맷을 인식해 AgentFinish를 뱉었다면 루프 탈출
if isinstance(agent_step, AgentFinish):
    print(agent_step.return_values)
```

> 🔍 **내부 동작 요약: 이게 바로 LangChain Agent의 민낯!**
> 파서가 텍스트 출력을 해석해서 **`AgentAction`을 리턴**하면 ➔ **프레임워크(While 루프)가 이를 가로채 툴을 실행함.** ➔ **실행 결과를 LLM에게 다음 턴에서 다시 먹여줌.**
> 파서가 **`AgentFinish`를 리턴**하면 ➔ **루프 종료 및 답변 완성. (끝!)**
> 복잡해 보였던 "제너레이티브 AI 에이전트"라는 마법이 완벽하게 파이썬 제어문으로 해체되었습니다.

---

## 🚀 실습 및 확인 체크리스트

- [ ] 터미널 창에서 파이썬을 이용해 `python section5/main.py` 를 직접 구동해 보세요 (환경변수 세팅 요망).
- [ ] 터미널 창에 찍힌 `***Prompt to LLM was:***` 로그의 꼬리부분 `Thought:` 아래에, **루프가 두 번째로 돌 때 어떤 텍스트가 추가되어 있는지** 직접 확인하세요. (`Action: get_text_length` 와 `Observation: 3`이 들어가 있어야 정상입니다.)
- [ ] `main.py` 내의 `llm = ChatOpenAI(...)` 파트에서 `stop=["\nObservation", "Observation"]` 배열 인자를 일시적으로 지우고 재실행시켜 할루시네이션 폭주/파싱 에러 현상을 관찰해보세요.
