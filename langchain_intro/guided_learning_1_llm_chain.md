# 랭체인입문활용 — Guided 학습 노트: LLM Chain과 LCEL 멀티 체인

> **학습 방식**: 개념 이해 → 예시 코드 확인 → 집중 실습 → 확인 순으로 진행한다.
> **원본 자료**: [1.llm_chain.ipynb](./1.llm_chain.ipynb)
> **사용 패키지**: `langchain`, `langchain-openai`, `langchain-core`

---

## STAGE 1. 기본 LLM 체인 (Prompt + LLM + OutputParser)

### 개념

LangChain의 가장 기본적인 실행 단위는 프롬프트 템플릿, 언어 모델(LLM), 출력 파서의 결합이다.
이 요소들은 LCEL(LangChain Expression Language)의 파이프(`|`) 연산자로 깔끔하게 연결된다.

| 요소 | 역할 | 원본 파일의 API |
|------|------|----------------|
| **Prompt** | 동적 변수(`{input}`)를 받아 모델에 전달할 문자열을 완성 | `ChatPromptTemplate.from_template()` |
| **LLM** | 프롬프트를 입력받아 응답 생성 | `ChatOpenAI(model="gpt-4o-mini")` 또는 `init_chat_model()` |
| **OutputParser** | 모델의 응답 객체(`AIMessage`)에서 문자열만 추출 | `StrOutputParser()` |

### 예시 코드 — 3단 체인

```python
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 1. 컴포넌트 준비
prompt = ChatPromptTemplate.from_template("You are an expert in astronomy. Answer the question. <Question>: {input}")
llm = ChatOpenAI(model="gpt-4o-mini")
output_parser = StrOutputParser()

# 2. 파이프(|) 연산자로 체인 구성
chain = prompt | llm | output_parser

# 3. 체인 실행 (invoke)
result = chain.invoke({"input" : "지구의 자전 주기는 ?"})
print(result) # "지구의 자전 주기는 약 24시간..." 문자열 바로 출력
```

> 🔍 **내부 동작 원리**: LCEL 체인의 데이터 흐름
>
> ```text
> 입력: {"input" : "지구의 자전 주기는 ?"}
>   ↓ (prompt 실행)
> "You are an expert... <Question>: 지구의 자전 주기는 ?"
>   ↓ (llm 실행)
> AIMessage(content="지구의 자전 주기는 약 24시간...", ...)
>   ↓ (output_parser 실행)
> "지구의 자전 주기는 약 24시간..." (순수 문자열 반환)
> ```

> 💡 **실무 포인트**: `StrOutputParser`를 체인 마지막에 두면 `AIMessage` 객체를 파싱하는 번거로움(`response.content`)을 피할 수 있어 코드가 매우 직관적이 된다.

### LCEL과 `Runnable`은 내부적으로 어떻게 돌아갈까?

LCEL은 단순한 파이프 문법이 아니라, **`Runnable` 객체들을 조합해서 실행 그래프를 만드는 방식**이다.
즉 `prompt | llm | output_parser`는 사람이 보기에는 3단 파이프지만, 내부적으로는 `RunnableSequence`라는 순차 실행 객체로 묶인다.

| 개념 | 실제 내부 의미 |
|------|----------------|
| `prompt | llm` | 앞 단계 출력이 다음 단계 입력으로 전달되는 `RunnableSequence` |
| `{"a": chain1, "b": chain2}` | 같은 입력을 각 체인에 동시에 넣는 `RunnableParallel` |
| `RunnablePassthrough.assign(...)` | 기존 `dict` 상태를 유지한 채 새 키를 병합 |
| `RunnableLambda(func)` | 일반 Python 함수를 `Runnable`처럼 체인 안에서 실행 가능하게 래핑 |

핵심은 **프롬프트, 모델, 파서도 모두 `Runnable`처럼 동작한다**는 점이다.
그래서 각 단계가 모두 같은 인터페이스를 공유한다.

### LCEL의 각 요소는 전부 `Runnable`이어야 할까?

정확히 말하면, **실행 시점에는 모두 `Runnable`이어야 한다.**
하지만 작성할 때부터 모든 요소를 직접 `Runnable` 클래스로 만들 필요는 없다.
LCEL이 일부 값을 내부적으로 **자동 변환(coercion)** 해서 `Runnable` 그래프로 바꿔 실행한다.

| 작성한 값 | LCEL 내부 해석 |
|-----------|----------------|
| `prompt`, `llm`, `output_parser` | 이미 `Runnable`처럼 동작하는 실행 노드 |
| `some_python_function` | `RunnableLambda(some_python_function)` |
| `{"a": chain1, "b": chain2}` | `RunnableParallel(a=chain1, b=chain2)` |
| `chain_a | chain_b` | `RunnableSequence(chain_a, chain_b)` |

즉 사람이 쓸 때는 짧고 편한 문법으로 적지만,
LCEL은 그것을 내부적으로 **전부 실행 가능한 `Runnable` 노드들로 정규화한 뒤 실행**한다고 보면 된다.

반대로 문자열, 숫자, 리스트 같은 값은 그 자체로는 실행 노드가 아니므로
그냥 LCEL 단계로 둘 수는 없다. 이런 값은 프롬프트 변수나 함수의 반환값으로 들어가야 한다.

| 공통 메서드 | 의미 |
|------------|------|
| `invoke(input)` | 단일 입력 1개 실행 |
| `ainvoke(input)` | 비동기 단일 실행 |
| `batch(inputs)` | 여러 입력을 한 번에 실행 |
| `stream(input)` | 결과를 스트리밍으로 받기 |

> 🔍 **내부 실행 모델**
>
> ```text
> chain = prompt | llm | output_parser
>
> 내부적으로는 대략 이런 구조다:
> RunnableSequence(
>   first=prompt,
>   middle=[llm],
>   last=output_parser
> )
> ```

> 🔍 **왜 `Runnable` 개념이 중요한가?**
>
> `invoke`, `batch`, `stream`, `with_retry`, `with_config` 같은 공통 기능이
> 프롬프트/모델/파서/커스텀 함수에 동일하게 적용되기 때문이다.
> 즉 LCEL은 "문자열을 이어붙이는 문법"이 아니라
> "실행 가능한 노드들을 표준 인터페이스로 연결하는 방식"이라고 이해하면 된다.

### LCEL은 이걸 내부에서 어떻게 풀어서 실행할까?

LCEL의 동작은 크게 **구성 단계**와 **실행 단계**로 나눠 보면 이해가 쉽다.

#### 1. 구성 단계: 표현식을 `Runnable` 그래프로 변환

예를 들어 아래 코드는:

```python
chain = prompt | llm | output_parser
```

내부적으로는 거의 이런 의미가 된다.

```python
chain = RunnableSequence(
    first=prompt,
    middle=[llm],
    last=output_parser,
)
```

또 아래처럼 dict나 함수를 섞으면:

```python
chain = previous_chain | {
    "summary": summarize_chain,
    "category": classify_text,
}
```

LCEL은 대략 이렇게 해석한다.

```python
chain = previous_chain | RunnableParallel(
    summary=summarize_chain,
    category=RunnableLambda(classify_text),
)
```

즉 `|`, `dict`, `function` 같은 표현을 보고
LCEL이 실행 가능한 형태로 바꿔 **최종적으로는 `Runnable` 그래프**를 만든다.

#### 2. 실행 단계: 앞 단계 출력이 다음 단계 입력으로 전달

이후 `chain.invoke(input)`를 호출하면,
각 노드가 순서대로 실행되면서 출력이 다음 단계의 입력으로 전달된다.

```text
input
  ↓ step1.invoke(input)
output1
  ↓ step2.invoke(output1)
output2
  ↓ step3.invoke(output2)
final_output
```

병렬 노드(`RunnableParallel`)는 같은 입력을 여러 노드에 동시에 넣고,
그 결과를 `dict`로 합쳐 다음 단계로 넘긴다.

```text
input
  ├─ branch_a.invoke(input)
  ├─ branch_b.invoke(input)
  └─ branch_c.invoke(input)
      ↓
{
  "a": result_a,
  "b": result_b,
  "c": result_c
}
```

분기 노드(`RunnableBranch`)는 조건을 앞에서부터 평가하고,
처음으로 참이 되는 체인 하나만 실행한다.

#### 3. 그래서 `invoke`, `batch`, `stream`가 통일된다

모든 단계가 `Runnable` 그래프로 정리되기 때문에
체인 전체에도 동일한 실행 메서드를 적용할 수 있다.

- `invoke()`는 단건 실행
- `batch()`는 여러 입력을 묶어 실행
- `stream()`은 가능한 단계들에 대해 스트리밍 실행

이 점 때문에 LCEL은 단순 문법 설탕이 아니라,
**실행 인터페이스가 통일된 파이프라인 런타임**에 가깝다.

> 💡 **스트리밍에서 주의할 점**
>
> `RunnableSequence`는 각 단계가 스트리밍 가능한 경우 스트림을 이어받을 수 있다.
> 다만 `RunnableLambda`는 기본적으로 `transform` 기반 스트리밍 노드가 아니므로,
> 체인 중간에 두면 그 지점에서 스트리밍이 잠시 끊기거나 마지막까지 모은 뒤 다음 단계로 넘길 수 있다.

🔗 [RunnableSequence 공식 문서](https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.RunnableSequence.html)
🔗 [Runnable 공식 문서](https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.Runnable.html)
🔗 [StrOutputParser 문서](https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html)

---

## STAGE 2. 멀티 체인 1 — 순차 체인 연결

### 개념

하나의 체인 출력을 **다음 프롬프트의 입력으로 직접 연결**할 수 있다. 딕셔너리를 활용해 파이프를 확장한다.

### 예시 코드 — 번역 후 설명하는 체인

```python
from langchain.chat_models import init_chat_model

llm = init_chat_model("gpt-4o-mini")

translate_prompt = ChatPromptTemplate.from_template("다음 한국어를 영어로 번역하세요: {korean_word}")
explain_prompt = ChatPromptTemplate.from_template("다음 영어 단어를 한국어로 자세히 설명하세요: {english_word}")

# 체인 1 (한국어 입력 → 문자열 영어 출력)
chain1 = translate_prompt | llm | StrOutputParser()

# 체인 2 (체인 1의 출력을 english_word로 받음)
chain2 = (
    {"english_word": chain1}
    | explain_prompt 
    | llm
    | StrOutputParser()
)

result = chain2.invoke({"korean_word": "인공지능"})
print(result)
```

✅ **확인된 정보**: 원본 노트북에 명시된 `init_chat_model("gpt-4o-mini")` 사용법을 그대로 적용했다.

> 🔍 **내부 동작 원리**: 순차 체인 연결
>
> ```text
> 입력: {"korean_word": "인공지능"}
>   ↓ chain1 실행
> {"english_word": "artificial intelligence"}
>   ↓ explain_prompt 실행
> "다음 영어 단어를 한국어로 자세히 설명하세요: artificial intelligence"
>   ↓ llm, output_parser 실행
> 최종 한국어 설명 문자열
> ```

> 💡 **여기서 중요한 포인트**
>
> `{"english_word": chain1}` 형태의 딕셔너리는
> "키마다 Runnable을 실행해 결과를 모으는 병렬 매핑"으로 해석된다.
> 즉 LCEL은 단순 dict도 문맥에 따라 `RunnableParallel` 성격으로 처리한다.

🔗 [RunnableSequence 공식 문서](https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.RunnableSequence.html)

---

## STAGE 3. 멀티 체인 2 — `RunnablePassthrough.assign` (다단계 데이터 유지)

### 개념

다단계 체인을 구성할 때, 이전 단계의 정보(예: 원본 키워드)를 다음 단계로 버리지 않고 **유지하면서 새 필드를 추가**해야 할 때가 있다.
이때 `RunnablePassthrough.assign`을 사용해 입력 `dict`에 새로운 키를 추가한다.

### 예시 코드 — 주제 분석 → 개요 작성 → 본문 작성

```python
from langchain_core.runnables import RunnablePassthrough

# ... 생략 (analyze_prompt, outline_prompt, content_prompt 정의) ...

chain = (
    {"topic": RunnablePassthrough()}  # 외부에서 단순 문자열 전달 시 dict로 래핑
    | RunnablePassthrough.assign(
        # topic 값을 유지한 채로 keywords를 추가
        keywords=analyze_prompt | llm | StrOutputParser()
    )
    | RunnablePassthrough.assign(
        # topic, keywords 유지한 채 outline 추가
        outline=outline_prompt | llm | StrOutputParser()
    )
    | content_prompt
    | llm
    | StrOutputParser()
)

# 입력으로 딕셔너리가 아닌 단순 문자열을 인자로 넣음
result = chain.invoke("기후 변화와 지속 가능한 발전") 
```

> 🔍 **내부 동작 원리**: `RunnablePassthrough.assign`
>
> ```text
> 입력: "기후 변화..."
> {"topic": RunnablePassthrough()} 통과 시
> → state: {"topic": "기후 변화..."}
>
> 첫 번째 assign 통과 시
> → 원본 dict를 받아 복사한 후, keywords 키 결과 병합
> → state: {"topic": "기후 변화...", "keywords": "온실가스, 지구온난화, 환경"}
>
> 두 번째 assign 통과 시
> → state: {"topic": "...", "keywords": "...", "outline": "서론: ... 본론: ..."}
> ```

> 💡 **실무 포인트**: 여러 단계의 생성 파이프라인(요약 → 번역 → 교정 등)에서 이전 단계의 원본 데이터가 필요할 때 가장 많이 쓰이는 기법이다.

> 🔍 **왜 `assign`이 중요한가?**
>
> 일반 순차 체인은 "이전 출력 하나"만 다음 단계로 넘기는 흐름이 기본이다.
> 반면 `assign`은 상태 딕셔너리를 계속 유지하므로,
> 여러 중간 산출물(`topic`, `keywords`, `outline`)을 잃지 않고 후속 프롬프트에 함께 넣을 수 있다.

🔗 [RunnablePassthrough 공식 문서](https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html)

---

## STAGE 4. 병렬 체인 — `RunnableParallel`

### 개념

여러 개의 독립적인 질문을 병렬로 처리하고, 그 결과를 딕셔너리로 반환한다. 응답 시간을 크게 절약할 수 있다.

### 예시 코드 — 3가지 관점 동시 분석

```python
from langchain_core.runnables import RunnableParallel

positive_prompt = ChatPromptTemplate.from_template("{topic}의 긍정적인 측면")
negative_prompt = ChatPromptTemplate.from_template("{topic}의 부정적인 측면")

# RunnableParallel 안의 체인들은 동시에 실행된다
parallel_chain = RunnableParallel(
    positive=positive_prompt | llm | StrOutputParser(),
    negative=negative_prompt | llm | StrOutputParser()
)

results = parallel_chain.invoke({"topic": "원격 근무"})
print(results["positive"])
print(results["negative"])
```

> 💡 **실무 포인트**: `RunnableParallel`의 출력은 **딕셔너리**다.
> 이 병렬 체인의 출력을 곧바로 다음 프롬프트(예: 종합 프롬프트)의 입력 딕셔너리로 바로 연결해 종합 (Synthesis) 모델링 패턴을 쉽게 만들 수 있다.

> 🔍 **내부 동작 원리**: `RunnableParallel`
>
> ```text
> 입력: {"topic": "원격 근무"}
>   ├─ positive 체인 실행
>   └─ negative 체인 실행
>   ↓
> 결과 병합:
> {
>   "positive": "...",
>   "negative": "..."
> }
> ```

> 💡 **LCEL에서 자주 놓치는 포인트**
>
> `chain_a | {"x": chain_b, "y": chain_c}` 처럼 dict를 파이프 오른쪽에 두면
> 그 dict는 내부적으로 `RunnableParallel`로 변환된다.
> 즉 "같은 입력을 여러 Runnable에 동시에 넣고 결과를 키별로 모은다"가 기본 동작이다.

🔗 [RunnableParallel 공식 문서](https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.RunnableParallel.html)

---

## STAGE 5. 조건부 분기 라우팅

입력 내용에 따라 실행할 파이프라인(프롬프트)을 나누는 라우팅 기법이다.
원본 코드에서는 두 가지 방법이 제시되었다: `RunnableBranch`와 `RunnableLambda` 기반 라우팅.

### 방법 1 — 특정 조건식 평가 (RunnableBranch)

```python
from langchain_core.runnables import RunnableBranch

def detect_language(input_dict):
    # 한글 포함 여부 체크
    if any('가' <= char <= '힣' for char in input_dict.get("question", "")):
        return "korean"
    return "english"

# if 조건문 형태로 체인 분기
branch_chain = RunnableBranch(
    (lambda x: detect_language(x) == "korean", korean_prompt | llm | StrOutputParser()), # if
    (lambda x: detect_language(x) == "english", english_prompt | llm | StrOutputParser()), # elif
    default_prompt | llm | StrOutputParser() # else (기본값)
)
```

### 방법 2 — 함수를 통한 동적 체인 반환 (RunnableLambda)

라우팅 로직이 복잡해지면 함수 내에서 분기를 계산해 직접 실행할 체인을 invoke하는 것이 유연하다.

```python
from langchain_core.runnables import RunnableLambda

chains_dict = {
    "technical": tech_prompt | llm | StrOutputParser(),
    "general": general_prompt | llm | StrOutputParser()
}

def routing_chain(input_dict):
    topic = input_dict.get("topic", "").lower()
    
    if "code" in topic:
        route = "technical"
    else:
        route = "general"
        
    return chains_dict[route].invoke(input_dict) # 선택된 체인의 invoke 반환

router = RunnableLambda(routing_chain)
```

> 🔍 **내부 동작 원리**: `RunnableBranch` vs `RunnableLambda`
>
> - `RunnableBranch`는 조건식을 순서대로 평가해서 **처음 True가 된 분기**를 실행한다.
> - `RunnableLambda`는 일반 함수를 Runnable로 감싸므로, 복잡한 라우팅 로직을 Python 코드로 직접 작성할 수 있다.
> - 특히 `RunnableLambda`가 **또 다른 Runnable을 반환하면**, 그 Runnable이 이어서 실제 실행된다.

> 💡 **언제 무엇을 쓰나?**
>
> - 조건이 단순하면 `RunnableBranch`
> - 조건 계산, 외부 룩업, 동적 선택이 복잡하면 `RunnableLambda`

🔗 [RunnableBranch 공식 문서](https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.branch.RunnableBranch.html)
🔗 [RunnableLambda 공식 문서](https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.RunnableLambda.html)

---

## STAGE 6. RAG 스타일 멀티 체인

### 개념

Retrieval-Augmented Generation(RAG)의 기본 구조를 LCEL로 구현한 패턴이다.
질문이 들어오면 컨텍스트를 외부에서 가져오고, 가져온 컨텍스트와 질문을 병렬로 프롬프트에 주입한다.

### 예시 코드

```python
# 가상의 검색 함수
def retrieve_context(query: str) -> str:
    return f"검색된 컨텍스트: {query}에 대한 정보..."

# RAG 파이프라인
rag_chain = (
    RunnableParallel(
        question=RunnablePassthrough(),
        context=RunnableLambda(lambda x: retrieve_context(x["question"]))
    )
    | ChatPromptTemplate.from_template("""컨텍스트를 참고하여 질문에 답변하세요.
        컨텍스트: {context}
        질문: {question}""")
    | llm
    | StrOutputParser()
)

# 입력은 질문 단일 문자열 {"question": "..."} 형태
result = rag_chain.invoke({"question" : "LangChain 이란?"})
```

> 🔍 **내부 동작 원리**:
>
> ```text
> {"question": "..."} 입력됨
>   ↓ (RunnableParallel 분기)
>   ├─ question=RunnablePassthrough() → 원본 question 유지
>   └─ context=RunnableLambda(...) → retrieve_context(...) 호출값 유지
>   ↓ (딕셔너리로 결합)
> {"question": "...", "context": "검색된 컨텍스트..."}
>   ↓ (프롬프트 주입)
> 프롬프트 완성 후 LLM 통과
> ```

> 💡 **이 패턴을 LCEL 관점에서 보면**
>
> 1. 입력 상태를 유지할 필드와 새로 계산할 필드를 병렬로 만든다.
> 2. 병합된 `dict`를 프롬프트에 그대로 넣는다.
> 3. 이후 단계는 이 `dict` 전체를 하나의 상태처럼 다룬다.

🔗 [RunnableParallel 공식 문서](https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.RunnableParallel.html)
🔗 [RunnablePassthrough 공식 문서](https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.passthrough.RunnablePassthrough.html)
🔗 [RunnableLambda 공식 문서](https://api.python.langchain.com/en/latest/core/runnables/langchain_core.runnables.base.RunnableLambda.html)

---

## 핵심 요약

```text
✅ 3대 요소 연결: ChatPromptTemplate | ChatOpenAI | StrOutputParser
✅ `|` 연산자는 내부적으로 `RunnableSequence`를 만든다
✅ dict 리터럴은 문맥에 따라 `RunnableParallel`로 해석된다
✅ 여러 단계 데이터 할당 유지: RunnablePassthrough.assign
✅ 동시 평가 진행: RunnableParallel
✅ 조건 라우팅 처리: RunnableBranch 또는 RunnableLambda 통한 분기
✅ RAG 패턴: 데이터를 검색(RunnableLambda)해 입력 질문과 합치는 병렬(RunnableParallel) 데이터 조합
```

---

## 실습 체크리스트

- [ ] LCEL 3단 체인(`prompt | llm | output_parser`) 직접 작성하기
- [ ] `RunnablePassthrough.assign` 으로 두 단계 프로세스(예: 번역 후 요약) 생성하고 이전 변수 유지 확인하기
- [ ] `RunnableParallel` 로 3개 이상의 프롬프트 병렬 처리 결과(`results["키"]`) 접근 확인하기
- [ ] `RunnableBranch` 에서 조건(if/elif/else) 순서 변경해보고 기본 조건이 타는지 테스트하기
- [ ] RAG 체인에서 커스텀 `retrieve_context` 함수에 간단한 조건 로직(예: 하드코딩된 사전) 넣어서 답변 확인하기
