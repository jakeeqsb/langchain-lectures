# Section 6 — LCEL 딥다이브: RunnablePassthrough와 데이터 흐름의 마법

> **강의 주제**: LCEL 체인 내부에서 데이터 딕셔너리가 어떻게 할당되고 전달되는지(`RunnablePassthrough.assign`, `itemgetter`), 그리고 일반 함수가 어떻게 파이프라인에 통합되는지(`RunnableLambda`)에 대한 심층 분석
> **난이도**: 고급
> **사용 패키지**: `langchain-core`

---

## 1. RunnablePassthrough.assign() — 파이프라인의 데이터 컨테이너

> **"Prompt Template은 `question`과 `context` 두 개의 변수를 요구하는데, 체인의 첫 시작점에는 `question`밖에 없습니다. 빈 `context`는 대체 어디서, 어떻게 채워 넣어야 할까요?" (Why)**
> LCEL 파이프(`|`)는 데이터를 순차적으로 다음 컴포넌트로 넘깁니다. 만약 앞에서 문자열 하나만 덜렁 넘기면 프롬프트 템플릿이 필요한 2개의 인자 중 하나를 잃어버리게 됩니다. `RunnablePassthrough.assign()`은 **"기존 입력 데이터(딕셔너리)는 그대로 통과(Passthrough)시키면서, 내가 지정한 새로운 키(Key)와 실행 결과(Value)만 딕셔너리에 추가(Assign)해주는"** LCEL 데이터 흐름의 핵심 마법사입니다.

`RunnablePassthrough`는 그 자체로는 데이터를 전혀 변경하지 않는 'Identity Function(항등 함수)'처럼 동작합니다. 하지만 `.assign()` 메서드와 결합하면 파이프라인 중간에 필요한 변수들을 동적으로 계산해서 덧붙일 수 있습니다.

```python
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

# 초기 입력 데이터: {"question": "What is Pinecone?"}

retrieval_chain = (
    # 입력된 딕셔너리는 그대로 유지한 채, 'context'라는 키를 새로 계산해서 덧붙임
    RunnablePassthrough.assign(
        context=itemgetter("question") | retriever | format_docs
    )
    | prompt_template
    # ...
)
```

> 💡 **실무 포인트**:
> **왜 람다(`lambda x: x["question"]`) 대신 `itemgetter`를 쓸까요?**
> 실무에서는 파이썬 기본 유틸리티 모듈의 `operator.itemgetter`를 사용하는 것이 관례입니다. `lambda`보다 실행 속도가 C 레벨로 최적화되어 있어 미세하게 빠르고, 가독성이 좋아 LCEL의 선언형 파이프라인 문법(`|`)과 훨씬 시각적으로 잘 어울립니다.
>
> - 나쁜 예: `lambda x: x["question"] | retriever` (가독성 저하)
> - 좋은 예: `itemgetter("question") | retriever`

> 🔍 **내부 동작 원리**: (`RunnablePassthrough.assign`의 데이터 병합 과정)
>
> ```text
> 1. 초기 invoke() 호출
>    입력: {"question": "What is Pinecone?"}
> 
> 2. RunnablePassthrough.assign() 진입
>    [그대로 유지] ──▶ {"question": "What is Pinecone?"} (A)
>    [새로 계산] ────▶ context 서브 체인 실행:
>                     1) itemgetter("question")이 "What is Pinecone?" 추출
>                     2) retriever가 문서 검색 ➔ [Doc1, Doc2, Doc3]
>                     3) format_docs가 텍스트 병합 ➔ "Pinecone is a DB..."
>                     결과물 (B)
> 
> 3. 병합 (Assign) 완료 후 다음 파이프로 전달
>    출력(A+B): {
>        "question": "What is Pinecone?", 
>        "context": "Pinecone is a DB..."
>    }
> ```
>
> 이렇게 완성된 딕셔너리가 다음 파이프라인인 `prompt_template`의 `{question}`과 `{context}` 플레이스홀더에 정확히 매핑됩니다.

🔗 [RunnablePassthrough 공식 문서](https://reference.langchain.com/python/langchain-core/runnables/passthrough/RunnablePassthrough)

---

## 2. RunnableLambda — 일반 함수를 파이프라인으로

> **"우리가 직접 만든 `format_docs` 함수는 LangChain의 `Runnable` 클래스를 상속받지도 않았는데, 어떻게 `| format_docs` 처럼 파이프 문법이 에러 없이 작동할까요?" (Why)**
> LangChain은 개발자의 편의를 위해 "Syntactic Sugar(문법적 설탕)"를 제공합니다. 순수 파이썬 함수를 파이프라인(`|`) 배열 안에 넣기만 하면, LangChain이 내부적으로 이를 감지하고 자동으로 `RunnableLambda`라는 객체로 감싸줍니다. 덕분에 우리는 복잡한 클래스 디자인 없이도 커스텀 로직을 파이프라인에 자유롭게 끼워 넣을 수 있습니다.

위 코드에서 사용한 `format_docs` 함수는 단순한 파이썬 함수입니다. `.invoke()` 메서드 같은 건 당연히 없습니다.

```python
# 순수 파이썬 함수
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 파이프라인 내부에 작성하면:
# context=itemgetter("question") | retriever | format_docs

# LangChain이 내부적으로 이렇게 변환합니다:
# context=itemgetter("question") | retriever | RunnableLambda(format_docs)
```

> 💡 **실무 포인트**:
> **LangSmith를 통한 체인 모니터링 시 나타나는 `RunnableLambda`**
> 프로덕션 환경에서 LangSmith로 Tree Trace를 열어보면, 우리가 작성하지 않은 `RunnableLambda` 또는 `RunnableSequence`라는 노드들이 보일 것입니다. 이것이 바로 우리가 쓴 일반 함수(`format_docs`) 나 `itemgetter`가 실행될 때 LangChain 엔진이 자동으로 변환하여 로깅한 흔적입니다. 당황하지 말고 "아, 내 함수가 익명 Runnable로 래핑되어 잘 돌고 있구나"라고 이해하시면 됩니다.

🔗 [Runnable / RunnableLambda 관련 공식 레퍼런스](https://reference.langchain.com/v0.3/python/core/runnables/langchain_core.runnables.base.Runnable.html)

---

## 핵심 요약

```text
✅ RunnablePassthrough: 입력을 변경하지 않고 그대로 넘겨주는(Bypass) 역할을 합니다.
✅ .assign(): 입력을 유지한 상태에서, 특정 키값만 서브 체인(Sub-chain)을 돌려 계산한 뒤 딕셔너리에 추가해줍니다. (Prompt Template 변수 주입의 일등공신)
✅ itemgetter: 딕셔너리에서 특정 키를 추출할 때 람다(lambda) 대신 사용하여 가독성과 파이프라인 결합성을 높입니다.
✅ RunnableLambda: 순수 파이썬 함수를 파이프라인에 넣으면 LangChain이 알아서 백그라운드에서 Runnable 객체로 래핑해 스트리밍, 배치 등의 기능을 지원하게 만듭니다.
```

---

## 실습 체크리스트

- [ ] `RunnablePassthrough` 없이 딕셔너리를 직접 조작하려고 시도해보고, 프롬프트 템플릿 단계에서 어떤 `KeyError`가 발생하는지 확인해보기
- [ ] LangSmith 대시보드에서 해당 RAG 체인의 동작 Trace를 열어보고, 내부적으로 쪼개진 3개의 스텝(`itemgetter`, `retriever`, `format_docs(RunnableLambda)`)이 어떻게 병렬/순차적으로 동작하는지 눈으로 확인해보기
