# RAG 아키텍처 비교 강의 노트

> **강의 주제**: Three ways to do RAG (Two-step, Agentic, Hybrid)
> **난이도**: 중급 / 고급
> **사용 패키지**: `langchain`, `langgraph`

---

## 1. 2단계 RAG (Two-Step RAG / LCEL 기반)

> **"가장 빠르고 통제 가능한 RAG를 만들고 싶다면?" (Why)**
> 초기 RAG 시스템의 표준입니다. "무조건 검색을 먼저 하고, 그 결과를 바탕으로 생성한다"는 고정된(Fixed) 파이프라인(LCEL)을 가집니다. LLM이 중간에 개입하여 "검색을 할까 말까?"를 고민하지 않기 때문에 속도가 매우 빠르고 디버깅이 쉽습니다.

```python
# 가장 전형적인 Two-Step RAG 체인 (LCEL)
# 검색(Retrieval) -> 포맷팅 -> 프롬프트 -> LLM 생성 순서로 무조건 흘러갑니다.
chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
```

> 💡 **실무 포인트**: 
> 사내 규정 챗봇, 단순 매뉴얼 검색기 등 "질문의 의도가 명확하고, 백무조건 검색이 필요한 도메인"에서 여전히 현역으로 가장 널리 쓰이는 아키텍처입니다. 속도(Latency)가 가장 빠르다는 압도적 장점이 있습니다.

> 🔍 **내부 동작 원리**:
>
> ```text
> [User Input] 
>      ↓ (1. Retrieval 단계가 무조건 먼저 실행됨)
> [Vector Store] → [Retrieved Documents]
>      ↓ (2. Generation 단계)
> [LLM Prompt에 Context로 주입]
>      ↓
> [Final Answer]
> ```

🔗 [LCEL 기반 기본 RAG 구현 가이드](https://python.langchain.com/docs/tutorials/rag/)

---

## 2. 에이전틱 RAG (Agentic RAG / RAG Agent)

> **"검색 여부와 방식을 LLM에게 전적으로 위임한다면?" (Why)**
> Two-Step RAG는 "안녕?" 이라는 일상적인 인사말에도 무조건 벡터 DB를 검색해버리는 비효율/에러를 낳을 수 있습니다. Agentic RAG는 ReAct 에이전트에게 '검색 툴(Tool)'을 쥐어주고, LLM이 사용자 질문을 보고 직접 "이건 검색이 필요하겠군", "어떤 키워드로 검색할까?"를 스스로 판단하게 만듭니다.

```python
from langchain.agents import create_agent

# LLM에게 검색 도구(retriever_tool)를 주고 알아서 판단하게 만듦
agent = create_agent(llm, tools=[retriever_tool], system_prompt="...")
```

> 💡 **실무 포인트**: 
> 유연성(Flexibility)은 최고지만, 상용 서비스(Production)에서는 추천하지 않는 아키텍처입니다. LLM에게 너무 많은 자유도(Freedom)를 쥐어주면 무한 루프에 빠지거나, 엉뚱한 키워드로 계속 검색을 시도하는 등 통제 불능 상태가 생기고 API 토큰 비용과 지연 시간(Latency)이 기하급수적으로 폭증할 수 있습니다.

> 🔍 **내부 동작 원리**:
>
> ```text
> [User Input] 
>      ↓
> [LLM (Reasoning)] "검색이 필요한 질문인가? Yes" → "검색 키워드는 X로 해야지"
>      ↓
> [Tool Execution] (검색 결과 리턴)
>      ↓
> [LLM (Reasoning)] "검색 결과가 충분한가? No" → "다른 키워드 Y로 다시 검색 툴 호출!"
>      ↓
> [LLM] 최종 답변 생성
> ```

🔗 [Agentic RAG 개념 및 튜토리얼](https://python.langchain.com/docs/tutorials/qa_chat_history/#agents)

---

## 3. 하이브리드 RAG (Hybrid RAG / 커스텀 워크플로우)

> **"통제력과 유연성, 두 마리 토끼를 모두 잡는 엔터프라이즈급 RAG는?" (Why)**
> 현존하는 엔터프라이즈 프로덕션 환경에서 **가장 많이 채택되고 추천하는 방식(Best Practice)**입니다. LCEL의 답답함과 Agentic RAG의 통제 불능이라는 단점을 보완하기 위해, `LangGraph` 등을 이용해 **검색 전/후의 파이프라인을 사람이 직접 제어**하는 중간 단계를 삽입합니다.

하이브리드 RAG는 보통 다음과 같은 중간 단계(Intermediate Steps)를 파이프라인에 강제합니다:
1. **Query Pre-processing (질문 재작성/라우팅)**: 검색하기 좋게 질문을 다듬거나, 의상 질문은 DB A로, 정책 질문은 DB B로 라우팅.
2. **Retrieval Validation (검색 검증)**: 가져온 문서가 질문과 관련이 있는지 LLM이 평가(Grade).
3. **Post-generation Validation (할루시네이션 검증)**: 최종 답변이 원본 문서에 기반했는지 마지막으로 확인.

```python
# LangGraph를 이용해 각 단계를 엣지(Edge)로 명시적으로 이어붙여 통제력을 확보
# (이후 구체적 구현은 LangGraph 챕터에서 진행)
graph.add_edge("query_rewrite", "retrieve")
graph.add_conditional_edges("retrieve", grade_documents)
graph.add_edge("generate", "check_hallucination")
```

> 💡 **실무 포인트**: 
> "LLM에게 툴을 주고 알아서 하라고 기도하는 것(Agentic RAG)"에서 벗어나, "이 조건일 땐 반드시 이 노드로 빠져서 확인을 거쳐라"라고 명확히 지시선을 그려주는 아키텍처(Flow Engineering)입니다. Self-RAG나 CRAG 같은 최신 RAG 논문들이 모두 이 아키텍처를 기반으로 설계되었습니다.

> 🔍 **내부 동작 원리 (Flow Engineering)**:
>
> ```text
> [User Input]
>      ↓
> (Node 1: Query Router) 질문 분류 및 재작성
>      ↓
> (Node 2: Retriever) Vector DB 검색
>      ↓
> (Node 3: Grader) 검색된 문서의 관련성 평가
>    ├── [관련성 낮음] → (Node 4: 웹 검색(Tavily)으로 Fallback)
>    └── [관련성 높음] → (Node 5: Generation) 최종 답변 생성
>                            ↓
>                    (Node 6: 응답 검증) Hallucination Check
> ```

🔗 [LangGraph 기반 Advanced RAG 아키텍처](https://langchain-ai.github.io/langgraph/tutorials/rag/langgraph_adaptive_rag/)

---

## 핵심 요약

```text
✅ [Two-Step] 검색 후 생성이 고정되어 있어 가장 빠르고 예측 가능하지만 유연성이 떨어진다. 단순 QA에 좋음.
✅ [Agentic] LLM이 검색 시점과 키워드를 자체 판단하므로 유연성이 좋으나, 통제 불능 늪에 빠질 위험이 있어 상용화에 부적합.
✅ [Hybrid] LangGraph 등을 이용해 질문 재작성, 문서 검증, 응답 검증 노드를 사람이 명시적으로 제어하는 방식으로 기업 실무에서 가장 선호됨.
```

---

## 실습 체크리스트

- [ ] LCEL 코드에서 두 가지 체인(`retriever | format_docs` 병렬화 등)을 짜보며 데이터 흐름 짚어보기
- [ ] Agentic RAG를 사용하다 무한 검색 에러 루프 거쳐보기 (통제가 필요한 이유 피부로 느끼기)
- [ ] LangGraph 기본 튜토리얼을 읽어보며, 조건부 엣지(Conditional Edge)가 RAG에서 어떻게 검증망(Guardrails) 역할을 하는지 숙지하기
