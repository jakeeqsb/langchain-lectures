# Section 6 — Advanced RAG 학습 로드맵

> **강의 주제**: 현재 Section 6의 기초 RAG 내용을 기준으로, 다음에 학습해야 할 고급 RAG 주제를 체계적으로 정리하기
> **난이도**: 중급 → 고급
> **목표**: "기초 RAG를 이해한 상태"에서 "실무형 RAG 시스템을 설계하고 개선하는 단계"로 넘어가기
> **기준 범위**: [6-1.rag_intro.md](./6-1.rag_intro.md), [6-2.basic_structure.md](./6-2.basic_structure.md), [6-4.ingestion.md](./6-4.ingestion.md), [6-5.retrieve.md](./6-5.retrieve.md)

---

## 1. 지금 어디까지 왔는가?

> **"현재 Section 6에서 이미 배운 범위는 어디까지일까요?" (Why)**
> 다음 단계를 제대로 잡으려면, 지금 배운 것이 "RAG 전체"가 아니라 "RAG의 기초 파이프라인"이라는 점을 먼저 분명히 알아야 합니다. 그래야 왜 다음 학습이 검색 품질, 체인 구조, 운영성으로 넘어가야 하는지 납득할 수 있습니다.

현재 노트 기준으로는 아래까지 학습한 상태다.

- RAG가 왜 필요한가
- Prompt Stuffing의 한계
- Document Loader / Text Splitter / Embeddings / Vector Store
- Ingestion 파이프라인
- 기본 Retriever
- Naive 2-step RAG
- LCEL 이전의 수동 Retrieval 체인

즉, 지금 단계는 **"RAG가 동작하는 최소 구조를 이해한 상태"**다.  
다음 단계는 **"검색 품질을 올리고, 체인을 더 유연하게 만들고, 운영 가능한 구조로 바꾸는 단계"**다.

> 💡 **실무 포인트**: 초급 RAG와 실무 RAG의 차이는 보통 모델 변경보다도 검색 품질, 청킹 전략, 평가 체계, 운영 파이프라인에서 크게 벌어진다.

---

## 2. 왜 이제 Advanced RAG로 넘어가야 할까?

> **"기본 RAG만으로도 답은 나오는데, 왜 더 복잡한 걸 배워야 하나요?" (Why)**
> 기초 RAG는 "돌아가는 데모"를 만들기엔 충분하지만, 실제 서비스에서는 문서가 더러워지고, 질문이 모호해지고, 검색 노이즈가 늘어나고, 비용과 지연시간 문제가 생깁니다. 결국 고급 RAG는 더 똑똑한 모델을 붙이는 게 아니라, **검색과 컨텍스트 구성 자체를 더 정교하게 설계하는 과정**입니다.

Advanced RAG의 핵심 축은 보통 3개다.

```text
1. Retrieval 품질 향상
2. 체인 / 제어 흐름 고도화
3. 평가 / 운영 / 관측성 강화
```

---

## 3. 다음에 배워야 할 Advanced 주제

### 3-1. LCEL 기반 RAG 체인

> **"왜 수동 파이프라인 다음에 LCEL을 배워야 할까요?" (Why)**
> 수동 체인은 내부 동작을 이해하기엔 좋지만, 조합성, 스트리밍, 비동기, 추적성 면에서 한계가 큽니다. LCEL은 RAG 파이프라인을 Runnable 단위로 선언적으로 조립하게 해주며, 이후 고급 패턴의 기반이 됩니다.

- `Runnable`, `RunnablePassthrough`, `RunnableLambda`
- `prompt | llm | parser` 체인 사고방식
- `create_retrieval_chain` 류의 고수준 조합
- streaming / async / trace 친화적 구조

> 💡 **실무 포인트**: LCEL은 "문법 예쁘게 쓰기"가 아니라, 체인을 재사용 가능하고 관측 가능하게 만드는 구조화 도구다.

---

### 3-2. 검색 품질 튜닝 (Chunking / k / overlap)

> **"왜 실무 RAG에서 청킹이 그렇게 중요할까요?" (Why)**
> RAG의 답변 품질은 결국 LLM이 아니라 Retriever가 무엇을 가져오느냐에 크게 좌우됩니다. 잘못 자른 청크는 아무리 좋은 모델을 써도 복구가 안 됩니다.

- `chunk_size`
- `chunk_overlap`
- 토큰 기반 splitter vs 문자 기반 splitter
- 문서 타입별 splitter 전략: Markdown, PDF, 코드 문서, FAQ / 짧은 KB 문서
- `k` 값 튜닝

> 💡 **실무 포인트**: 많은 팀이 모델만 바꾸다가 성능 개선이 안 된다고 느끼는데, 실제로는 chunking 실험표 하나만 제대로 만들어도 훨씬 큰 개선을 얻는 경우가 많다.

---

### 3-3. Query Transformation 계열 Retriever

> **"질문을 그대로 검색하면 왜 한계가 생길까요?" (Why)**
> 사용자의 질문은 검색에 최적화된 문장이 아닐 수 있습니다. 짧고, 애매하고, 도메인 용어를 생략하거나, 너무 구어체일 수 있습니다. 그래서 고급 RAG에서는 질문 자체를 더 검색 친화적인 형태로 바꿔서 retrieval 품질을 끌어올립니다.

- `MultiQueryRetriever`
- query rewriting / paraphrasing
- `SelfQueryRetriever`
- metadata-aware query construction

> 💡 **실무 포인트**: retrieval quality가 낮을 때 무조건 reranker부터 붙이기보다, 질문을 더 잘 검색되게 바꾸는 쪽이 더 싸고 간단할 때가 많다.

---

### 3-4. 2단계 검색과 Reranking

> **"왜 한 번 검색해서 끝내지 않고 두 번 거를까요?" (Why)**
> 벡터 검색은 recall은 좋지만 precision이 부족할 수 있습니다. 즉, 비슷한 문서를 넓게 잘 가져오지만, 최종적으로 LLM에 넣기엔 노이즈가 많을 수 있습니다. 그래서 1차로 넓게 뽑고, 2차로 다시 줄이는 구조가 중요해집니다.

- 1차 retrieval: recall 확보
- 2차 reranking: precision 확보
- `ContextualCompressionRetriever`
- cross-encoder reranker 개념
- token budget 내 context 압축

> 💡 **실무 포인트**: "많이 가져오면 안전하다"는 착각이 흔하다. 실제로는 너무 많이 넣으면 노이즈가 늘고 답변 품질이 떨어진다. reranking은 이 노이즈를 줄이는 핵심 단계다.

---

### 3-5. Parent Document / Multi-vector Retrieval

> **"작은 청크는 검색엔 좋고, 큰 청크는 답변엔 좋은데 둘 중 뭘 택해야 할까요?" (Why)**
> 작은 청크는 검색 정확도가 좋지만 문맥이 짧고, 큰 청크는 문맥은 풍부하지만 검색 정밀도가 떨어집니다. Parent document retrieval은 이 두 요구를 동시에 만족시키기 위한 대표 패턴입니다.

- child chunk로 검색
- parent chunk / 원문 단락으로 복원
- multi-vector indexing 개념
- 검색 단위와 답변 단위를 분리하는 설계

> 💡 **실무 포인트**: 문서 QA, 기술 문서 검색, 긴 정책 문서 답변에서는 parent-child 구조가 체감 성능을 크게 올려준다.

---

### 3-6. Metadata Filtering / Structured Retrieval

> **"왜 의미 검색만으로는 부족할까요?" (Why)**
> 실무 문서는 항상 버전, 날짜, 소스, 제품군, 팀, 언어 같은 구조적 정보를 함께 갖고 있습니다. 이 메타데이터를 안 쓰면 검색 범위가 너무 넓어지고, 유사하지만 틀린 문서가 섞이게 됩니다.

- source / date / version / doc_type 기반 필터링
- 특정 제품 문서만 검색
- 특정 기간 이후 문서만 검색
- metadata-aware retrieval

> 💡 **실무 포인트**: 실무 RAG는 semantic search 단독이 아니라, semantic search + metadata filter의 조합으로 성능을 안정화하는 경우가 많다.

---

### 3-7. Agentic RAG / LangGraph

> **"모든 질문에 무조건 retrieval을 해야 할까요?" (Why)**
> 어떤 질문은 검색이 필요 없고, 어떤 질문은 한 번 검색으로 부족하며, 어떤 질문은 검색과 재질문이 반복되어야 합니다. Agentic RAG는 retrieval 자체를 동적 의사결정 대상으로 다룹니다.

- 2-step RAG vs agentic RAG
- 언제 검색할지 판단
- 여러 번 검색할지 판단
- 검색 실패 시 fallback
- LangGraph 기반 상태 분기

> 💡 **실무 포인트**: Agentic RAG는 강력하지만 복잡도도 같이 올라간다. 모든 문제에 바로 LangGraph를 붙이기보다, 먼저 2-step RAG 한계를 명확히 경험한 뒤 넘어가는 편이 좋다.

---

### 3-8. RAG 평가와 관측성

> **"성능이 좋아졌는지 어떻게 증명할까요?" (Why)**
> RAG는 체감 품질만으로 판단하면 착각하기 쉽습니다. Retrieval이 좋아졌는지, 답변이 더 grounded해졌는지, hallucination이 줄었는지를 평가 체계 없이 알기 어렵습니다.

- LangSmith trace
- retrieval relevance
- groundedness
- answer correctness
- 평가용 질문셋 / golden set

> 💡 **실무 포인트**: Advanced RAG의 핵심은 "새 기능 추가"보다 **측정 가능한 개선**이다. 평가 없는 고도화는 대부분 감으로 끝난다.

---

### 3-9. 운영용 Ingestion

> **"문서를 한 번 넣고 끝나는 게 아닌데, 운영에서는 뭐가 달라지나요?" (Why)**
> 데모에서는 문서 한 번 색인하면 끝이지만, 실무에서는 문서가 계속 바뀌고 중복도 생기고 실패도 발생합니다. 그래서 ingestion은 배치 작업이 아니라 운영 파이프라인 관점으로 봐야 합니다.

- 증분 인덱싱
- 중복 제거
- 문서 버전 관리
- 실패 재처리
- 재색인 전략
- embedding 모델 교체 시 migration

> 💡 **실무 포인트**: 검색 품질만큼 중요한 게 인덱스 청결도다. 오래된 문서와 최신 문서가 섞이면 retrieval 품질이 천천히 망가진다.

---

### 3-10. 답변 제약과 Citation

> **"정답만 잘 말하면 되는 거 아닌가요?" (Why)**
> 실제 서비스에선 답의 내용만큼이나 "왜 그렇게 답했는지"가 중요합니다. 특히 기업, 금융, 의료, 정책 문서에서는 근거 없는 답변을 막고 출처를 명시하는 설계가 필수적입니다.

- citation / source attribution
- structured output
- "모르면 모른다" 정책
- unsupported claim 차단
- 답변 포맷 제약

> 💡 **실무 포인트**: RAG의 목적은 단순 정확도 향상만이 아니라, **근거 기반 답변 시스템**을 만드는 데 있다.

---

## 4. Advanced 단계로 가려면 어떤 순서로 학습해야 할까?

> **"주제가 많아 보이는데, 어떤 순서가 제일 효율적일까요?" (Why)**
> Advanced RAG는 개념을 많이 아는 것보다, 기초 구조를 흔들지 않고 단계적으로 올리는 학습 순서가 중요합니다. 무작정 LangGraph부터 들어가면 오히려 핵심이 흐려집니다.

추천 순서는 아래와 같다.

1. 현재 있는 naive RAG를 **LCEL 버전으로 다시 구현**
2. 같은 데이터셋으로 **chunking 실험표 작성**
3. **기본 retriever -> query transformation -> rerank** 순서로 고도화
4. **parent document retrieval**로 문맥 복원 패턴 익히기
5. **2-step RAG vs agentic RAG** 비교
6. 마지막에 **LangSmith 평가셋** 붙이기

---

## 5. 가장 추천하는 학습 순서

```text
1. LCEL / create_retrieval_chain
2. chunking 실험 + metadata 설계
3. MultiQueryRetriever / SelfQueryRetriever
4. reranker / contextual compression
5. ParentDocumentRetriever
6. LangGraph agentic RAG
7. LangSmith evaluation
```

> 💡 **실무 포인트**: 하나만 고르라면, 다음 핵심 묶음은 **LCEL 기반 RAG + retriever 고도화 + LangSmith 평가**다. 이 세 개가 붙어야 실무형 RAG로 넘어갈 수 있다.

---

## 6. 권장 실습 과제

- [ ] 현재 naive RAG 코드를 `create_retrieval_chain` 기반으로 다시 구현하기
- [ ] `chunk_size`, `chunk_overlap`, `k` 값을 바꾼 실험표 만들기
- [ ] 같은 질문에 대해 `basic retriever`와 `multi-query` 결과 비교하기
- [ ] retrieval 결과를 1차/2차로 줄이는 reranking 실험 해보기
- [ ] metadata 필터를 넣었을 때와 안 넣었을 때 precision 차이 관찰하기
- [ ] parent document retrieval 구조를 도식화해보기
- [ ] LangSmith에 질문셋을 올리고 retrieval relevance를 평가해보기

---

## 7. 공식 문서 링크

- Retrieval 개요: <https://docs.langchain.com/oss/python/langchain/retrieval>
- RAG 튜토리얼: <https://docs.langchain.com/oss/python/langchain/rag>
- Agentic RAG with LangGraph: <https://docs.langchain.com/oss/python/langgraph/agentic-rag>
- Retriever 관련 공식 문서: <https://docs.langchain.com/oss/python/langchain/retrieval>
- Self-query retrieval 관련 문서: <https://docs.langchain.com/oss/python/langchain/retrieval>
- Parent document retrieval 관련 문서: <https://docs.langchain.com/oss/python/langchain/retrieval>
- RAG 체인 구성 관련 문서: <https://docs.langchain.com/oss/python/langchain/rag>
- RAG 평가: <https://docs.langchain.com/langsmith/evaluate-rag-tutorial>

---

## 핵심 요약

```text
✅ 지금 Section 6은 기초 RAG 파이프라인까지 학습한 상태다.
✅ 다음 단계는 모델 교체보다 retrieval 품질과 체인 구조 고도화가 핵심이다.
✅ Advanced RAG는 LCEL, retriever 고도화, reranking, metadata filtering, agentic flow, evaluation으로 확장된다.
✅ 가장 효율적인 순서는 LCEL -> chunking 실험 -> retriever 고도화 -> reranking -> parent retrieval -> agentic RAG -> evaluation 이다.
✅ 실무형 RAG로 가려면 반드시 LangSmith 같은 평가/관측 체계를 붙여야 한다.
```
