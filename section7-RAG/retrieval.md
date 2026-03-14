# RAG Retrieval Pipeline 강의 노트

> **강의 주제**: RAG 파이프라인의 검색(Retrieval) 단계 구현 및 Agent 연동
> **난이도**: 중급
> **현재 코드 기준 파일**: `section7-RAG/backend/core.py`
> **사용 패키지**: `langchain`, `langchain-pinecone`, `langchain-openai`

---

## 1. 다중 모델 지원을 위한 `init_chat_model`

> **"왜 모델 클래스(예: ChatOpenAI)를 직접 쓰지 않고 `init_chat_model`을 쓸까요?" (Why)**
> LLM 애플리케이션을 개발하다 보면 비용, 성능, API 한도 등의 이유로 OpenAI에서 Anthropic(Claude), Google(Gemini) 등 타 벤더 모델로 교체해야 할 일이 잦습니다. `init_chat_model`은 문자열 인자 하나만으로 다양한 프로바이더의 모델을 똑같은 인터페이스로 팩토리(Factory) 패턴처럼 리턴해주어, 코드 수정 없이 벤더 확장을 가능하게 해줍니다.

```python
from langchain.chat_models import init_chat_model

# OpenAI의 gpt-5.2 모델 초기화 (벤더 종속성을 낮춤)
model = init_chat_model("gpt-5.2", model_provider="openai")
```

> 💡 **실무 포인트**:
> 실무에서는 "openai", "anthropic" 등의 하드코딩된 문자열 대신 `.env` 설정이나 데이터베이스 설정 값을 읽어와 동적으로 모델을 갈아끼울 수 있게 라우팅 아키텍처를 잡습니다. 특정 모델의 API 장애가 발생했을 때 즉각 폴백(Fallback) 모델로 전환하는 로직을 짤 때 핵심이 되는 함수입니다.

🔗 [init_chat_model (Model Factory) 공식 문서](https://python.langchain.com/docs/how_to/chat_models_universal_init/)

---

## 2. LLM용 컨텍스트 정보와 앱용 메타데이터 분리 (`content_and_artifact`)

> **"검색 툴(Tool)이 텍스트만 리턴하면 되지, 왜 굳이 'Artifact'라는 복잡한 개념을 쓸까요?" (Why)**
> RAG 검색 결과에는 '모델이 읽어볼 본문(Content)'도 있지만, 사용자 UI에 보여줄 장황한 메타데이터(문서 URL, 작성자 등 Raw Document)도 존재합니다. 이 두 정보를 한꺼번에 모델(LLM)에 던져버리면 프롬프트가 오염되고 토큰 비용이 낭비됩니다. LLM에게는 요약된 본문(Content)만 주고, 백엔드/프론트엔드 앱에는 원본 객체(Artifact)를 우회해서 통과시키기 위해 존재합니다.

```python
from langchain.tools import tool

# response_format을 "content_and_artifact"로 지정하면 튜플을 리턴해야 합니다.
@tool(response_format="content_and_artifact")
def retrieve_context(query: str):
    """Retrieve relevant documentation to help answer user queries about LangChain."""
    # 1) Vector Store에서 나와 가장 비슷한 Top 4 문서 검색
    retrieved_docs = vectorstore.as_retriever().invoke(query, k=4)

    # 2) LLM에게 보내줄 가벼운 텍스트(문자열) 직렬화 (Content)
    serialized = "\n\n".join(
        f"Source: {doc.metadata.get('source', 'Unknown')}\n\nContent: {doc.page_content}"
        for doc in retrieved_docs
    )

    # 3) (Content 문자열, Artifact 객체) 순으로 반환
    return serialized, retrieved_docs
```

> 💡 **실무 포인트**:
> LLM 비용 최적화(Token Optimization)의 핵심 중 하나입니다. "UI에 버튼으로 그려야 할 복잡한 메타데이터"를 무심코 시리얼라이즈해서 LLM 인풋 프롬프트로 밀어넣는 주니어 엔지니어링 실수가 가장 흔하게 발생합니다. LLM용 경로와 UI(App)용 우회 경로를 완벽히 분리하세요.

> 🔍 **내부 동작 원리 (개발자 관점 예시)**:
>
> 외부 API(예: 날씨 API)를 호출하는 상황을 가정해 보겠습니다. 이 API는 기온뿐 아니라 습도, 풍속 등 수백 개의 메타데이터가 담긴 엄청나게 무거운 JSON 딕셔너리를 반환합니다.
>
> ```python
> # ❌ 1. Content만 넘기는 과거의 방식 
> @tool
> def get_weather(city: str):
>     raw_api_response = {"temperature": 25, "humidity": 60, "wind_speed": 5, ...}
>     content_string = f"{city} 온도는 {raw_api_response['temperature']}도 입니다."
>     return content_string 
> ```
>
> 위 방식에서는 LLM이 텍스트(온도)만 보고 대답할 수 있어서 토큰은 절약되지만, 프론트엔드 개발자가 화면에 '습도'와 '풍속'을 그려야 하니 데이터를 달라고 해도 줄 방법이 없습니다. 툴이 텍스트만 만들고 무거운 원본 객체(`raw_api_response`)를 버렸기 때문입니다.
>
> ```python
> # ✅ 2. content_and_artifact 로 분리한 방식
> @tool(response_format="content_and_artifact")
> def get_weather_advanced(city: str):
>     raw_api_response = {"temperature": 25, "humidity": 60, "wind_speed": 5, ...}
>     
>     # ✨ Content: LLM의 프롬프트 모델로 들어갈 짧은 문자열 
>     content = f"{city} 온도는 {raw_api_response['temperature']}도 입니다."
>     
>     # 📦 Artifact: LLM한테는 안 주지만, 백엔드 서버 메모리에 고이 저장해둘 원본 객체
>     artifact = raw_api_response
>     
>     return content, artifact
> ```
>
> 이렇게 우회로를 뚫어놓으면, LLM은 텍스트(Content)만 읽어 빠르고 정확하게 정답을 만들어내고, 백엔드 개발자는 대화 기록안에 짱박혀 있던 원본 객체(Artifact)를 꺼내 프론트엔드 UI 화면용 데이터로 내려줄 수 있게 됩니다. RAG에서의 Document 객체도 정확히 이 메커니즘을 타고 UI 렌더링용으로 넘어갑니다.

🔗 [Tool Artifacts 및 툴 응답 형식 관리 가이드](https://python.langchain.com/docs/how_to/tool_artifacts/)

---

## 3. 검색 Tool을 장착한 에이전트(Agent) 생성

> **"그냥 프롬프트에 검색 텍스트를 바로 때려넣지 않고, 왜 굳이 'Agent'한테 검색 툴을 쥐어줄까요?" (Why)**
> 정적인 프롬프트 체인은 "반드시 1번 검색하고, 1번 답해라"는 고정된 워크플로우만 가능합니다. 하지만 Agent에게 검색 도구를 넘겨주면, 스스로 질문을 분석한 후 "이 질문은 검색이 필요하겠군", "검색 결과가 이상한데 다른 키워드로 한 번 더 검색해볼까?"처럼 코드를 직접 짜지 않아도 알아서 예외 상황과 뎁스를 해결(Reasoning)할 수 있기 때문입니다.

```python
from langchain.agents import create_agent

system_prompt = (
    "You are a helpful AI assistant that answers questions about LangChain documentation. "
    "Use the tool to find relevant information before answering questions. "
    "Always cite the sources you use in your answers. "
    "If you cannot find the answer in the retrieved documentation, say so."
)

# Retrieval 툴(retrieve_context)을 에이전트에게 권한 부여
agent = create_agent(model, tools=[retrieve_context], system_prompt=system_prompt)

# User 메세지로 Agent 실행
messages = [{"role": "user", "content": "what are deep agents?"}]
response = agent.invoke({"messages": messages})
```

> 💡 **실무 포인트**:
> `system_prompt` 마지막 줄의 **"If you cannot find the answer in the retrieved documentation, say so."**는 단순한 문장이 아니라, 가장 강력한 환각(Hallucination) 억제 장치입니다. Agent는 기본적으로 '어떻게 해서든 답변을 지어내려는 본능'이 있으므로 명확한 Fallback 행동 강령을 시스템 프롬프트 레벨에서 제어해야 합니다. (이것을 Anti-Hallucination Prompting 이라고 부릅니다.)

🔗 [Agents 및 ReAct 아키텍처 공식 문서](https://python.langchain.com/docs/concepts/agents/)

---

## 4. 메세지 히스토리에서 Artifact(원본 출처) 추출하기

> **"검색 결과(Artifact)를 LLM한테는 숨겼는데, 우리 백엔드 서버(앱)는 그걸 어떻게 꺼내서 사용자 UI에 던져주나요?" (Why)**
> 앞서 `content_and_artifact` 방식으로 리턴한 Artifact는, 전체 대화 내역(`response["messages"]`) 중 모델이 툴을 호출했던 흔적인 `ToolMessage` 객체 안에 조용히 보관되어 있습니다. 백엔드 로직은 대화 배열을 순회하며 이 `ToolMessage`를 찾아 스니핑(Sniffing)해 와야 합니다. 그래야 "AI의 답변"과 "답변의 근거가 된 출처 문헌"을 묶어서 서비스할 수 있습니다.

```python
from langchain.messages import ToolMessage

answer = response["messages"][-1].content
context_docs = []

# 대화 히스토리 전체를 순회하며 ToolMessage 안에 숨겨진 Artifact 빼내기
for message in response["messages"]:
    if isinstance(message, ToolMessage) and hasattr(message, "artifact"):
        if isinstance(message.artifact, list):
            context_docs.extend(message.artifact)

return {"answer": answer, "context": context_docs}
```

> 💡 **실무 포인트**:
> 이 구조를 이해하는 것이 RAG UX의 핵심입니다. 실무 서비스에서는 사용자에게 단순히 "답변 텍스트"만 보여주면 신뢰도가 매우 떨어집니다. 추출해 낸 `context_docs`를 활용해 프론트엔드에서 "[1] 출처 링크", "참고 문헌 보기" 등의 UI를 그려주어야 완벽한 Agentic UX가 완성됩니다.

---

## 핵심 요약

```text
✅ [모델 확성] init_chat_model을 사용하면 하드코딩 없이 OpenAI ↔ Anthropic 간 자유로운 모델 로테이션이 가능하다.
✅ [토큰 최적화] @tool(response_format="content_and_artifact") 기능을 활용하면, LLM용 텍스트와 앱 구동용 메타데이터를 완벽히 분리해 불필요한 토큰 낭비를 막을 수 있다.
✅ [Agent 활용] 검색 기능(Retrieval)을 프롬프트 주입 방식이 아닌 'Tool'로 분리해서 Agent에게 쥐어주면, 검색 여부와 횟수를 자체적으로 판단하여 더 유연하고 지능적인 RAG가 구축된다.
✅ [환각 통제] Agent System Prompt에 "검색 결과에 없으면 모른다고 답해라"는 규칙을 명시하여 할루시네이션(Hallucination) 빈도를 획기적으로 낮춘다.
✅ [출처 노출] 모델 구동 후 ToolMessage 객체 내부에 격리된 artifact를 순회 추출함으로써, 사용자 UI에 답변의 신뢰성(Grounding/Source)을 시각적으로 제공할 수 있다.
```

---

## 5. LangSmith 추적(Tracing)과 `as_retriever()`의 중요성

> **"벡터 스토어 객체를 그대로 검색 툴로 쓰지 않고, 왜 굳이 `.as_retriever()`로 변환해서 써야 할까요?" (Why)**
> LangChain 공식 문서의 일부 예제에서는 벡터 스토어의 기본 함수(`vectorstore.similarity_search`)를 그대로 툴로 사용하는 경우가 있습니다. 하지만 실무에서는 반드시 `as_retriever()`로 변환하여 사용해야 합니다. 왜냐하면 LangSmith(모니터링 툴)에서 추적(Tracing)을 할 때, 단순 벡터 스토어 함수는 로깅이 예쁘게 남지 않지만, Retriever 객체를 통과하면 검색된 문서, 점수, 소요 시간 등이 완벽하게 구조화되어 대시보드에 렌더링되기 때문입니다.

```python
# ❌ [비추천] 벡터 스토어를 직접 검색 도구로 사용 (LangSmith 로깅이 불친절함)
# retrieved_docs = vectorstore.similarity_search(query)

# ✅ [추천] Retriever 인터페이스로 변환하여 사용 (LangSmith 추적 최적화)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
retrieved_docs = retriever.invoke(query)
```

> 💡 **실무 포인트**:
> RAG 파이프라인 디버깅의 80%는 **"왜 이 문서가 검색되었는가?"** 혹은 **"왜 정답이 있는 문서가 검색되지 않았는가?"**를 찾는 과정입니다. `as_retriever()`를 사용해 LangSmith에 검색 과정(Trace)을 깔끔하게 남겨두면, LLM이 'Deep Agents'를 검색할 때 "What are deep agents?"라는 사용자의 질문을 "LangChain's deep agents definition"이라는 더 나은 검색어로 자체적으로 재작성(Query Transformation)해서 던진 정황까지 완벽하게 시각화하여 확인할 수 있습니다.

> 🔍 **내부 동작 원리 (LangSmith Trace 시각화)**:
>
> ```text
> [Human Input] "What are deep agents?"
>      ↓
> [LLM (Agent)] "검색이 필요하겠군. 검색 키워드를 'LangChain's deep agents definition'으로 재작성해서 Tool 호출!"
>      ↓
> [Tool: retrieve_context] (as_retriever 사용 시 로깅됨)
>      ├── 📄 Doc 1: "...deep agents are..." (Source: blog.langchain)
>      └── 📄 Doc 2: "...complex open-ended tasks..." 
>      ↓
> [LLM (Agent)] 검색된 내용을 바탕으로 최종 답변 생성: "Deep agent is a term LangChain coined..."
> ```

🔗 [Vector Store 연동 및 Retriever 공식 문서](https://python.langchain.com/docs/concepts/retrieval/)

---

## 실습 체크리스트

- [ ] `.env`에 `OPENAI_API_KEY`, `PINECONE_API_KEY`를 설정하고, `ingestion.py`에서 생성했던 Pinecone 인덱스 이름과 본 코드의 인덱스 이름(`langchain-docs-2026`)이 동일한지 동기화 확인
- [ ] `init_chat_model`의 문자열을 변경해가며(예: `gpt-4o-mini`, `gpt-3.5-turbo`) 모델 팩토리 패턴이 동적으로 잘 작동하는지 테스트
- [ ] `retrieve_context` 함수 내부의 튜플 리턴 형태(`return serialized, retrieved_docs`)와 데코레이터 선언부를 분석하고, `response_format`을 의도적으로 "content" 로만 변경 시 발생하는 파서 에러 재현해 보기
- [ ] 최종 출력되는 로컬 변수 `result["context"]` 배열 안에 LangChain Document 객체(및 Source 메타데이터)가 훼손 없이 들어있는지 디버거를 통해 점검
- [ ] 질의문(query)을 'LangChain'과 아예 관련 없는 내용(예: "어제 날씨 어땠어?")으로 변경 시, Agent가 검색 툴 사용을 스킵하거나 "문서에 없다"고 대답하는 행동 양식 확인
- [ ] LangSmith 대시보드를 열어 Trace 기록을 확인하고, Agent가 내 질문을 어떻게 검색용 Query로 재작성했는지, `as_retriever()`를 통해 검색된 청크들이 어떻게 기록되는지 눈으로 확인해 보기
