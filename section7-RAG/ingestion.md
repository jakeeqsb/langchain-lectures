# RAG Ingestion Pipeline 강의 노트 (통합본)

> **강의 주제**: LangChain 기반 RAG ingestion 단계의 전체 구성 (문서 크롤링부터 비동기 벡터 저장까지)
> **난이도**: 중급
> **현재 코드 기준 파일**: `section7-RAG/ingestion.py`
> **확인된 패키지 버전**:
> `langchain>=1.2.6`,
> `langchain-openai>=1.1.7`,
> `langchain-tavily>=0.2.17`,
> `langchain-pinecone>=0.2.13`,
> `langchain-chroma>=1.1.0`,
> `langsmith>=0.6.4`

---

## 1. 롱 컨텍스트 시대, RAG는 죽었는가?

> **"LLM 컨텍스트 윈도우가 100만~200만 토큰으로 커졌는데, 왜 아직도 RAG가 필요할까요?" (Why)**
> 책 한 권을 통째로 컨텍스트에 넣을 수 있게 되었지만, 매번 100만 토큰을 모델에 주입하는 것은 속도가 매우 느리고 비용이 기하급수적으로 발생합니다. 대형 컨텍스트 윈도우(Long Context Window) 모델이 등장하면서 RAG의 필요성에 의문을 품는 시각이 있지만, RAG는 여전히 다음과 같은 강력한 이점을 가집니다.

1. **비용 효율성 (Cost Efficiency)**: 필요한 부분만 모델에 전달하므로 훨씬 빠르고 저렴합니다.
2. **환각 감소 및 정확성 (Precision & Noise Reduction)**: 관련성 높은 청크만 필터링해서 제공하므로 근거 없는 답변(Hallucination)을 줄이고 모델이 중간 문서 정보를 놓치는 'Lost in the middle' 현상을 방지합니다.
3. **출처 추적 (Traceability)**: 사용자에게 어느 문서에서 답변을 발췌했는지 명확한 출처를 제공할 수 있어 규제가 심한 도메인이나 신뢰성이 중요한 서비스에서 필수적입니다.

> 💡 **실무 포인트**:
> RAG와 Long Context 모델은 경쟁 관계가 아닌 **상호 보완적인 관계**입니다. RAG로 1차 필터링을 한 뒤, 기존 모델로는 소화 불가능했던 수십 개의 다중 문서(Multi-document)를 추출하고 통째로 롱 컨텍스트 모델에 넣어 비교 분석을 시키는 등 결합된(Hybrid) 아키텍처가 실무에서 각광받고 있습니다.

> 🔍 **내부 동작 원리**:
>
> ```text
> [Long Context 단독]
> 수백 장의 문서 통째로 입력 → 병목 발생, 비용 급증, 일부 정보 유실 가능성
> 
> [RAG + Long Context]
> 수백 장의 문서 → Vector Search → Top-K 청크 / 관련 문서 집중 추출 → 모델 입력 → 빠르고 정확하며 비용 효율적
> ```

---

## 2. Ingestion 단계는 왜 따로 분리할까?

> **"왜 ingestion 단계가 필요할까요?" (Why)**
> RAG는 그냥 "문서를 넣고 검색한다"로 끝나지 않습니다. 원본 문서는 길고, 형식이 제각각이고, 검색 가능한 벡터로 바로 바뀌지도 않습니다. 그래서 실제 서비스에서는 먼저 문서를 수집하고, 정제하고, 잘게 나누고, 임베딩하고, 벡터 저장소에 넣는 ingestion 단계를 따로 만듭니다.

전체 Ingestion의 목표는 다음 순서입니다.
1. 문서 사이트 크롤링 (Tavily)
2. `Document` 객체로 변환
3. 텍스트 분할 (Chunking)
4. 임베딩(Vector 변환) 생성 및 벡터 저장소(Pinecone/Chroma) 인덱싱

> 🔍 **내부 동작 원리**:
>
> ```text
> 문서 사이트 URL
>   ↓
> 크롤러 (TavilyCrawl)
>   ↓
> 페이지별 원시 데이터 (raw_content)
>   ↓
> LangChain Document (page_content, metadata)
>   ↓
> Text Splitter (Recursive Character Splitter)
>   ↓
> Embedding Model (OpenAI API)
>   ↓
> Vector Store (Pinecone / Chroma)
> ```

> 💡 **실무 포인트**:
> Ingestion 작업을 사용자가 질문(Retrieval)할 때마다, 혹은 애플리케이션 시작 시마다 실행하면 비용과 시간이 폭증합니다. 보통은 배치 작업, 스케줄러(Airflow 등), 문서 변경 감지 이벤트 파이프라인으로 분리해서 관리하고, 서비스단(API)에서는 검색(Retrieval)만 수행하도록 구성합니다.

---

## 3. 환경 변수와 초기 설정

> **"왜 환경 설정을 먼저 잡아야 할까요?" (Why)**
> LLM 애플리케이션은 파이프라인 단계별로 외부 API를 여러 개 붙입니다. 임베딩 모델(OpenAI), 벡터 DB(Pinecone), 크롤링(Tavily), 관측 도구(LangSmith)가 각각 인증 키를 요구합니다. 비즈니스 로직 작성 전에 실행 가능한 안정적인 환경을 세팅하는 것이 최우선입니다.

```python
import certifi
import os
import ssl
from dotenv import load_dotenv

load_dotenv()

# SSL 인증서 번들을 명시해 일부 로컬/사내/VPN 환경의 인증서 검증 문제를 방지합니다.
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
```

> ⚠️ **현재 코드 점검 포인트**:
> `section7-RAG/ingestion.py`에는 `cetifi.where()` 오타가 있습니다. 그대로 실행하면 모듈 에러가 발생하므로 `certifi.where()`로 고쳐야 합니다.

---

## 4. Tavily 기반 문서 크롤링 및 Document 변환

> **"직접 크롤러를 짜지 않고 Tavily를 쓰는 이유? 그리고 왜 Dictionary를 바로 쓰지 않고 Document로 감싸나요?" (Why)**
> 문서 크롤링은 동적 사이트 렌더링, bot 캡챠 방어, 인증서 문제 등 운영 이슈가 다양해 검증된 외부 도구에 위임하는 편이 안전합니다. 또한, 이후 연결되는 LangChain 생태계의 splitter나 vector store 컴포넌트들은 주로 `Document` 인터페이스를 표준 입력으로 기대하기 때문입니다.

```python
from langchain_tavily import TavilyCrawl
from langchain_core.documents import Document

tavily_crawl = TavilyCrawl()

# 1) 사이트 크롤링
res = tavily_crawl.invoke({
    "url": "https://python.langchain.com/",
    "max_depth": 2, # 시작 URL에서 얼마나 깊게 링크를 따라갈지 결정
    "extract_depth": "advanced",
})

# 2) Langchain Document로 래핑
all_docs = []
for result in res["results"]:
    all_docs.append(
        Document(
            page_content=result["raw_content"],
            metadata={"source": result["url"]}, # 문서 출처 메타데이터
        )
    )
```

> 💡 **실무 포인트**:
> - `max_depth`를 곧바로 크게 잡으면 크롤링 범위가 기하급수적으로 폭증하여 과금 폭탄의 원인이 될 수 있습니다. 1이나 2부터 시작해서 수집 결과를 점검하고 올리는 것이 좋습니다.
> - 메타데이터에 최소한 `source`, `title`, `crawl_timestamp`를 함께 기록하면 추후 검색 품질 이슈가 발생했을 때 혹은 환각이 확인되었을 때 디버깅하는 데 결정적인 단서가 됩니다.

🔗 [Tavily LangChain Integration 문서](https://docs.tavily.com/documentation/integrations/langchain)
🔗 [LangChain Document 공식 문서](https://python.langchain.com/api_reference/core/documents/langchain_core.documents.base.Document.html)

---

## 5. 텍스트 청킹 (Text Chunking)

> **"왜 문서를 잘게 나누는(Chunking) 과정이 필수적일까요?" (Why)**
> 문서를 통째로 임베딩하면 넓은 문맥이 하나로 합쳐져(희석되어) 검색 정확도가 떨어집니다. 의미를 보존하면서도 검색에 용이한 작은 단위로 텍스트를 나누고 교차시키는 청킹 작업이 있어야만 가장 관련성 높은 문구를 찾아낼 수 있습니다.

```python
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter

# 4000자 단위로 자르고, 끊기는 부분의 문맥 보존을 위해 200자씩 겹치게(overlap) 설정합니다.
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=4000, 
    chunk_overlap=200
)

splitted_docs = text_splitter.split_documents(all_docs)
```

> 💡 **실무 포인트**: 
> 청크를 너무 작게 자르면 대명사 지칭 대상이 날아가 의미가 소실되고, 너무 크게 자르면 검색 정확도(Retrieval Precision)가 떨어집니다. `chunk_size`, `chunk_overlap`, 그리고 벡터 DB 조회 시의 검색 한도수(`Top-K`)는 한 세트처럼 함께 묶어서 튜닝 파라미터로 설정해야 합니다.

> 🔍 **내부 동작 원리**:
>
> ```text
> 원본 Document (엄청나게 긴 텍스트)
>   ↓
> 1차 분할 (문단 기준 "\n\n") → 길이가 여전히 오버되면 계속 쪼갬
> 2차 분할 (줄바꿈 기준 "\n")
> 3차 분할 (단어 기준 " ")
>   ↓
> 연속된 청크 생성 (chunk_overlap만큼 다음 청크에 중복 텍스트 부여)
> ```

🔗 [RecursiveCharacterTextSplitter 공식 문서](https://python.langchain.com/api_reference/text_splitters/character/langchain_text_splitters.character.RecursiveCharacterTextSplitter.html)

---

## 6. 임베딩 모델 초기화와 Rate Limit 대응

> **"대량의 문서를 벡터로 적재할 때 겪게 되는 핵심 난관은 무엇일까요?" (Why)**
> 클라우드 기반의 임베딩 API(OpenAI 등)는 등급별로 초당/분당 처리 가능한 토큰 수 제한이 빡빡하게 걸려있습니다. 제한을 무시하고 요청을 쏟아내면 API 서비스 측에서 429 에러(Rate Limit Error)를 반환하고 파이프라인이 즉시 마비됩니다. 

```python
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small", 
    chunk_size=50, # 한 번에 일괄 처리할 텍스트 수
    retry_min_seconds=10, # 429 에러 발생 시 최소 10초 대기 후 재시도
)
```

> 💡 **실무 포인트**:
> `retry_min_seconds` 파라미터가 없거나 너무 짧으면 요청 한도에 걸릴 때 백오프 타이머가 오작동해 작업이 무너집니다. `chunk_size`는 "문서 개수 한도"처럼 보이지만, 실제 병목과 속도 위반 원인은 API가 산정하는 합산 '토큰 수(TPM)'입니다. 예제와 같이 재시도 쿨타임을 걸어두면 내부적으로 지수 백오프(Exponential Backoff)를 수행해 문제를 피해갈 수 있습니다.

🔗 [OpenAIEmbeddings 공식 문서](https://python.langchain.com/api_reference/openai/embeddings/langchain_openai.embeddings.base.OpenAIEmbeddings.html)

---

## 7. 벡터 스토어 선택 및 비동기(Async) 배치 처리

> **"수천 개의 청크를 벡터 DB에 저장할 때 비동기 방식과 배치 처리를 동원하는 이유는 무엇일까요?" (Why)**
> 문서 청크를 하나 단위로 순차적으로 임베딩하고 저장하면 네트워크 왕복 대기 시간(I/O Bound) 때문에 전체 프로세스가 하루 종일 걸릴 수 있습니다. 해결책은 청크 데이터를 적정 크기의 **배치(Batch)**로 나누고 다수의 배치를 동시에 **비동기(Asynchronous)**로 밀어넣는 병렬 처리 아키텍처입니다.

| 데이터베이스 | 운용 방식 | 장점 | 단점 |
|------|------|------|------|
| **Chroma** | 로컬 / 파일 기반 SQLite 백엔드 | 설정이 매우 단순하고 로컬 실험에 아주 빠르다 | 팀원 간 DB 공유가 어렵고, 서버 볼륨 확장성 제약 |
| **Pinecone** | 완전 관리형 클라우드 서버리스 | 운영 편의, 확장성이 뛰어나며 즉각 조회가 가능하다 | 네트워크 의존성 / 인덱스 차원 매칭 오류 주의 |

비동기 방식으로 파이프라인 효율을 아래처럼 극대화할 수 있습니다.

```python
import asyncio
from langchain_chroma import Chroma
# from langchain_pinecone import PineconeVectorStore

# 로컬 테스트용 Chroma 벡터 스토어 초기화
vectorstore = Chroma(
    persist_directory="chroma_db", 
    embedding_function=embeddings
)

async def index_documents_async(documents: list, batch_size: int = 500):
    # 문서를 batch_size (예: 500개) 크기로 분할
    batches = [documents[i : i + batch_size] for i in range(0, len(documents), batch_size)]

    async def add_batch(batch: list, batch_num: int):
        try:
            # 벡터 저장소에 비동기로 문서 추가 (임베딩 적용 후 인덱싱 실행)
            await vectorstore.aadd_documents(batch)
            return True
        except Exception as e:
            return False

    # 분할된 모든 배치를 비동기로 동시 실행
    tasks = [add_batch(batch, i + 1) for i, batch in enumerate(batches)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
# 메인 Async 함수 내에서 대규모 인덱싱 실행
# await index_documents_async(splitted_docs, batch_size=500)
```

> 💡 **실무 포인트**:
> - 벡터 스토어 적재 시에는 임베딩 벡터와 함께 **반드시 원본 텍스트 구조(`page_content`)와 출처를 함께 저장**해야 합니다. 임베딩 벡터로는 원래 텍스트 문장을 복원할 수 없습니다.
> - Pinecone의 경우, 생성된 인덱스 차원(Dimension) 수와 현재 사용하는 `text-embedding-3-small` 임베딩 모델의 산출 차원 수가 다르다면 곧바로 실패합니다(차원 불일치 에러). 가장 기초적이지만 빈번하게 발생하는 실수입니다.

> 🔍 **내부 동작 원리 (배치 비동기 인덱싱)**:
>
> ```text
> 전체 분할 문서 청크 6,500개
>   ↓ (슬라이싱)
> Batch 1 (500) | Batch 2 (500) | ... | Batch 13 (500)
>   ↓ (asyncio.gather()로 비동기 동시 실행)
> Batch 1 → Embedding API 호출 대기 → Vector DB 적재
> Batch 2 → Embedding API 호출 대기 → Vector DB 적재
> ...
>   ↓
> I/O 블로킹 구간이 겹치면서 전체 인덱싱 시간이 압도적으로 절축됨
> ```

🔗 [PineconeVectorStore 참조 문서](https://python.langchain.com/api_reference/pinecone/vectorstores/langchain_pinecone.vectorstores.PineconeVectorStore.html)
🔗 [Chroma 공식 문서](https://python.langchain.com/api_reference/chroma/vectorstores/langchain_chroma.vectorstores.Chroma.html)

---

## 핵심 요약

```text
✅ [Long Context vs RAG] 롱 컨텍스트 LLM 시대에도 비용을 아끼고, 환각을 제어하고, 출처를 확보하는 데 RAG 모델은 핵심 요소다.
✅ [Ingestion Flow_1] 전체 아키텍처는 크롤링 수집(Tavily) → 공용 포맷 변환(Document) → 청크 쪼개기(Splitter) 파이프라인으로 구성된다.
✅ [Ingestion Flow_2] 문맥 파손 방지를 위해 RecursiveCharacterTextSplitter로 오버랩을 두고 적절히 데이터를 자른다.
✅ [병목 관리_1] 임베딩 시의 Rate Limit (429 에러) 방어 전략으로 retry_min_seconds 값을 통한 지수 백오프 전략이 필수적이다.
✅ [병목 관리_2] 대규모 청크의 네트워크 I/O 병목을 해결하기 위해 asyncio.gather()를 이용한 비동기식 배치 처리를 수행하여 적재 속도를 높인다.
✅ [저장소 선택] Vector DB는 로컬에서 빠르게 검증할 땐 Chroma가, 다중 클라이언트 서비스 운영에선 서버리스인 Pinecone이 적합하다.
✅ [주의점] certifi.where() 모듈 오타와 Pinecone 인덱스 차원 불일치는 파이프라인을 가장 빠르게 멈추게 하는 흔한 실수다.
```

---

## 실습 체크리스트

- [ ] `.env`에 `OPENAI_API_KEY`, `PINECONE_API_KEY`, `TAVILY_API_KEY`를 설정하고 작동 환경 사전 점검하기
- [ ] `cetifi.where()` 오타를 수정한 뒤 `ingestion.py`가 정상 실행되는지 확인하기
- [ ] Tavily의 `max_depth`에 따라 크롤링 결과 개수와 소요 시간이 얼마나 차이 나는지 비교 분석해 보기
- [ ] 생성된 `Document` 객체의 메타데이터에 실제 크롤링 URL 주소(`source`)가 올바로 매핑되어 있는지 확인하기
- [ ] `RecursiveCharacterTextSplitter`에서 `chunk_size`를 4000 또는 1000 단위로 각각 설정해 보고 전체 청크 개수 차이 분석하기
- [ ] 비동기 인덱싱 함수(`index_documents_async`)의 `batch_size`를 조절해 벡터 스토어의 적재 속도 병목 모니터링하기
- [ ] `OpenAIEmbeddings`의 `retry_min_seconds`를 주석 처리해 의도적으로 429 Rate Limit 에러 재현해 보기
- [ ] 데이터베이스를 바꿔 Chroma로 구동 후 프로젝트 루트 영역에 `chroma_db` 디렉토리가 생성되는지 확인하기
