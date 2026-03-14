from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from dotenv import load_dotenv

load_dotenv()

# 1. 단일 Runnable 객체들
prompt = ChatPromptTemplate.from_template("{topic}에 대해 짧은 농담을 해줘.")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# 2. RunnableSequence 생성 (| 연산자)
# 내부적으로 prompt, model, parser가 RunnableSequence로 묶입니다.
joke_chain = prompt | model | parser


# 3. RunnableParallel의 암시적 생성 (딕셔너리 사용)
# 딕셔너리를 사용하면 내부적으로 RunnableParallel로 정규화됩니다.
parallel_chain = RunnableParallel({
    "joke": joke_chain,
    "original_topic": RunnablePassthrough()
})


# 4. 공통 인터페이스 invoke() 사용
print("=== Invoke 결과 ===")
result = parallel_chain.invoke({"topic": "AI 엔지니어"})
print(result)

# 5. 공통 인터페이스 stream() 사용
# 전체 병렬 체인에 대해서도 동일하게 stream 메서드 적용이 가능합니다.
print("\n=== Stream 결과 ===")
for chunk in parallel_chain.stream({"topic": "파이썬"}):
    print(chunk, end="", flush=True)