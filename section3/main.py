from langchain_core.messages import ToolMessage
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

"""
여기서 : 를 기준으로 좌측은 provider, 우측은 모델이고 
init_chat_model이 이 모델 정보를 받으면 둘로 나눠서 해당 provider에 맞는 모델을 불러 옴.

여기서 반환값은 BaseChatModel 인터페이스를 구현함 
"""
llm = init_chat_model(model="ollama:qwen2.5:7b", temperature=0)

"""
LLM이 이해할수 있는 JSON 스키마를 자동으로 생성한다. 
어떤 정보를 캐치하는지는 :
- tool 이름: 함수 이름
- 설명 : docstring 첫줄
- 인자 이름.타입: 마라미터 타입 힌팅
- 파라미터 유무 : required 등록 여부 결정

주의: 타입 힌팅 없이 @tool을 사용하면 스키마 생성이 실패 하거나 불안전해진다. 
타입 힌팅은 필수다
"""


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog.

    Args:
        product: 조회할 제품 이름 (예: laptop, headphones, keyboard)
    """
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply a discount tier to a price and return the final price.
    Available tiers: bronze (5%), silver (12%), gold (23%).

    Args:
        price: 원래 가격
        discount_tier: 할인 등급 (bronze, silver, gold)
    """
    discount_percentage = {"bronze": 0.05, "silver": 0.12, "gold": 0.23}
    discount = discount_percentage.get(discount_tier, 0)
    return price * (1 - discount)


tools = [get_product_price, apply_discount]

# Tool 이름으로 빠르게 조회하기 위한 dict (Agent Loop에서 사용)
tools_dict = {t.name: t for t in tools}

"""
각 tool의 JSON 스키마를 수집 
provider에 맞는 포맷으로 변환 
-> Anthropic 이면 tools_schema = [t.to_anthropic_tool_schema() for t in tools]
-> OpenAI 이면 tools_schema = [t.to_openai_tool_schema() for t in tools]

llm_with_tools.invoke(message) 호출마다: payload에 메시지와 툴을 같이 제공

이 bind_tool을 빠뜨리고 llm.invoke(message)를 호출하면, 에이전트는 이 존재 자체를 모르기 때문에 
항상 tool_calls 가 항상 빈 리스트로 반환 된다 
"""
llm_with_tools = llm.bind_tools(tools)

"""
매시지 타입과 OPENAI 맵핑
SystemMessage - LLM 행동 지침 : "system"
HumanMessage - 사용자 입력 : "user"
ToolMessage - Tool 실행 결과 : "tool"
AIMessage - LLM 응답 : "assistant"
"""
messages = [
    SystemMessage(
        content="""
    You are a helpful assistant. 
    You have access to a product catalog tool and a discount tool.

    "STRICT RULES:\n"
        "1. NEVER guess prices — always call get_product_price first.\n"
        "2. Call apply_discount ONLY after getting the real price.\n"
        "3. NEVER calculate discounts yourself — always use apply_discount tool.\n"
        "4. If no discount tier specified, ask the user — do NOT assume one."
    """
    ),
    HumanMessage(content="What is the price of a laptop?"),
]


MAX_ITERATIONS = 10


def run(messages):

    for iteration in range(1, MAX_ITERATIONS + 1):
        # [Thought] LLM 호출 → "다음에 뭘 할지" 결정
        ai_message = llm_with_tools.invoke(messages)
        tool_calls = ai_message.tool_calls

        # [Final Answer] tool_call 없으면 → 최종 답변 반환
        if not tool_calls:
            return ai_message.content

        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        # [Execute] Tool 실제 실행
        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found in tools_dict")
        observation = tool_to_use.invoke(tool_args)
        print(f"  [Tool Result] {observation}")

        # [Observe] 결과를 messages에 추가 → LLM이 다음 호출에서 기억함
        messages.append(ai_message)  # AI의 tool_call 결정 기록
        messages.append(
            ToolMessage(
                content=str(observation),
                tool_call_id=tool_call_id,  # 반드시 일치시킬 것
            )
        )

    # 최대 반복 도달 시
    print("ERROR: Max iterations reached without a final answer")
    return None
