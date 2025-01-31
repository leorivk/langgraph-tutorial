# %% [markdown]
# - 아주 specific한 에이전트를 개발하는 경우 유리
# - 답변 생성 시 다양한 정보가 필요하다면 병렬 처리를 통해 시간 절약 가능

# %%
from dotenv import load_dotenv

load_dotenv()

# %%
from typing_extensions import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str  # 사용자 질문
    answer: str  # 세율
    tax_base_equation: str  # 과세표준 계산 수식
    tax_deduction: str  # 공제액
    market_ratio: str  # 공정 시장 가액 비율
    tax_base: str  # 과세표준

graph_builder = StateGraph(AgentState)

# %%
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name = 'real_estate_tax_collection',
    persist_directory = './real_estate_tax_collection'
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# %%
query = "5억짜리 집 1채, 10억짜리 집 1채, 20억짜리 집 1채를 가지고 있을 때, 세금을 얼마나 내나요?"

# %%
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model='gpt-4o')
small_llm = ChatOpenAI(model='gpt-4o-mini')

rag_prompt = hub.pull("rlm/rag-prompt")



# %%
tax_base_retrieval_chain = (
    {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()
)

tax_base_equation_prompt = ChatPromptTemplate.from_messages([
    ("system", "사용자의 질문에서 과세표준을 계산하는 방법을 수식으로 나타내주세요. 부연 설명 없이 수식만 나타내 주세요."),
    ("user", "{tax_base_equation_information}")
])

tax_base_equation_chain = (
    {"tax_base_equation_information": RunnablePassthrough()} | tax_base_equation_prompt | llm | StrOutputParser()
)

tax_base_chain = {"tax_base_equation_information": tax_base_retrieval_chain} | tax_base_equation_chain

def get_tax_base_equation(state: AgentState) -> AgentState:
    """
    종합부동산세 과세표준 계산 수식
    `node`로 활용되기 때문에 `state`를 인자로 받지만, 
    고정된 기능을 수행하기 때문에 `state` 활용 X

    Args:
        state (AgentState): 현재 에이전트의 상태를 나타내는 객체

    Returns:
        AgentState: 'tax_base_equation' 키를 포함하는 새로운 `state` 반환
    """
    # 과세표준을 계산하는 방법을 묻는 질문 정의
    tax_base_equation_question = '주택에 대한 종합부동산세 계산시 과세표준을 계산하는 방법을 수식으로 표현해서 알려주세요'
    
    # tax_base_chain을 사용하여 질문 실행 및 결과 추출
    tax_base_equation = tax_base_chain.invoke(tax_base_equation_question)
    
    # state에서 'tax_base_equation' 키에 대한 값 반환
    return {'tax_base_equation': tax_base_equation}

# %%
get_tax_base_equation({})

# %% [markdown]
# - LLM은 번역, 요약, 분석에 능하지만 이들을 한 번에 수행하는 것은 잘 하지 못함

# %%
tax_deduction_chain = (
    {"context": retriever, "question": RunnablePassthrough()} | rag_prompt | llm | StrOutputParser()
)

def get_tax_deduction(state: AgentState) -> AgentState:
    """
    종합부동산세 공제금액에 관한 정보 추출
    `node`로 활용되기 때문에 `state`를 인자로 받지만, 
    고정된 기능을 수행하기 때문에 `state` 활용 X

    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체

    Returns:
        AgentState: 'tax_deduction' 키를 포함하는 새로운 state 반환
    """
    # 공제금액을 묻는 질문 정의
    tax_deduction_question = '주택에 대한 종합부동산세 계산시 공제금액을 알려주세요'
    
    # tax_deduction_chain을 사용하여 질문 실행 및 결과 추출
    tax_deduction = tax_deduction_chain.invoke(tax_deduction_question)
    
    # state에서 'tax_deduction' 키에 대한 값 반환
    return {'tax_deduction': tax_deduction}

# %%
get_tax_deduction({})

# %%
from langchain_community.tools import TavilySearchResults
from datetime import date

tavily_search_tool = TavilySearchResults(
    max_results=5,
    search_depth="advanced",
    include_answer=True,
    include_raw_content=True,
    include_images=True,
)

tax_market_ratio_prompt = ChatPromptTemplate.from_messages([
    ("system", f"아래 정보를 기반으로 사용자의 질문에 답변해주세요.\n\nContext:\n{{context}}"),
    ("human", "{query}")
])

def get_market_ratio(state: AgentState) -> AgentState:
    """
    web 검색을 통해 주택 공시가격에 대한 공정시장가액비율 추출
    `node`로 활용되기 때문에 `state`를 인자로 받지만, 
    고정된 기능을 수행하기 때문에 `state` 활용 X
    
    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체

    Returns:
        AgentState: 'market_ratio' 키를 포함하는 새로운 state 반환
    """
    # 오늘 날짜에 해당하는 공정시장가액비율을 묻는 쿼리 정의
    query = f'오늘 날짜:({date.today()})에 해당하는 주택 공시가격 공정시장가액비율은 몇%인가요?'
    
    # tavily_search_tool을 사용하여 쿼리 실행 및 컨텍스트 추출
    context = tavily_search_tool.invoke(query)
    
    # tax_market_ratio_chain 구성 및 쿼리와 컨텍스트 처리
    tax_market_ratio_chain = (
        tax_market_ratio_prompt
        | llm
        | StrOutputParser()
    )
    
    # tax_market_ratio_chain 사용하여 시장 비율 계산
    market_ratio = tax_market_ratio_chain.invoke({'context': context, 'query': query})
    
    # state에서 'market_ratio' 키에 대한 값 반환
    return {'market_ratio': market_ratio}


# %%
get_market_ratio({})

# %%
from langchain_core.prompts import PromptTemplate

tax_base_calculation_prompt = PromptTemplate.from_template("""
주어진 내용을 기반으로 과세 표준을 계산해주세요.
과세 표준 계산 공식: {tax_base_equation}
공제액: {tax_deduction}
공정시장가액비율: {market_ratio}
사용자 주택 공시 가격 정: {query}
""")

def calculate_tax_base(state: AgentState):
    """
    주어진 state에서 과세표준 계산

    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체

    Returns:
        AgentState: 'tax_base' 키를 포함하는 새로운 state 반환
    """
    # state에서 필요한 정보 추출
    tax_base_equation = state['tax_base_equation']
    tax_deduction = state['tax_deduction']
    market_ratio = state['market_ratio']
    query = state['query']
    
    # tax_base_calculation_chain 구성 및 과세표준 계산
    tax_base_calculation_chain = (
        tax_base_calculation_prompt
        | llm
        | StrOutputParser()
    )
    
    # tax_base_calculation_chain 사용하여 과세표준 계산
    tax_base = tax_base_calculation_chain.invoke({
        'tax_base_equation': tax_base_equation,
        'tax_deduction': tax_deduction,
        'market_ratio': market_ratio,
        'query': query
    })

    # state에서 'tax_base' 키에 대한 값 반환
    return {'tax_base': tax_base}

# %%
tax_rate_calculation_prompt = ChatPromptTemplate.from_messages([
    ('system', '''당신은 종합부동산세 계산 전문가입니다. 아래 문서를 참고해서 사용자의 질문에 대한 종합부동산세를 계산해주세요

종합부동산세 세율:{context}'''),
    ('human', '''과세표준과 사용자가 소지한 주택의 수가 아래와 같을 때 종합부동산세를 계산해주세요

과세표준: {tax_base}
주택 수:{query}''')
])

def calculate_tax_rate(state: AgentState):
    """
    주어진 state에서 세율 계산 함수입니다.

    Args:
        state (AgentState): 현재 에이전트의 state를 나타내는 객체

    Returns:
        dict: 'answer' 키를 포함하는 새로운 state를 반환
    """
    # state에서 필요한 정보를 추출
    query = state['query']
    tax_base = state['tax_base']
    
    # retriever를 사용하여 쿼리를 실행하고 컨텍스트 추출
    context = retriever.invoke(query)
    
    # tax_rate_chain을 구성하여 세율 계산
    tax_rate_chain = (
        tax_rate_calculation_prompt
        | llm
        | StrOutputParser()
    )
    
    # tax_rate_chain을 사용하여 세율 계산
    tax_rate = tax_rate_chain.invoke({
        'context': context, 
        'tax_base': tax_base, 
        'query': query
    })

    # state에서 'answer' 키에 대한 값을 반환
    return {'answer': tax_rate}

# %%
graph_builder.add_node('get_tax_base_equation', get_tax_base_equation)
graph_builder.add_node('get_tax_deduction', get_tax_deduction)
graph_builder.add_node('get_market_ratio', get_market_ratio)
graph_builder.add_node('calculate_tax_base', calculate_tax_base)
graph_builder.add_node('calculate_tax_rate', calculate_tax_rate)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, 'get_tax_base_equation')
graph_builder.add_edge(START, 'get_tax_deduction')
graph_builder.add_edge(START, 'get_market_ratio')
graph_builder.add_edge('get_tax_base_equation', 'calculate_tax_base')
graph_builder.add_edge('get_tax_deduction', 'calculate_tax_base')
graph_builder.add_edge('get_market_ratio', 'calculate_tax_base')
graph_builder.add_edge('calculate_tax_base', 'calculate_tax_rate')
graph_builder.add_edge('calculate_tax_rate', END)

# %%
graph = graph_builder.compile()

# %%
