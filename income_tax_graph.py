# %% [markdown]
# https://arxiv.org/abs/2310.11511
# 
# ![image.png](attachment:image.png)

# %%
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")

vector_store = Chroma(
    embedding_function=embedding_function,
    collection_name = 'income_tax_collection',
    persist_directory = './income_tax_collection'
)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})


# %%
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    query: str
    context: List[Document]
    answer: str

graph_builder = StateGraph(AgentState)

# %%
def retrieve(state: AgentState) -> AgentState:
    query = state['query']  
    docs = retriever.invoke(query)  
    return {'context': docs}  


# %%
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model='gpt-4o')

# %%
from langchain import hub

generate_prompt = hub.pull("rlm/rag-prompt")
generate_llm = ChatOpenAI(model='gpt-4o', max_completion_tokens=100)

def generate(state: AgentState) -> AgentState:
    context = state['context']  
    query = state['query']      
    
    rag_chain = generate_prompt | generate_llm
    
    response = rag_chain.invoke({'question': query, 'context': context})
    
    return {'answer': response.content}  


# %% [markdown]
# - `retrieve` 노드에서 추출된 문서와 사용자의 질문이 관련이 있는지 판단
#     - 문서와 관련이 없다면 질문에 답변할 수 없다고 판단하고 마무리
#     - 문서와 관련이 있다면 `generate` 노드로 이동해서 답변을 생성

# %%
from langchain import hub
from typing import Literal

doc_relevance_prompt = hub.pull("langchain-ai/rag-document-relevance")

def check_doc_relevance(state: AgentState) -> Literal['relevant', 'irrelevant']:
    query = state['query']  
    context = state['context']  

    doc_relevance_chain = doc_relevance_prompt | llm
    
    response = doc_relevance_chain.invoke({'question': query, 'documents': context})

    if response['Score'] == 1:
        return 'relevant'
    
    return 'irrelevant'

# %%
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

dictionary = ['사람과 관련된 표현 -> 거주자']

rewrite_prompt = PromptTemplate.from_template(f"""
사용자의 질문을 보고, 우리의 사전을 참고해서 사용자의 질문을 변경해주세요 
사전: {dictionary}                                           
질문: {{query}}
""")

def rewrite(state: AgentState) -> AgentState:
    query = state['query']  
    
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    response = rewrite_chain.invoke({'query': query})
    
    return {'query': response}


# %% [markdown]
# - 답변이 `retrieve` 노드에서 추출된 문서와 관련이 있는지 판단
#     - 문서와 관련이 없다면 `generate` 노드로 이동해서 답변을 다시 생성
#     - 문서와 관련이 있다면 `check_helpfulness` 노드로 이동해서 답변이 도움이 되는지 판단

# %%
from langchain_core.prompts import PromptTemplate

hallucination_prompt = PromptTemplate.from_template("""
You are a teacher tasked with evaluating whether a student's answer is based on facts or not,  
Given facts, which are excepts from income tax law, and a student's answer;                         
If the student's answer is based on documents, resond with "not hallucinated"
If the student's answer is not based on documents, resond with "hallucinated"

documents : {documents}
student_answer : {student_answer}
""")

hallucination_llm = ChatOpenAI(model='gpt-4o', temperature=0)

def check_hallucination(state: AgentState) -> Literal['hallucinated', 'not hallucinated']:
    answer = state['answer']  
    context = state['context']
    context = [doc.page_content for doc in context]

    hallucination_chain = hallucination_prompt | hallucination_llm | StrOutputParser()
    
    response = hallucination_chain.invoke({"student_answer": answer, "documents": context})
    
    return response

# %%
query = "연봉이 5천만원인 거주자의 소득세는 얼마인가요?"
context = retriever.invoke(query)

print("document start")
for document in context:
    print(document.page_content)
print("document end")

generate_state = {"query": query, "context": context}
answer = generate(generate_state)
print(f"answer == {answer}")

hallucination_state = {"answer": answer, "context": context}

check_hallucination(hallucination_state)



# %% [markdown]
# 
# - 생성된 답변이 사용자의 질문과 관련이 있는지 판단
#     - 질문과 관련이 있다면 사용자에게 답변을 리턴
#     - 질문과 관련이 없다면 `rewrite` 노드로 이동해서 사용자의 질문을 변경
#         - `rewrite` 후 다시 `retrieve` 합니다
# - `check_helpfulness` node를 생성하지 않고 `check_helpfulness_grader`만 추가해서 conditional_edge로 사용

# %%
from langchain import hub

helpfulness_prompt = hub.pull("langchain-ai/rag-answer-helpfulness")

def check_helpfulness_grader(state: AgentState) -> Literal['helpful', 'unhelpful']:
    query = state['query']
    answer = state['answer']

    helpfulness_chain = helpfulness_prompt | llm

    response = helpfulness_chain.invoke({"question": query, "student_answer": answer})
    print(f"helpfulness response : {response}")

    if response["Score"] == 1:
        return "helpful"
    return "unhelpful"

def check_helpfulness(state: AgentState) -> AgentState:
    return state

# %% [markdown]
# - node를 추가하고 edge로 연결

# %%
query = "연봉이 5천만원인 거주자의 소득세는 얼마인가요?"
context = retriever.invoke(query)

generate_state = {"query": query, "context": context}
answer = generate(generate_state)
print(f"answer == {answer}")

helpfulness_state = {"query": query, "answer": answer}

check_helpfulness(helpfulness_state)



# %%
graph_builder.add_node("retrieve", retrieve)
graph_builder.add_node("generate", generate)
graph_builder.add_node("rewrite", rewrite)
graph_builder.add_node("check_doc_relevance", check_doc_relevance)
graph_builder.add_node("check_hallucination", check_hallucination)
graph_builder.add_node("check_helpfulness", check_helpfulness)

# %%
from langgraph.graph import START, END

graph_builder.add_edge(START, "retrieve")
graph_builder.add_conditional_edges(
    "retrieve",
    check_doc_relevance,
    {
        "relevant": "generate",
        "irrelevant": END
    }
)
graph_builder.add_conditional_edges(
    "generate",
    check_hallucination,
    {
        "not hallucinated": "check_helpfulness",
        "hallucinated": "generate"
    }
)
graph_builder.add_conditional_edges(
    "check_helpfulness",
    check_helpfulness_grader,
    {
        "helpful": END,
        "unhelpful": "rewrite"
    }
)

graph_builder.add_edge("rewrite", "retrieve")

# %%
graph = graph_builder.compile()


