from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from tools import web_search,scrape_url
from langchain_mistralai import ChatMistralAI

# 1. Env Setup
load_dotenv()


# 2. Model Setup
model = ChatMistralAI(
        model="open-mistral-7b",
        temperature=0,
        max_tokens=400,
    )

# 3.1. Build Agent - search_agent
def create_search_agent():
    return create_agent(
        model=model,
        tools=[web_search]
    )

# 3.2. Build Agent - reader_agent
def create_reader_agent():
    return create_agent(
        model=model,
        tools=[scrape_url]
    )


# Create Writer chain

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert research writer. Write clear, structured and insightful reports."),
    ("human", """Write a detailed research report on the topic below.

Topic: {topic}

Research Gathered:
{research}

Structure the report as:
- Introduction
- Key Findings (minimum 3 well-explained points)
- Conclusion
- Sources (list all URLs found in the research)

Be detailed, factual and professional."""),
])

writer_chain = writer_prompt|model|StrOutputParser()


# critic chain

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "you are a sharp and constructive reserach critic. Be specific, factual, genuine, reasonable"),
    ("human","""
Review the reserach repoort and evalute strictly as 
Report :{report}
Respond in the exact format
Score : X/10
Strengths:
-...
-...
Areas to improve:
-...
-...
One line verdict
...."""),
])

critic_chain = critic_prompt|model|StrOutputParser()
