import os
#from langchain_community.utilities import SQLDatabase
os.environ["OPENAI_API_KEY"] = ""
from langchain_community.utilities import SQLDatabase
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
print(db.dialect)
print(db.get_usable_table_names())
db.run("SELECT * FROM Artist LIMIT 10;")

from langchain.chains import create_sql_query_chain
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
chain = create_sql_query_chain(llm, db)
response = chain.invoke({"question": "How many employees are there"})
print(response)

print(db.run(response))

#from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool

#execute_query = QuerySQLDataBaseTool(db=db)
#write_query = create_sql_query_chain(llm, db)
#chain = write_query | execute_query
#chain.invoke({"question": "How many employees are there"})

#from operator import itemgetter

#from langchain_core.output_parsers import StrOutputParser
#from langchain_core.prompts import PromptTemplate
#from langchain_core.runnables import RunnablePassthrough

#answer_prompt = PromptTemplate.from_template(
#    """Given the following user question, corresponding SQL query, and SQL result, answer the user question.

#Question: {question}
#SQL Query: {query}
#SQL Result: {result}
#Answer: """
#)

#answer = answer_prompt | llm | StrOutputParser()
#chain = (
#    RunnablePassthrough.assign(query=write_query).assign(
#        result=itemgetter("query") | execute_query
#    )
#    | answer
#)

#print(chain.invoke({"question": "How many employees are there"}))

from langchain_community.agent_toolkits import create_sql_agent

agent_executor = create_sql_agent(llm, db=db, agent_type="openai-tools", verbose=True)

print(agent_executor.invoke(
    {
        "input": "List the total sales per country. Which country's customers spent the most?"
    }
))

######### few shot prompt #####
examples = [
    {"input": "List all artists.", "query": "SELECT * FROM Artist;"},
    {
        "input": "Find all albums for the artist 'AC/DC'.",
        "query": "SELECT * FROM Album WHERE ArtistId = (SELECT ArtistId FROM Artist WHERE Name = 'AC/DC');",
    },
    {
        "input": "List all tracks in the 'Rock' genre.",
        "query": "SELECT * FROM Track WHERE GenreId = (SELECT GenreId FROM Genre WHERE Name = 'Rock');",
    },
    {
        "input": "Find the total duration of all tracks.",
        "query": "SELECT SUM(Milliseconds) FROM Track;",
    },
    {
        "input": "List all customers from Canada.",
        "query": "SELECT * FROM Customer WHERE Country = 'Canada';",
    },
    {
        "input": "How many tracks are there in the album with ID 5?",
        "query": "SELECT COUNT(*) FROM Track WHERE AlbumId = 5;",
    },
    {
        "input": "Find the total number of invoices.",
        "query": "SELECT COUNT(*) FROM Invoice;",
    },
    {
        "input": "List all tracks that are longer than 5 minutes.",
        "query": "SELECT * FROM Track WHERE Milliseconds > 300000;",
    },
    {
        "input": "Who are the top 5 customers by total purchase?",
        "query": "SELECT CustomerId, SUM(Total) AS TotalPurchase FROM Invoice GROUP BY CustomerId ORDER BY TotalPurchase DESC LIMIT 5;",
    },
    {
        "input": "Which albums are from the year 2000?",
        "query": "SELECT * FROM Album WHERE strftime('%Y', ReleaseDate) = '2000';",
    },
    {
        "input": "How many employees are there",
        "query": 'SELECT COUNT(*) FROM "Employee"',
    },
]

from langchain_community.vectorstores import FAISS
from langchain_core.example_selectors import SemanticSimilarityExampleSelector
from langchain_openai import OpenAIEmbeddings

example_selector = SemanticSimilarityExampleSelector.from_examples(
    examples,
    OpenAIEmbeddings(),
    FAISS,
    k=5,
    input_keys=["input"],
)

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotPromptTemplate,
    MessagesPlaceholder,
    PromptTemplate,
    SystemMessagePromptTemplate,
)

system_prefix = """You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run, then look at the results of the query and return the answer.
Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.
You can order the results by a relevant column to return the most interesting examples in the database.
Never query for all the columns from a specific table, only ask for the relevant columns given the question.
You have access to tools for interacting with the database.
Only use the given tools. Only use the information returned by the tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.

If the question does not seem related to the database, just return "I don't know" as the answer.

Here are some examples of user inputs and their corresponding SQL queries:"""

few_shot_prompt = FewShotPromptTemplate(
    example_selector=example_selector,
    example_prompt=PromptTemplate.from_template(
        "User input: {input}\nSQL query: {query}"
    ),
    input_variables=["input", "dialect", "top_k"],
    prefix=system_prefix,
    suffix="",
)

full_prompt = ChatPromptTemplate.from_messages(
    [
        SystemMessagePromptTemplate(prompt=few_shot_prompt),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

import streamlit as st

st.title("DB Query Assistance")
#st.text("Query:")
query = st.text_area("Enter query:")
submit = st.button("Submit")

if submit:

# Example formatted prompt
    prompt_val = full_prompt.invoke(
        {
            "input": query,
            "top_k": 5,
            "dialect": "SQLite",
            "agent_scratchpad": [],
        }
    )
    #print(prompt_val.to_string())

    agent = create_sql_agent(
        llm=llm,
        db=db,
        prompt=full_prompt,
        verbose=True,
        agent_type="openai-tools",
    )

    st.write(agent.invoke({"input": query}))

#import streamlit as st

#title = st.text_input('Query', 'How many artists are there')
#st.write('The answer:', agent.invoke({"input": "How many artists are there?"}))


