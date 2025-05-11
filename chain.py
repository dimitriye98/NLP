"""LangChain system for querying flight price data using natural language."""

import time

from pathlib import Path
from typing import List, Union

from dotenv import load_dotenv
from langchain import hub
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.openapi.toolkit import RequestsToolkit
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_community.utilities.requests import TextRequestsWrapper
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
from langchain_together import ChatTogether

# Load environment variables (for Together API key)
load_dotenv()

def create_db_chain(db_path: Union[str, Path] = "flights.db") -> object:
    """Create a LangChain agent for querying the flights database.

    Args:
        db_path: Path to the SQLite database file

    Returns:
        An agent executor that can answer questions about the flight data
    """
    # Create database connection
    db_uri = f"sqlite:///{db_path}"
    db = SQLDatabase.from_uri(db_uri)


    # Initialize language model
    llm = ChatTogether(
        model="deepseek-ai/DeepSeek-V3",  # Using Mixtral for better SQL generation
        temperature=0,  # We want deterministic SQL queries
        max_tokens=1000  # Allow for longer responses
    )

    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    tools = toolkit.get_tools()
    del tools[2] # Remove the list tables tool, not necessary since we work with a single table

    tools.append(DuckDuckGoSearchResults(output_format="list"))

    webtoolkit = RequestsToolkit(
        requests_wrapper=TextRequestsWrapper(),
        allow_dangerous_requests=True,
    )
    tools.extend(webtoolkit.get_tools())


    system_message = """
    System: You are an EXPERT flight booking agent designed to identify the best flight options for a user.
    You specialize in complex compound queries, often involving multiple flights and conditions.
    You are an EXPERT in composing complex SQL queries to answer questions about the flight data.
    You have access to a toolkit for web browsing. This allows you to search the web for information.
    Do NOT use the web toolkit for flight information, use the database instead
    When performing searches, always request the linked pages from the search results and read them before returning an answer.
    Do NOT answer based on only the snippets provided by the search results without reading the linked pages.
    Do NOT answer from memory, when asked a factual question (other than one directly pertaining to flight itinerary information) always perform a web search to verify the answer.
    Do NOT query the database for non-flight itinerary information. If the user asks a question which is not directly pertaining to flight itinerary information, always perform a web search to verify the answer.
    You have access to an SQLite database of one-way flight prices and availability with a single table named itineraries.
    When asked a question involving flight itinerary information you should compose a syntatically correct SQLite query to run, then look at the results of the query and return the answer.
    You have access to a tool for verifying syntactic validity of a query and should always utilize it prior to running the query.
    Unless the user specifies a specific number of examples they wish, always limit your query to at most 5 results.
    Never query for all the columns from the table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    Use only one tool at a time, waiting for its result prior to performing a subsequent tool call.
    Do NOT attempt to make multiple tool calls at once.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    Do NOT attempt to make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. Your access is read-only and these WILL error.

    WHEN performing a database query you should ALWAYS query the schema of the itineraries table.
    Do NOT skip this step.
    Then you should query the data in the table to answer the user's question.

    REMEMBER flights in the itineraries table are one-way flights. To meet user requirements, you are very likely to need to perform self cartesian JOINs on the table to consider all segments the user's itinerary may contain.

    Today's date is: April 16, 2022. If the user does not specify otherwise, you SHALL search for flights within one year from now.
    """

    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_message
    )

    return agent_executor

def query_flights(user_question: str, agent_executor=None, db_path: Union[str, Path] = "flights.db"):
    """Query the flights database using natural language.

    Args:
        user_question: Natural language question about the flight data
        agent_executor: Optional pre-created agent executor
        db_path: Path to the SQLite database if agent_executor not provided

    Returns:
        The agent's responses to the question
    """
    # Create agent if not provided
    if agent_executor is None:
        agent_executor = create_db_chain(db_path)

    for step in agent_executor.stream({
        "messages": [{"role": "user", "content": user_question}]
    }, stream_mode="values"):
        step["messages"][-1].pretty_print()

agent = create_db_chain()
if __name__ == "__main__":
    # Example questions
    example_questions = [
        "What are the most expensive routes in the dataset?",
        "What is the average price for flights from New York to London?",
        "Which cities have the most flights?"
    ]

    for q in example_questions:
        print(f"\nQuestion: {q}")
        query_flights(q, agent)
