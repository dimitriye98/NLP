"""LangChain system for querying flight price data using natural language."""

import time

from pathlib import Path
from typing import List, Union

from dotenv import load_dotenv
from langchain import hub
from langchain_community.agent_toolkits import SQLDatabaseToolkit
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

    # prompt_template = hub.pull("langchain-ai/sql-agent-system-prompt")
    # assert len(prompt_template.messages) == 1
    # system_message = prompt_template.format(dialect="SQLite", top_k=5)
    # print(system_message)
    system_message = """
    System: You are a flight booking agent designed to identify the best flight options for a user.
    You specialize in complex compound queries, often involving multiple flights and conditions.
    You have access to an SQLite database of one-way flight prices and availability with a single table named itineraries.
    You will be asked a question and should compose a syntatically correct SQLite query to run, then look at the results of the query and return the answer.
    You have access to a tool for verifying syntactic validity of a query and should always utilize it prior to running the query.
    Unless the user specifies a specific number of examples they wish, always limit your query to at most 5 results.
    Never query for all the columns from the table, only ask for the relevant columns given the question.
    You have access to tools for interacting with the database.
    Only use the below tools. Only use the information returned by the below tools to construct your final answer.
    Use only one tool at a time, waiting for its result prior to performing a subsequent tool call.
    Do NOT attempt to make multiple tool calls at once.
    You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

    Do NOT attempt to make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database. Your access is read-only and these WILL error.

    To start you should ALWAYS query the schema of the itineraries table.
    Do NOT skip this step.
    Then you should query the data in the table to answer the user's question.
    """

    # Create SQL agent with error handling
    # agent_executor = create_sql_agent(
    #     llm=llm,
    #     db=db,
    #     agent_type="zero-shot-react-description",
    #     verbose=True,
    #     handle_parsing_errors=True,  # Handle output parsing errors gracefully
    #     max_iterations=3  # Limit retries on errors
    # )
    agent_executor = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_message
    )

    return agent_executor

def query_flights(user_question: str, agent_executor=None, db_path: Union[str, Path] = "flights.db"):# -> List[str]:
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
    # try:
    # Get the response
    # result = agent_executor.invoke({"input": user_question})
    # return [result["output"]]
    # except Exception as e:
    #     # raise e
    # return [f"Error querying database: {str(e)}"]

agent = create_db_chain()
if __name__ == "__main__":
    # Example usage
    # agent = create_db_chain()

    # Example questions
    example_questions = [
        "What are the most expensive routes in the dataset?",
        "What is the average price for flights from New York to London?",
        "Which cities have the most flights?"
    ]

    for q in example_questions:
        print(f"\nQuestion: {q}")
        query_flights(q, agent)
        # answers = query_flights(q, agent)
        # time.sleep(1)
        # for answer in answers:
        #     print(f"Answer: {answer}")
