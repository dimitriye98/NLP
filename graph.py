from typing import TypedDict, Union, Annotated, Sequence
from enum import Enum

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_together import ChatTogether
from langgraph.graph import StateGraph, END
from duckduckgo_search import DDGS
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages

# Initialize database connection
db = SQLDatabase.from_uri("sqlite:///flights.db")
llm = lambda t: ChatTogether(
    model="Qwen/Qwen2.5-VL-72B-Instruct",
    temperature=t,
    max_tokens=1000
)

toolkit = SQLDatabaseToolkit(db=db, llm=llm(0.3)) # We want some variation in the SQL queries, to allow for self-correction of mistakes
schema = toolkit.get_tools()[1].invoke("itineraries")

# Define our query types
class QueryType(str, Enum):
    FLIGHT_ITINERARY = "flight_itinerary"
    GENERAL_TRAVEL = "general_travel"
    UNRELATED = "unrelated"

# Define our state
class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    query_type: QueryType | None
    current_search_results: list[str] | None
    query_attempts: int | None
    sql_query: str | None
    next_action: str | None

# Function to classify the query type
def classify_query(previous_state: State | str) -> State:
    if isinstance(previous_state, str):
        state = State(
            messages=[HumanMessage(content=previous_state)],
            query_type=None,
            current_search_results=[],
            query_attempts=0,
            sql_query="",
            next_action=""
        )
    else:
        state = previous_state

    last_message = state["messages"][-1]
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a query classifier. Determine if the user's question is about:
        1. flight_itinerary - Questions about specific flights, routes, or itineraries
        2. general_travel - General travel questions about destinations, tips, etc.
        3. unrelated - Questions not related to travel
        Respond with ONLY ONE of these exact terms."""),
        ("human", "{query}")
    ])

    chain = prompt | llm(0) # Classification should be deterministic
    result = chain.invoke({"query": last_message.content})

    # Extract the last word from the response and convert to QueryType
    content = result.content
    if isinstance(content, str):
        # Split by whitespace and take the last non-empty word
        words = [w for w in content.split() if w.strip()]
        query_type = words[-1].strip().lower() if words else "unrelated"
    else:
        # If we somehow get a non-string, convert to string first and process similarly
        words = [w for w in str(content).split() if w.strip()]
        query_type = words[-1].strip().lower() if words else "unrelated"

    try:
        state["query_type"] = QueryType(query_type)
    except ValueError:
        # If the LLM returns an invalid type, default to unrelated
        state["query_type"] = QueryType.UNRELATED

    state["messages"] = add_messages(state["messages"], [result]) # For debugging
    return state

# Function to handle unrelated queries
def handle_unrelated(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a witty assistant. The user has asked a question unrelated to travel. Respond with a humorous message that politely refuses to answer while suggesting they ask about travel instead."),
        MessagesPlaceholder(variable_name="messages")
    ])

    chain = prompt | llm(0.7) # Allow for a good bit of creativity in witty responses
    response = chain.invoke({"messages": state["messages"]})

    state["messages"].append(response)
    state["next_action"] = "end"
    return state

# Function to perform search for general travel queries
def perform_search(state: State) -> State:
    # First, use LLM to compose an optimized search query
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a search query optimizer. Convert the user's travel question into
        a focused search query that will yield the most relevant results. Focus on key terms
        and remove unnecessary words. Do not include special search operators. Respond ONLY with the search query and nothing else."""),
        MessagesPlaceholder(variable_name="messages")
    ])

    chain = prompt | llm(0)  # Use temperature 0 for consistent query generation
    response = chain.invoke({"messages": state["messages"]})

    # Ensure we get a string from the response
    content = response.content
    if isinstance(content, str):
        optimized_query = content.strip()
    else:
        # If we somehow get a non-string, convert to string first
        optimized_query = str(content).strip()

    with DDGS() as ddgs:
        results = list(ddgs.text(optimized_query, max_results=10))

    search_results =[result["body"] for result in results]
    state["current_search_results"] = search_results
    state["messages"] = add_messages(state["messages"], [ToolMessage(content="\n".join([f"Search query used: {optimized_query}"] + search_results), tool_call_id="")])
    state["next_action"] = "process_search"
    return state

# Function to process and present search results
def process_search_results(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a travel information assistant. Use the search results to answer
        the user's query. If you need more information, include 'SEARCH:' followed by what you
        want to search for next in your response. For example: 'SEARCH: visa requirements for Bali'.
        Only do this if the current results don't adequately answer the question."""),
        MessagesPlaceholder(variable_name="messages"),
    ])

    chain = prompt | llm(0.3)  # Slight variation allowed for natural responses
    response = chain.invoke({
        "messages": state["messages"],
        "search_results": state["current_search_results"]
    })

    state["messages"].append(response)

    # Check if the response requests another search
    content = response.content
    if isinstance(content, str) and "SEARCH:" in content:
        state["next_action"] = "search"
    else:
        state["next_action"] = "end"

    return state

# Function to compose SQL query
def compose_sql_query(state: State) -> State:
    prompt = ChatPromptTemplate.from_messages([
        ("system", f"""
            You are an SQLite expert working with a flights database. Convert the user's flight itinerary question into a SQL query.
            The database has an itineraries table with the following schema:
                {schema}
            These are ONE-WAY flights, not round trips. Compose joins as necessary to satisfy the user's query.
            Think through your approach prior to responding. Ensure that the SQLite query is enclosed in markdown code tags flagged for SQL, as follows:
            ```sql
                [your query here]
            ```
            Today's date is April 14, 2022.
        """),
        MessagesPlaceholder(variable_name="messages")
    ])

    chain = prompt | llm(0.3)
    response = chain.invoke({"messages": state["messages"]})

    # Extract SQL query from markdown
    content = response.content
    state["query_attempts"] = state.get("query_attempts", 0) + 1  # Increment attempt counter
    try:
        if "```sql" in content and "```" in content[content.index("```sql")+6:]:
            sql_block_start = content.index("```sql") + 6
            sql_block_end = content.index("```", sql_block_start)
            sql_query = content[sql_block_start:sql_block_end].strip()
            if not sql_query:  # Empty query after stripping
                raise ValueError("Empty SQL query")
            state["messages"] = add_messages(state["messages"], [response])
        else:
            raise ValueError("No SQL query found in markdown block")
    except (ValueError, IndexError) as e:
        # If we can't extract a valid SQL query, treat it as an error
        state["messages"] = add_messages(state["messages"], [AIMessage(content=f"Failed to generate valid SQL query: {str(e)}")])
        state["next_action"] = "compose_query"
        return state

    state["sql_query"] = sql_query
    return state

# Function to execute SQL query
def execute_query(state: State) -> State:
    try:
        # Execute the SQL query
        result = db.run(state["sql_query"])
    except Exception as e:
        # Handle any SQL execution errors
        state["messages"] = add_messages(state["messages"], [AIMessage(content=f"Error executing query: {str(e)}")])
        state["next_action"] = "handle_error"
        return state

    try:
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a flight itinerary assistant. Format these SQL results into a helpful response. If there's something wrong or missing from the results, request a new query by saying ERROR somewhere in your response. Do NOT say ERROR if the results are correct."),
            MessagesPlaceholder(variable_name="messages"),
            ("system", "SQL Query: {sql_query}\nQuery Results: {query_results}")
        ])

        chain = prompt | llm(0.3)
        response = chain.invoke({
            "messages": state["messages"],
            "sql_query": state["sql_query"],
            "query_results": result
        })
    except Exception as e:
        # Handle any LLM errors
        state["messages"] = add_messages(state["messages"], [AIMessage(content=f"Error processing results: {str(e)}")])
        state["next_action"] = "handle_error"
        return state

    # Check if the LLM wants to retry with a different query
    if "ERROR" in response.content:
        state["messages"] = add_messages(state["messages"], [AIMessage(content=response.content)])
        state["next_action"] = "handle_error"
    else:
        state["messages"] = add_messages(state["messages"], [AIMessage(content=result), response])
        state["next_action"] = "end"

    return state

# Function to handle query errors
def handle_query_error(state: State) -> State:
    error_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an SQL expert working with a flights database. The previous query didn't give us what we needed. Explain what went wrong and suggest how to fix it."),
        MessagesPlaceholder(variable_name="messages")
    ])

    chain = error_prompt | llm(0) # Error analysis should be deterministic
    error_response = chain.invoke({"messages": state["messages"]})

    state["messages"] = add_messages(state["messages"], [AIMessage(content=error_response.content)])
    state["next_action"] = "compose_query"
    return state

# Define the workflow
def create_workflow() -> StateGraph:
    # Create workflow
    workflow = StateGraph(state_schema=State)

    # Add nodes
    workflow.add_node("classify", classify_query)

    # Add downstream routes
    workflow.add_node("unrelated", handle_unrelated)
    workflow.add_node("perform_search", perform_search)
    workflow.add_node("compose_query", compose_sql_query)
    workflow.add_node("execute_query", execute_query)
    workflow.add_node("handle_error", handle_query_error)
    workflow.add_node("process_search", process_search_results)
    workflow.add_conditional_edges(
        "classify",
        lambda x: x["query_type"],
        path_map={
            QueryType.UNRELATED: "unrelated",
            QueryType.GENERAL_TRAVEL: "perform_search",
            QueryType.FLIGHT_ITINERARY: "compose_query"
        }
    )

    # Query handling
    workflow.add_edge("compose_query", "execute_query")
    workflow.add_conditional_edges(
        "execute_query",
        lambda x: x["next_action"],
        path_map={
            "handle_error": "handle_error",
            "end": END
        }
    )
    workflow.add_edge("handle_error", "compose_query")

    workflow.add_edge("perform_search", "process_search")
    workflow.add_conditional_edges(
        "process_search",
        lambda x: x["next_action"],
        path_map={
            "search": "perform_search",
            "end": END,
        }
    )

    # Set entry point
    workflow.set_entry_point("classify")

    return workflow

# Create the workflow
app = create_workflow().compile()