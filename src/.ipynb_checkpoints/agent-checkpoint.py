"""
Hotel Booking AI Data Analyst Agent
-------------------------------------
A read-only AI agent that answers natural language questions
about hotel booking data using SQL and Groq LLM.
"""

import os
import sqlite3
import pandas as pd
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Load API key from .env file
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Connect to the SQLite database
DB_PATH = "data/processed/hotel.db"
def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False)

# Initialize Groq LLM
llm = ChatGroq(
    api_key=GROQ_API_KEY,
    model_name="llama-3.3-70b-versatile"
)

# ─── TOOL 1: Run SQL Query ──────────────────────────────
def run_sql(question: str) -> dict:
    """
    Converts a natural language question into SQL
    and runs it against the hotel bookings database.
    Only SELECT queries are allowed.
    """

    # Prompt that tells LLM to write SQL
    sql_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a SQL expert working with a hotel bookings database.
        
The database has one table called 'bookings' with these key columns:
- hotel (City Hotel or Resort Hotel)
- is_canceled (0 or 1)
- lead_time (days before arrival booking was made)
- arrival_date_month (month name)
- arrival_month_num (month number 1-12)
- market_segment (Online TA, Direct, Groups, Corporate, etc)
- deposit_type (No Deposit, Non Refund, Refundable)
- customer_type (Transient, Contract, Group, Transient-Party)
- adr (average daily rate in euros)
- total_of_special_requests (number of special requests)
- previous_cancellations (number of previous cancellations)
- total_nights (total nights stayed)
- total_guests (total number of guests)
- country (guest country code)
- distribution_channel (TA/TO, Direct, Corporate, GDS)

RULES:
- Only write SELECT queries. Never write INSERT, UPDATE, DELETE or DROP.
- Always use ROUND() for decimal numbers.
- Always use ORDER BY to sort results meaningfully.
- Return ONLY the SQL query, nothing else. No explanation, no backticks."""),
        ("human", "{question}")
    ])

    # Generate SQL from question
    sql_chain = sql_prompt | llm
    sql_response = sql_chain.invoke({"question": question})
    sql_query = sql_response.content.strip()

    # Safety check — block any non-SELECT queries
    if not sql_query.upper().startswith("SELECT"):
        return {
            "error": "Only SELECT queries are allowed.",
            "sql": sql_query,
            "result": None
        }

    # Run the SQL query
    try:
        result_df = pd.read_sql(sql_query, get_connection())
        return {
            "sql": sql_query,
            "result": result_df.to_dict(orient="records"),
            "error": None
        }
    except Exception as e:
        return {
            "sql": sql_query,
            "result": None,
            "error": str(e)
        }


# ─── TOOL 2: Get KPI ────────────────────────────────────
def get_kpi(metric: str, hotel_filter: str = "All") -> dict:
    """
    Returns pre-defined KPI calculations from the dataset.
    Supported metrics: cancellation_rate, avg_adr, total_bookings, avg_lead_time
    """

    where_clause = ""
    if hotel_filter != "All":
        where_clause = f"WHERE hotel = '{hotel_filter}'"

    kpi_queries = {
        "cancellation_rate": f"SELECT ROUND(AVG(is_canceled) * 100, 2) as cancellation_rate FROM bookings {where_clause}",
        "avg_adr": f"SELECT ROUND(AVG(adr), 2) as avg_daily_rate FROM bookings {where_clause} AND is_canceled = 0".replace("WHERE AND", "WHERE"),
        "total_bookings": f"SELECT COUNT(*) as total_bookings FROM bookings {where_clause}",
        "avg_lead_time": f"SELECT ROUND(AVG(lead_time), 0) as avg_lead_time FROM bookings {where_clause}"
    }

    if metric not in kpi_queries:
        return {"error": f"Unknown metric. Choose from: {list(kpi_queries.keys())}"}

    try:
        result = pd.read_sql(kpi_queries[metric], get_connection())
        return {"metric": metric, "result": result.to_dict(orient="records")[0]}
    except Exception as e:
        return {"error": str(e)}


# ─── TOOL 3: Explain Result ─────────────────────────────
def explain_result(question: str, sql_result: list) -> str:
    """
    Takes a SQL result and converts it into a plain English
    business explanation using the Groq LLM.
    """

    explain_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a hotel business analyst. 
Your job is to explain data query results in clear, simple English.
Keep your explanation to 3-4 sentences maximum.
Focus on the business meaning, not the technical details.
Always mention specific numbers from the data."""),
        ("human", """The user asked: {question}
        
The data shows: {data}

Please explain what this means for the hotel business.""")
    ])

    explain_chain = explain_prompt | llm
    response = explain_chain.invoke({
        "question": question,
        "data": str(sql_result)
    })

    return response.content.strip()


# ─── MAIN AGENT FUNCTION ────────────────────────────────
def ask_agent(question: str) -> dict:
    """
    Main function that takes a natural language question,
    runs SQL, and returns a plain English explanation.
    """

    print(f"\nQuestion: {question}")

    # Step 1: Run SQL
    sql_output = run_sql(question)

    if sql_output["error"]:
        return {
            "question": question,
            "sql": sql_output.get("sql", ""),
            "result": None,
            "explanation": f"Error: {sql_output['error']}"
        }

    # Step 2: Explain result
    explanation = explain_result(question, sql_output["result"])

    return {
        "question": question,
        "sql": sql_output["sql"],
        "result": sql_output["result"],
        "explanation": explanation
    }


# ─── TEST THE AGENT ─────────────────────────────────────
if __name__ == "__main__":
    test_questions = [
        "Which market segment has the highest cancellation rate?",
        "Show me the average daily rate by hotel type",
        "Which country sends the most bookings?"
    ]

    for q in test_questions:
        output = ask_agent(q)
        print(f"\nSQL: {output['sql']}")
        print(f"Result: {output['result'][:2]}")  # Show first 2 rows
        print(f"Explanation: {output['explanation']}")
        print("-" * 50)