from typing import TypedDict, Annotated
from langgraph.graph import StateGraph, END
from models import llm  
from langchain_core.messages import HumanMessage, SystemMessage


class TableGenerationState(TypedDict):
    input_text: str  
    identified_headers: Annotated[list, "headers"]  
    generated_table: Annotated[str, "table"]  

# ----------------------
# Step 1: Header Identification Node
# ----------------------
def identify_headers(state: TableGenerationState):
    input_text = state["input_text"]
    
    messages = [
        SystemMessage(content="""You are a data schema expert. Analyze the text and:
1. Identify all potential table headers
2. Distinguish between column headers and row headers
3. Consider both explicit and implicit relationships
4. Return ONLY a comma-separated list of column headers"""),
        
        HumanMessage(content=f"""Analyze this text and extract column headers for tabular representation:
        
        {input_text}
        
        Guidelines:
        - Prefer short, specific header names
        - Use exact terminology from text when possible
        - Avoid generic terms like 'Attribute' or 'Details'
        - Prioritize measurable/quantifiable data
        
        Column headers (comma-separated):""")
    ]
    
    response = llm.invoke(messages)
    headers = [h.strip() for h in response.content.split(",") if h.strip()]
    return {"identified_headers": headers}

# ----------------------
# Step 2: Table Generation Node
# ----------------------
def generate_table(state: TableGenerationState):
    input_text = state["input_text"]
    headers = state["identified_headers"]
    
    messages = [
        SystemMessage(content="""You are a precision data formatter. Create tables with:
1. EXACTLY the provided headers
2. No additional columns
3. Direct value mapping from source text
4. Strict markdown formatting"""),
        
        HumanMessage(content=f"""Text: {input_text}
        
        Required columns: {', '.join(headers)}
        
        Create a markdown table with:
        - Header row: {' | '.join(headers)}
        - Data rows containing ONLY values from the text
        - No empty cells (use N/A if missing)
        - Maintain original units and formatting
        
        Markdown Table:""")
    ]
    
    response = llm.invoke(messages)
    return {"generated_table": response.content}

# ----------------------
# Step 3: Validation Node
# ----------------------
def validate_table(state: TableGenerationState):
    table = state["generated_table"]
    headers = state["identified_headers"]
    
    if not all(header in table for header in headers):
        raise ValueError("Missing headers in generated table")
    
    print(f"\nIdentified Headers: {headers}\n")
    return state


workflow = StateGraph(TableGenerationState)
workflow.add_node("header_identifier", identify_headers)
workflow.add_node("table_generator", generate_table)
workflow.add_node("validator", validate_table)

workflow.set_entry_point("header_identifier")
workflow.add_edge("header_identifier", "table_generator")
workflow.add_edge("table_generator", "validator")
workflow.add_edge("validator", END)

app = workflow.compile()

def generate_table_from_text(text: str):
    results = app.invoke({"input_text": text})  
    return results["generated_table"] 

# ----------------------
# Example Usage for Testing & Markdown → DataFrame Parsing
# ----------------------
from io import StringIO
import pandas as pd

if __name__ == "__main__":
    input_text = """
    Patients with rheumatoid arthritis and osteoarthritis were monitored over a period of four weeks to evaluate the effectiveness of Medorin. During Week 1, rheumatoid arthritis patients experienced a 20% reduction in joint swelling, a mobility improvement of +2 points, and an 18% decrease in inflammatory markers. In the same week, osteoarthritis patients showed a 15% reduction in joint swelling, a +1.5-point mobility improvement, and a 12% decrease in inflammatory markers. By Week 4, the rheumatoid arthritis group saw a 35% reduction in joint swelling, a +3-point increase in mobility, and a 25% drop in inflammatory markers, while osteoarthritis patients achieved a 30% reduction in swelling, a +2.5-point mobility improvement, and a 20% reduction in inflammatory markers.
    """


    md_table = generate_table_from_text(input_text)
    print("Raw Markdown:\n", md_table, "\n")

    lines = md_table.splitlines()
    table_lines = [ln for ln in lines if ln.strip().startswith("|")]
    clean_md = "\n".join(table_lines)

    df = pd.read_csv(
        StringIO(clean_md),
        sep="|",
        header=0,
        skipinitialspace=True
    )
    # Drop any 'Unnamed' columns and strip whitespace
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df.columns = df.columns.str.strip()

    print("Clean DataFrame:\n", df)
