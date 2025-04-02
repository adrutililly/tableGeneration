# tableGeneration.py

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage
from langchain_ollama import OllamaLLM  # not langchain_community
from pprint import pprint
import pandas as pd

llm = OllamaLLM(model="mistral")  # or "llama3"

from dotenv import load_dotenv
load_dotenv()

# STEP 1: Load and combine PDF text
def load_pdf(path):
    loader = PyPDFLoader(path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)
    documents = splitter.split_documents(pages)
    return " ".join([doc.page_content for doc in documents])

# STEP 2: Check if table is needed
def table_needed(state):
    input_text = state["input_text"]
    prompt = f"""
Given the following passage, check if a table is needed.
Conditions:
1. More than 20 numerical values.
2. At least 3 structured sentences.

Passage:
{input_text}

Answer "yes" or "no".
"""
    result = llm(prompt).strip().lower()


    return {
        "table_needed": result == "yes",
        "output_text": input_text if result == "yes" else None
    }

# STEP 3: Segment the text
def segment_text(state):
    if not state["table_needed"]:
        return {"segments": []}
    prompt = f"""
Your task is to divide a passage into smaller passages grouped by similar facts.
Keep related numerical values and categorical data within the same section.
Separate sections using “__NEW_PASSAGE__".

Passage: {state["output_text"]}
"""
    output = llm(prompt).strip().lower()
    segments = output.split("__NEW_PASSAGE__")
    return {"segments": [s.strip() for s in segments if s.strip()]}

# STEP 4: Extract table headers
def extract_headers(state):
    segments = " ".join(state["segments"])
    
    # Step 1: Ask for headers
    prompt = f"""
Your task is to extract potential row and column headers from the passage below.

🧠 Think about the context:
- If the passage compares **time periods** (e.g., Q1, Jan, 2023), treat them as COLUMN headers.
- If the passage compares **groups or cohorts** (e.g., Group A, Control, Male/Female), treat those as COLUMN headers.
- Use metric names (e.g., Blood Pressure, Revenue, Satisfaction Score) as ROW headers.
- Avoid guessing. If the layout isn't clear, do your best based on typical structure used in reports.

Return your output in this format:
Row Headers: <comma-separated>
Column Headers: <comma-separated>

Passage:
{segments}
"""

    output = llm(prompt).strip()

    # Step 2: Parse LLM response
    lines = [line.strip() for line in output.split("\n") if line.strip()]
    row_headers, column_headers = [], []

    for line in lines:
        if "row headers" in line.lower():
            row_headers = [r.strip() for r in line.split(":", 1)[1].split(",")]
        elif "column headers" in line.lower():
            column_headers = [c.strip() for c in line.split(":", 1)[1].split(",")]

    # Step 3: Check for reversal
    confirm_prompt = f"""
You provided the following headers:

Row Headers: {row_headers}
Column Headers: {column_headers}

Are these correct, or are they reversed? Reply only:
- "correct"
- "reversed"
"""
    confirmation = llm(confirm_prompt).strip().lower()
    if "reversed" in confirmation:
        row_headers, column_headers = column_headers, row_headers

    return {
        "row_headers": row_headers,
        "column_headers": column_headers
    }


# STEP 5: Generate table content
def generate_table_content(state):
    table = {
        row: {col: "NA" for col in state["column_headers"]}
        for row in state["row_headers"]
    }

    segments = " ".join(state["segments"])
    for row in state["row_headers"]:
        for col in state["column_headers"]:
            prompt = f"""
Your task is to extract the value for "{row}" under the column "{col}" from the passage below.
Each cell should contain a single value or "NA" if not found. No extra context needed.

Passage: {segments}
"""
            value = llm(prompt).strip().lower()
            table[row][col] = value
    return {"table": table}

# STEP 6: Local structure check
def local_structure_check(state):
    table = state["table"]
    input_table = pd.DataFrame.from_dict(table, orient="index").to_markdown()

    prompt = f"""
Your task is to verify if the extracted values match their respective column categories.

**Check the following:**
1. Are all numerical values placed in their correct column?
2. Are there any misclassifications (e.g., "66 years" appearing under Revenue)?

**Table to check:**
{input_table}
"""
    result = llm(prompt).strip().lower()
    return {"validated_table": table, "errors": [result]}


# STEP 7: Global structure check
def global_structure_check(state):
    table = state["validated_table"]
    input_table = pd.DataFrame.from_dict(table, orient="index").to_markdown()

    prompt = f"""
Your task is to ensure the table structure is correct.

**Checks:**
1. Do all rows have the same number of columns?
2. Are there missing or duplicate headers?
3. Do values align with headers?

Table:
{input_table}
"""
    result = llm(prompt).strip().lower()
    return {"final_table": table, "errors": state["errors"] + [result]}


# STEP 8: Factual accuracy check (with pseudo-citations)
def factual_accuracy_check(state):
    table = state["final_table"]
    passage = state["output_text"]
    input_table = pd.DataFrame.from_dict(table, orient="index").to_markdown()

    prompt = f"""
Your task is to verify the factual accuracy of each value in the table below,
based on the passage.

**Instructions:**
- For each value, indicate the sentence number where it was found.
- If unverifiable, mark [NA].

**Passage:**
{passage}

**Table:**
{input_table}
"""
    result = llm(prompt).strip().lower()
    return {
        "verified_table": table,
        "citations": result,
        "errors": state["errors"]
    }


# STEP 9: Merge (if multiple tables - dummy for now)
def merge_tables(state):
    # We'll assume just one table for now, but simulate merging logic
    tables = [state["verified_table"]]  # You could add more here
    merged_markdowns = [
        pd.DataFrame.from_dict(tbl, orient="index").to_markdown()
        for tbl in tables
    ]
    table_text = "\n\n---\n\n".join(merged_markdowns)

    prompt = f"""
You are given multiple partial tables below. Your task is to merge them into a single, coherent table.

**Instructions:**
- Use shared headers to align columns.
- Combine rows where categories match.
- Ensure consistency across rows and columns.

**Tables:**
{table_text}

Return the merged table in Markdown format.
"""
    result = llm(prompt).strip().lower()
    return {
        "merged_table_markdown": result,
        "merged_table": state["verified_table"]  # Optionally keep structured one
    }


# STEP 10: Auto-QA generation
def auto_qa_validation(state):
    table = state["verified_table"]
    input_table = pd.DataFrame.from_dict(table, orient="index").to_markdown()

    prompt = f"""
Based on the table below, generate 5 question-answer pairs that test if the table is informative.

**Table:**
{input_table}

Only return the Q&A pairs.
"""
    result = llm(prompt).strip().lower()
    return {"qa_results": result}


# RUN FULL PIPELINE
def run_pipeline(pdf_path):
    state = {"input_text": load_pdf(pdf_path)}
    steps = [
        table_needed,
        segment_text,
        extract_headers,
        generate_table_content,
        local_structure_check,
        global_structure_check,
        factual_accuracy_check,
        merge_tables,
        auto_qa_validation
    ]
    for step in steps:
        state.update(step(state))
    return state

# MAIN
if __name__ == "__main__":
    final_state = run_pipeline("sample.pdf")  # <- change to your actual PDF
    print("\n🔹 Final Table:")
    pprint(final_state["merged_table"])
    print("\n🔸 Auto QA Results:")
    pprint(final_state["qa_results"])
    print("\n⚠️ Errors (if any):")
    pprint(final_state.get("errors", []))

    # Optional: Export table
    df = pd.DataFrame.from_dict(final_state["merged_table"], orient="index")
    df.to_excel("generated_table.xlsx")
    print("\n✅ Table exported to 'generated_table.xlsx'")
