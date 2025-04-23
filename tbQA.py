# tbQA.py
# LangGraph workflow to generate verification questions for a table extracted from text,
# extract answers from both the table and the original text, and compare them for accuracy.

from typing import TypedDict, List, Annotated, Dict
from langgraph.graph import StateGraph, END
from models import llm  # Pre-configured LLM instance
from langchain_core.messages import HumanMessage, SystemMessage
from tbGen import generate_table_from_text  # Your table generation pipeline

# ----------------------------
# Define the workflow state schema
# ----------------------------
class QAState(TypedDict):
    input_text: str
    # Raw source text used to generate and verify the table

    generated_table: str
    # Markdown table produced by the table-generation pipeline

    questions: Annotated[List[str], "questions"]
    # List of verification questions about table values/trends

    table_answers: Annotated[List[str], "table_answers"]
    # Answers extracted from the table for each question

    text_answers: Annotated[List[str], "text_answers"]
    # Answers extracted from the original text for each question

    validation_results: Annotated[List[bool], "validation"]
    # Boolean list indicating whether each table answer matches the text answer


# ----------------------------
# Step 1: Question Generation Node
# ----------------------------
def generate_questions(state: QAState) -> dict:
    """
    Prompts the LLM to create 3–5 concise verification questions
    focusing on numerical values, comparisons, or trends in the table.
    """
    table = state["generated_table"]

    messages = [
        SystemMessage(content="""
You are a QA analyst. Generate specific questions to verify data accuracy between a table and its source text. Follow these rules:
1. Create questions about numerical values, comparisons, and trends
2. Ask for exact values from specific table cells. 
3. Phrase questions to be answerable in 1-2 words
4. Generate 3-5 critical questions
5. Return each question on a separate line
"""),
        HumanMessage(content=f"""Table to verify:

{table}

Generate verification questions:""")
    ]

    response = llm.invoke(messages)
    # Split on newline and remove any empty lines, cap at 5 questions
    questions = [q.strip() for q in response.content.split("\n") if q.strip()]
    return {"questions": questions[:5]}


# ----------------------------
# Step 2: Table Answer Extraction Node
# ----------------------------
def extract_table_answers(state: QAState) -> dict:
    """
    For each generated question, ask the LLM to answer solely based on the table.
    Ensures answers use exact table values or 'N/A' if missing.
    """
    questions = state["questions"]
    table = state["generated_table"]

    answers: List[str] = []
    for q in questions:
        messages = [
            SystemMessage(content="""
Answer the question based SOLELY on the provided table. Follow these rules:
1. Use EXACT values from the table. 
2. Include units if present unless it includes a numerical value.
3. If information is missing, respond with 'N/A'
4. Keep answers concise (1-3 words)
"""),
            HumanMessage(content=f"""Table:
{table}

Question: {q}
Answer:""")
        ]
        response = llm.invoke(messages)
        answers.append(response.content.strip())

    return {"table_answers": answers}


# ----------------------------
# Step 3: Text Answer Extraction Node
# ----------------------------
def extract_text_answers(state: QAState) -> dict:
    """
    For each question, ask the LLM to answer solely based on the original text.
    Ensures answers match the phrasing/units in the source text.
    """
    questions = state["questions"]
    input_text = state["input_text"]

    answers: List[str] = []
    for q in questions:
        messages = [
            SystemMessage(content="""
Answer the question based SOLELY on the provided text. Follow these rules:
1. Use EXACT values from the text
2. Include units if present
3. If information is missing, respond with 'N/A'
4. Keep answers concise (1-3 words)
"""),
            HumanMessage(content=f"""Text:
{input_text}

Question: {q}
Answer:""")
        ]
        response = llm.invoke(messages)
        answers.append(response.content.strip())

    return {"text_answers": answers}


# ----------------------------
# Step 4: Validation Node
# ----------------------------
def validate_answers(state: QAState) -> dict:
    """
    Compares each table-derived answer with the corresponding text-derived answer.
    Normalizes formatting (case, spaces, punctuation) and returns a list of booleans.
    """
    table_answers = state["table_answers"]
    text_answers = state["text_answers"]

    validation: List[bool] = []
    for t_ans, txt_ans in zip(table_answers, text_answers):
        # Normalize both strings for fair comparison
        normalize = lambda s: (
            s.lower()
             .replace(" ", "")
             .replace(",", "")
             .strip("°%$")
        )
        valid = normalize(t_ans) == normalize(txt_ans)
        validation.append(valid)

    return {"validation_results": validation}


# ----------------------------
# Build and Compile the QA Workflow
# ----------------------------
workflow = StateGraph(QAState)
workflow.add_node("generate_questions", generate_questions)
workflow.add_node("extract_table_answers", extract_table_answers)
workflow.add_node("extract_text_answers", extract_text_answers)
workflow.add_node("validate_answers", validate_answers)

workflow.set_entry_point("generate_questions")
workflow.add_edge("generate_questions", "extract_table_answers")
workflow.add_edge("extract_table_answers", "extract_text_answers")
workflow.add_edge("extract_text_answers", "validate_answers")
workflow.add_edge("validate_answers", END)

qa_app = workflow.compile()


# ----------------------------
# Main QA Function to Run the Pipeline
# ----------------------------
def qa_table_from_text(text: str) -> dict:
    """
    1. Generates a table from text using tbGen.generate_table_from_text
    2. Runs the QA workflow on that table and the same text
    3. Returns the full state, including questions, answers, and validation flags
    """
    # Step A: Create the table
    generated_table = generate_table_from_text(text)

    # Step B: Invoke QA workflow
    results = qa_app.invoke({
        "input_text": text,
        "generated_table": generated_table
    })
    return results


# ----------------------------
# Enhanced Example Usage (for local testing)
# ----------------------------
if __name__ == "__main__":
    sample_text = """
    Patients with rheumatoid arthritis and osteoarthritis were monitored over a period of four weeks to evaluate the effectiveness of Medorin...
    """

    # Run QA pipeline
    results = qa_table_from_text(sample_text)

    # Summarize QA report
    print("\n" + "="*20 + " QA Report " + "="*20)
    total = len(results["questions"])
    correct = sum(results["validation_results"])
    print(f"Questions Generated: {total}")
    print(f"Correct Answers  : {correct}/{total}\n")

    # Print individual Q/A and validation
    for idx, (q, t_ans, txt_ans, valid) in enumerate(zip(
        results["questions"],
        results["table_answers"],
        results["text_answers"],
        results["validation_results"]
    ), start=1):
        status = "✅" if valid else "❌"
        print(f"{idx}. {q}")
        print(f"   Table Answer: {t_ans}")
        print(f"   Text  Answer: {txt_ans}")
        print(f"   Match       : {status}\n")

    # Final assessment
    match_rate = correct / total if total else 0
    if match_rate == 1:
        print("All answers match perfectly! 🎉")
    elif match_rate >= 0.8:
        print("Most answers match. 👍")
    else:
        print("Significant discrepancies found. ⚠️")
