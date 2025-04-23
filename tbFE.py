# table_frontend.py
# Streamlit app to generate a table from text and verify its accuracy via QA

import streamlit as st
import pandas as pd
from io import StringIO

# Import the table-generation and QA pipelines
from tbGen import generate_table_from_text
from tbQA import qa_table_from_text

# ----------------------
# Streamlit App Definition
# ----------------------

# Title and instructions
st.title("📊 Text-to-Table Generator & QA Verifier")
st.write(
    """
    Paste or type in any unstructured text containing numerical data, 
    hit **Generate**, and the app will:
    1. Extract a markdown table from your text  
    2. Parse it into a DataFrame  
    3. Generate verification questions  
    4. Compare table answers vs. source-text answers  
    5. Show you which items match  
    """
)

# Text input area
input_text = st.text_area(
    label="Enter text to extract a table:",
    height=200,
    placeholder="Paste your unstructured text here..."
)

# Generate button
if st.button("Generate & Verify"):
    if not input_text.strip():
        st.warning("Please enter some text before generating the table.")
    else:
        with st.spinner("Generating table..."):
            try:
                # --- Step 1: Generate raw output (Markdown or DataFrame) ---
                raw = generate_table_from_text(input_text)

                # --- Step 2: Handle both DataFrame and Markdown string outputs ---
                if isinstance(raw, pd.DataFrame):
                    # If the pipeline returned a DataFrame directly, use it
                    df = raw
                    # Render a Markdown preview for display
                    md_table = df.to_markdown(index=False)
                else:
                    # Otherwise assume it's a Markdown string and parse it
                    md_table = raw
                    lines = md_table.splitlines()
                    table_lines = [ln for ln in lines if ln.strip().startswith("|")]
                    md_content = "\n".join(table_lines)

                    df = pd.read_csv(
                        StringIO(md_content),
                        sep="|",
                        header=0,
                        skipinitialspace=True
                    )
                    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
                    df.columns = df.columns.str.strip()

                # --- Display results ---
                st.subheader("1️⃣ Markdown Table")
                st.code(md_table, language="markdown")

                st.subheader("2️⃣ Parsed DataFrame")
                st.dataframe(df)

                # Download as CSV
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download as CSV",
                    data=csv,
                    file_name="extracted_table.csv",
                    mime="text/csv"
                )

                # --- Step 3: QA-based Verification ---
                with st.spinner("Generating and verifying QA..."):
                    qa_results = qa_table_from_text(input_text)

                questions = qa_results["questions"]
                table_answers = qa_results["table_answers"]
                text_answers = qa_results["text_answers"]
                validation = qa_results["validation_results"]

                qa_df = pd.DataFrame({
                    "Question": questions,
                    "Table Answer": table_answers,
                    "Text Answer": text_answers,
                    "Match": ["✅" if v else "❌" for v in validation]
                })

                st.subheader("3️⃣ QA Verification Results")
                st.table(qa_df)

                total = len(validation)
                correct = sum(validation)
                st.markdown(
                    f"**Overall Accuracy:** {correct}/{total} "
                    f"({(correct/total*100):.1f}% match rate)"
                )
                if correct == total:
                    st.success("All answers match! 🎉")
                elif correct / total >= 0.8:
                    st.info("Most answers match. 👍")
                else:
                    st.error("Significant discrepancies detected. ⚠️")

            except Exception as e:
                st.error(f"Error during table generation or QA: {e}")
