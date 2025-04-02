# app.py

import streamlit as st
from tableGeneration import run_pipeline
import pandas as pd
import os

st.set_page_config(page_title="PDF → Table Extractor", layout="wide")
st.title("📄🔁 PDF to Structured Table Extractor")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    # Save the file temporarily
    with open("sample.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.info("⏳ Processing PDF... This may take a moment ⏳")
    state = run_pipeline("sample.pdf")

    st.success("✅ Table extracted!")

    # Display Table
    st.subheader("📊 Final Table")
    df = pd.DataFrame.from_dict(state["merged_table"], orient="index")
    st.dataframe(df)

    # Display QA Results
    st.subheader("🧠 Auto-Generated Questions & Answers")
    if isinstance(state["qa_results"], dict):
        for q, a in state["qa_results"].items():
            st.markdown(f"**Q:** {q}\n\n**A:** {a}")
    else:
        st.markdown(state["qa_results"])

    # Display Errors
    st.subheader("⚠️ Validation / QA Comments")
    for err in state.get("errors", []):
        st.error(err)

    # Download Table
    st.subheader("⬇️ Download Table as Excel")
    excel_filename = "output_table.xlsx"
    df.to_excel(excel_filename)
    with open(excel_filename, "rb") as f:
        st.download_button(
            label="Download Excel File",
            data=f,
            file_name="table_output.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
