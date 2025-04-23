# 📊 Text-to-Table Generator & QA Verifier

A LangGraph-powered pipeline and Streamlit front-end that transforms unstructured text into a markdown table, parses it into a pandas DataFrame, then automatically generates and verifies QA pairs to check table accuracy against the source text.

---

## 🚀 Features

- **Automated Table Extraction**  
  Use an LLM to identify column headers and generate a one-row or multi-row markdown table from free-form text.

- **DataFrame Parsing & Download**  
  Parse the markdown output into a pandas DataFrame for display, exploration, and CSV download.

- **QA-Based Verification**  
  Generate targeted verification questions, extract answers from both table and text, and flag any mismatches.

- **Interactive Web UI**  
  A Streamlit app to paste text, click a button, and see results end-to-end in your browser.

---

## 💻 Prerequisites

- **Python** ≥ 3.8  
- **pip** or **pip3**  
- An LLM client configured in `models.py` (e.g., Azure OpenAI, Ollama, etc.)

---

## 🛠 Installation

1. **Clone the repo**  
   ```bash
   git clone https://github.com/adrutililly/tableGeneration.git
   cd text-to-table-qa
   ```

2. **(Optional) Create a virtual environment**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

---

## ⚙️ File Structure

```
.
├── tbGen.py            # Core table‐generation LangGraph pipeline
├── tbQA.py             # QA‐generation & verification LangGraph pipeline
├── tbFE.py             # Streamlit front-end (“Text-to-Table & QA Verifier”) 
├── models.py           # LLM client setup (Azure, Ollama, etc.)
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

---

## ▶️ Running the App

```bash
streamlit run tbFE.py
```

Or, if `streamlit` isn’t on your PATH:

```bash
python3 -m streamlit run tbFE.py
```

Then open your browser to `http://localhost:8501`.

---

## 📦 Using the Library

You can also import the pipelines directly:

```python
from tbGen import generate_table_from_text
from tbQA import qa_table_from_text

text = "Your unstructured input here..."
# Step 1: Generate Markdown table or DataFrame
md_table_or_df = generate_table_from_text(text)

# Step 2: Generate & verify QA pairs
qa_results = qa_table_from_text(text)

print("Table (markdown):", md_table_or_df)
print("Questions:", qa_results["questions"])
print("Validation:", qa_results["validation_results"])
```

---

## 🐞 Troubleshooting

- **“DataFrame object has no attribute ‘splitlines’”**  
  Make sure `generate_table_from_text` returns a markdown string. The frontend now auto-detects DataFrames and skips Markdown parsing.

- **Streamlit not found**  
  Ensure you installed Streamlit in the same Python environment you’re using:
  ```bash
  python3 -m pip install streamlit
  ```

---

## 🤝 Contributing

1. Fork the repo & create a feature branch  
2. Commit your changes with clear messages  
3. Open a pull request for review  

---

## 📄 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.
