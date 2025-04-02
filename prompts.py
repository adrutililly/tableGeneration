table_decision = """Please evaluate the following passage and determine whether a table is needed based on these criteria:
1. The passage contains more than 20 numerical values.
2. The passage includes at least 3 structured sentences.

Passage:
{input_text}

Return:
- `True` if a table is needed.
- `False` if a table is not needed.
"""

segment_passages = """Your task is to split the following passage into a list of smaller, coherent segments. Each segment should group together similar facts and ensure that related numerical values and categorical data are kept together.

Input Passage:
{input_text}

Output:
Return a JSON object with a field "passages", which is an array of strings. Each string should represent one segment of the passage.
"""

header_extraction = """Please analyze the passage below to extract potential row and column headers. Use the following guidelines:

- **Column Headers:**  
  - Identify time periods (e.g., "01", "Jan", "2023") as column headers.
  - Identify groups or cohorts (e.g., "Group A", "Control", "Male/Female") as column headers.

- **Row Headers:**  
  - Identify metric names (e.g., "Blood Pressure", "Revenue", "Satisfaction Score") as row headers.

- **General Instructions:**  
  - If the layout is unclear, follow typical report structures and avoid making guesses.

Return your output in JSON format as follows:
{{
  "row_headers": [List of strings],
  "column_headers": [List of strings]
}}

Passage:
{segment}
"""


extracted_table= """You are given a text passage along with specific row headers and column headers. Your task is to extract the value for each row-column pair from the passage. 

If a value for a given (row, column) pair is not found in the passage, return "NA" for that entry.

Return the final output as a JSON list of dictionaries, where each dictionary corresponds to a row. The structure must be:

[
  {{
    "row": "<row header>",
    "columns": {{
      "<column1>": "<extracted value or 'NA'>",
      "<column2>": "<extracted value or 'NA'>",
      ...
    }}
  }},
  ...
]

No extra text or explanation—only this JSON structure.

---
**Passage:** {passage}

**Rows:** {rows}

**Columns:** {columns}"""


merge_tables = """You are given multiple partial tables, each with potentially overlapping or distinct column headers. Your goal is to merge these tables into one coherent table by aligning shared headers and combining rows where categories match.

Follow these instructions carefully:
1. Identify common columns across all partial tables and merge them into a unified set of headers.
2. If rows share the same category or identifier, combine them into a single row.
3. Ensure consistency in the final table—each row should appear only once, and each column should be represented only once in the unified header.
4. For columns missing values in certain rows, use "N/A" (or an appropriate placeholder).
5. Return the final merged table as a JSON object with the following structure:

{
  "data": [
    {
      "row_name": "<row identifier>",
      "columns": {
        "<column1>": "<value or 'N/A'>",
        "<column2>": "<value or 'N/A'>",
        ...
      }
    },
    ...
  ]
}

No extra text or explanations—only provide the JSON output.

---
**Partial Tables:**
{tables_text}
"""

simple_table = """You are given a text passage and a list of headers. Your task is to extract all relevant values from the passage for each header. 

- Return the output as a JSON array containing exactly one object.
- Within this object, each key should match one of the provided headers.
- The value for each key must be a list of extracted values found in the passage.
- If no values are found for a given header, the list should contain "NA".

Your response should include no extra text or explanation—only the JSON structure.

Example structure:
[
  {{
    "<header1>": ["<value1>", "<value2>", ...],
    "<header2>": ["NA"],
    ...
  }}
]

---
**Passage:** {passage}

**Headers:** {headers}
"""