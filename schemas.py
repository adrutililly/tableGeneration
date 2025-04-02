from pydantic import BaseModel, Field 
from typing import List, Dict, Optional 


class TableDecision(BaseModel):
    needs_table: bool

class PassageSegments(BaseModel):
    passages: List[str]




class HeaderExtractionOutput(BaseModel):
    row_headers: List[str] = Field(
        ...,
        description="A list of potential row headers extracted from the passage, typically representing metric names."
    )
    column_headers: List[str] = Field(
        ...,
        description="A list of potential column headers extracted from the passage, representing time periods or groups/cohorts."
    )



class RowData(BaseModel):
    """
    Represents a single row with its extracted values for each column.
    """
    row: Optional[str] = Field(
        ...,
        description="The row header or identifier."
    )
    columns: Optional[Dict[str, str]] = Field(
        ...,
        description="A mapping of each column header to its extracted value or 'NA' if not found."
    )

class ExtractionResult(BaseModel):
    """
    A container for the entire extraction result.
    """
    data: List[RowData] = Field(
        ...,
        description="A list of row data objects. Each object contains the row name and a dictionary of column-value pairs."
    )



class HeaderValues(BaseModel):
    """
    Represents a single JSON object where:
      - Each key is one of the provided headers.
      - The value for each key is a list of extracted strings (or ["NA"] if none found).
    """
    row_data: Optional[Dict[str, List[str]]] = Field(
        ...,
        description="Dictionary of header-value pairs, where values are lists of extracted strings or ['NA'] if no value is found."
    )

class ExtractionResultSimple(BaseModel):
    """
    A container for the entire extraction result.
    """
    data: List[HeaderValues] = Field(
        ...,
        description="A list containing exactly one HeaderValues object."
    )





class MergedRow(BaseModel):
    """
    Represents a single merged row in the final table.
    The 'row_name' attribute identifies the row.
    The 'columns' dictionary maps column headers to their corresponding values.
    """
    row_name: str = Field(
        ...,
        description="A unique identifier or category name for the row."
    )
    columns: Dict[str, str] = Field(
        ...,
        description="A mapping of column headers to their merged values (or 'N/A')."
    )

class MergedTable(BaseModel):
    """
    Represents the final merged table as a list of rows.
    """
    data: List[MergedRow] = Field(
        ...,
        description="A list of merged rows, each with a row name and column values."
    )





class QAPair(BaseModel):
    """
    Represents a single question-and-answer pair.
    """
    question: str = Field(
        ...,
        description="A concise question referencing the table."
    )
    answer: str = Field(
        ...,
        description="A factual answer derived solely from the table."
    )

class QAPairs(BaseModel):
    """
    Represents a collection of question-and-answer pairs.
    """
    data: List[QAPair] = Field(
        ...,
        description="A list of exactly 5 Q&A pairs referencing the table."
    )
