from pydantic import BaseModel, field_validator
from typing import List, Dict, Any, Optional


class ColumnInfo(BaseModel):
    name: str
    type: str
    nullable: bool = True
    key: str = ""
    default: Optional[Any] = None


class SchemaAnalysisRequest(BaseModel):
    table: str
    columns: List[str]
    all_tables: List[str]
    task: str = "analyze_schema"


class RelationshipInfo(BaseModel):
    from_table: str
    to_table: str
    foreign_key: str
    relationship_type: str  # one_to_many, many_to_one, etc.


class TableAnalysis(BaseModel):
    primary_key: str
    foreign_keys: List[Dict[str, str]]
    customer_identifier: Optional[str]
    entity_type: str
    relationships: List[Dict[str, Any]]


class SchemaAnalysisResponse(BaseModel):
    success: bool
    analysis: TableAnalysis


class BusinessOverviewRequest(BaseModel):
    schema_analysis: Dict[str, Any]
    database_type: str
    table_count: int
    selected_tables: List[str]  # NEW: Explicit list of selected tables
    task: str = "generate_business_overview"

    @field_validator("selected_tables")
    @classmethod
    def validate_selected_tables(cls, v):
        if not v or len(v) == 0:
            raise ValueError("selected_tables cannot be empty")
        return v


class BusinessOverviewResponse(BaseModel):
    success: bool
    overview: str
    discovered_actions: List[Dict[str, Any]]
    analyzed_tables: List[str]  # NEW: Tables that were analyzed

    @field_validator("discovered_actions")
    @classmethod
    def validate_actions_tables(cls, v, info):
        """Ensure all actions reference valid tables"""
        if "analyzed_tables" in info.data:
            valid_tables_lower = [t.lower() for t in info.data["analyzed_tables"]]

            for action in v:
                tables_needed = action.get("tables_needed", [])
                invalid_tables = [
                    t for t in tables_needed if t.lower() not in valid_tables_lower
                ]

                if invalid_tables:
                    raise ValueError(
                        f"Action '{action.get('key')}' references invalid tables: {invalid_tables}"
                    )

        return v


class QueryGenerationRequest(BaseModel):
    user_question: str
    schema: Dict[str, Any]
    customer_id: Optional[str]
    connection_type: str  # mysql, postgresql, mongodb
    is_authenticated: bool = False


class QueryGenerationResponse(BaseModel):
    success: bool
    query: str
    explanation: str
    tables_used: List[str]
    requires_auth: bool
    safe: bool
