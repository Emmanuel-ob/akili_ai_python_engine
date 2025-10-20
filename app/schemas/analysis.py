from pydantic import BaseModel
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
    task: str = "generate_business_overview"

class BusinessOverviewResponse(BaseModel):
    success: bool
    overview: str
    discovered_actions: List[Dict[str, Any]]

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