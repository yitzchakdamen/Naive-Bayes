from typing import List, Dict, Optional
from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    name: Optional[str]
    columns: List[str]
    columns_all: Dict[str, List[str]]
    target_variable: Optional[List[str]]

class PredictionRequest(BaseModel):
    model_name: str
    input_data: dict
