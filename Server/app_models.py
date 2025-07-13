from typing import List, Dict, Optional
from pydantic import BaseModel


class ModelInfoResponse(BaseModel):
    name: Optional[str]
    columns_all: List[Dict[str, List[str]]]
    target_variable: Optional[List[str]]
    columns:list
    
class PredictionRequest(BaseModel):
    model_name: str
    input_data: dict

