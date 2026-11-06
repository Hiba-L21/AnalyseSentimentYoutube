from pydantic import BaseModel, field_validator
from typing import List
class PredictRequest(BaseModel):
    comments: List[str]

    @field_validator("comments")
    @classmethod
    def validate_comments(cls, v):
        if not v:
            raise ValueError("La liste de commentaires ne peut pas être vide.")
        if len(v) > 500:
            raise ValueError("Maximum 500 commentaires par requête.")
        return [c.strip() for c in v if c.strip()]
class CommentResult(BaseModel):
    comment: str
    label: int          # -1, 0, 1
    sentiment: str      # Négatif, Neutre, Positif
    confidence: float   
class PredictResponse(BaseModel):
    total: int
    results: List[CommentResult]
    stats: dict        
class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str