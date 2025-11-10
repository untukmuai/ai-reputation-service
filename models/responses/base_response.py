from typing import Any, Dict, Optional
from pydantic import BaseModel
from datetime import datetime, timezone


class BaseResponse(BaseModel):
    """Base class for all response models"""
    
    success: bool = True
    message: Optional[str] = None
    data: Optional[Any] = None
    timestamp: str = None
    
    def __init__(self, **data):
        if 'timestamp' not in data or data['timestamp'] is None:
            data['timestamp'] = datetime.now(timezone.utc).isoformat()
        super().__init__(**data)
    
    class Config:
        extra = "forbid"
        validate_assignment = True
        use_enum_values = True


class ErrorResponse(BaseResponse):
    """Error response model"""
    
    success: bool = False
    error_code: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
