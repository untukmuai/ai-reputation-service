from typing import Any, Dict, Optional
from pydantic import BaseModel
from robyn.types import Body


class BaseRequest(Body):
    """Base class for all request models"""
    
    class Config:
        extra = "forbid"  # Prevent extra fields
        validate_assignment = True  # Validate on assignment
        use_enum_values = True  # Use enum values instead of enum objects