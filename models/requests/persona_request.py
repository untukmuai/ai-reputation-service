from typing import Any, Dict, Optional, List
from pydantic import BaseModel
from robyn.types import Body

class RequestSortingHat(BaseModel, Body):
    digital_dna: List[str]
    old_persona: Optional[str]
    old_tier: Optional[int]