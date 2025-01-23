from typing import Dict, List
from pydantic import BaseModel, Field

class TravelPlan(BaseModel):
    flight_details: Dict = Field(description="Flight information including price")
    accommodation: Dict = Field(description="Hotel/accommodation information including price")
    activities: List[str] = Field(description="List of recommended activities")
    total_cost: float = Field(description="Total cost of the trip")
