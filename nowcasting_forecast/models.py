from pydantic import BaseModel
from datetime import datetime
from typing import List


# pydantic models
class Statistic(BaseModel):
    name: str
    value: float
    # unit: str


class Forecast(BaseModel):

    t0_datetime_utc: datetime
    target_datetime_utc: datetime
    statistics: List[Statistic]
    gsp_id: int

