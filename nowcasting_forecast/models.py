""" Pydantic models """
from pydantic import BaseModel
from datetime import datetime
from typing import List


# pydantic models
class Statistic(BaseModel):
    """ Pydantic Statistic Model"""
    name: str
    value: float
    # unit: str


class Forecast(BaseModel):
    """ Pydantic Forecast Model"""
    t0_datetime_utc: datetime
    target_datetime_utc: datetime
    statistics: List[Statistic]
    gsp_id: int

