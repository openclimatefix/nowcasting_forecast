""" Sqlalchemy models for the database"""
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# Current these models have a primary index of 'id'.
# This keeps things very simple at the start.
# But there can be multiple forecasts for similar values.

# Later on it would be good to add a forecast latest table, where the latest forecast can be read.
# The primary keys could be 'gsp_id' and 'target_datetime_utc'.

# sqlalchemy models


# TODO add sql mixin for created_utc
class StatisticSQL(Base):
    """Statistic SQL model"""

    __tablename__ = "statistic"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    value = Column(Float)
    child_id = Column(Integer, ForeignKey("forecast.id"))

    forecast = relationship("ForecastSQL", back_populates="statistics")


class ForecastSQL(Base):
    """Forecast SQL model"""

    __tablename__ = "forecast"

    id = Column(Integer, primary_key=True)
    t0_datetime_utc = Column(DateTime)
    target_datetime_utc = Column(DateTime)
    gsp_id = Column(Integer)

    statistics = relationship("StatisticSQL", back_populates="forecast")
