""" Sqlalchemy models for the database"""
from datetime import datetime

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()

# Current these models have a primary index of 'id'.
# This keeps things very simple at the start.
# But there can be multiple forecasts for similar values.

# Later on it would be good to add a forecast latest table, where the latest forecast can be read.
# The primary keys could be 'gsp_id' and 'target_datetime_utc'.

# sqlalchemy models


class CreatedMixin:
    """Mixin to add created datetime to model"""

    created_utc = Column(DateTime(timezone=True), default=lambda: datetime.utcnow())


# TODO add sql mixin for created_utc
class StatisticSQL(Base, CreatedMixin):
    """Statistic SQL model"""

    __tablename__ = "statistic"

    # Temporary making table automatically. Really we want a migration script, perhaps using alembic
    # https: // alembic.sqlalchemy.org / en / latest /
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    name = Column(String)
    value = Column(Float)
    forecast_id = Column(Integer, ForeignKey("forecast.id"))

    forecast = relationship("ForecastSQL", back_populates="statistics")


class ForecastSQL(Base, CreatedMixin):
    """Forecast SQL model"""

    __tablename__ = "forecast"

    # Temporary making table automatically. Really we want a migration script, perhaps using alembic
    # https: // alembic.sqlalchemy.org / en / latest /
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True)
    t0_datetime_utc = Column(DateTime)
    target_datetime_utc = Column(DateTime)
    gsp_id = Column(Integer)

    statistics = relationship("StatisticSQL", back_populates="forecast")
