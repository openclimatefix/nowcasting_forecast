""" Pydantic and Sqlalchemy models for the database

The following class are made
1. Reusable classes
2. Location objects, where the forecast is for
3. ForecastValue objects, specific values of a forecast and time
4. Input data status, shows when the data was collected
5. Forecasts, a forecast that is made for one gsp, for several time steps into the future

Current these models have a primary index of 'id'.
This keeps things very simple at the start.
But there can be multiple forecasts for similar values.

Later on it would be good to add a forecast latest table, where the latest forecast can be read.
The primary keys could be 'gsp_id' and 'target_datetime_utc'.
"""

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field, validator
from sqlalchemy import Column, DateTime, Float, ForeignKey, Index, Integer, String
from sqlalchemy.orm import declarative_base, relationship

from nowcasting_forecast.database.utils import convert_to_camelcase, datetime_must_have_timezone

Base = declarative_base()


national_gb_label = "National-GB"
# TODO #3 Add forecast latest table, this make it easy to load the latest forecast


########
# 1. Reusable classes
########
class CreatedMixin:
    """Mixin to add created datetime to model"""

    created_utc = Column(DateTime(timezone=True), default=lambda: datetime.utcnow())


class EnhancedBaseModel(BaseModel):
    """Ensures that attribute names are returned in camelCase"""

    # Automatically creates camelcase alias for field names
    # See https://pydantic-docs.helpmanual.io/usage/model_config/#alias-generator
    class Config:  # noqa: D106
        alias_generator = convert_to_camelcase
        allow_population_by_field_name = True
        orm_mode = True


########
# 2. Location
########
class LocationSQL(Base):
    """Location that the forecast is for"""

    __tablename__ = "location"

    id = Column(Integer, primary_key=True)
    label = Column(String)
    gsp_id = Column(Integer)
    gsp_name = Column(String, nullable=True)
    gsp_group = Column(String, nullable=True)
    region_name = Column(String, nullable=True)

    forecast = relationship("ForecastSQL", back_populates="location")


class Location(EnhancedBaseModel):
    """Location that the forecast is for"""

    label: str = Field(..., description="")
    gsp_id: Optional[int] = Field(None, description="The Grid Supply Point (GSP) id", index=True)
    gsp_name: Optional[str] = Field(None, description="The GSP name")
    gsp_group: Optional[str] = Field(None, description="The GSP group name")
    region_name: Optional[str] = Field(None, description="The GSP region name")

    rm_mode = True

    def to_orm(self) -> LocationSQL:
        """Change model to LocationSQL"""
        return LocationSQL(
            label=self.label,
            gsp_id=self.gsp_id,
            gsp_name=self.gsp_name,
            gsp_group=self.gsp_group,
            region_name=self.region_name,
        )


########
# 3. ForecastValue
########
class ForecastValueSQL(Base, CreatedMixin):
    """One Forecast of generation at one timestamp"""

    __tablename__ = "forecast_value"

    id = Column(Integer, primary_key=True)
    target_time = Column(DateTime(timezone=True))
    expected_power_generation_megawatts = Column(Float)

    forecast_id = Column(Integer, ForeignKey("forecast.id"), index=True)
    forecast = relationship("ForecastSQL", back_populates="forecast_values")

    Index("index_forecast_value", CreatedMixin.created_utc.desc())


class ForecastValue(EnhancedBaseModel):
    """One Forecast of generation at one timestamp"""

    target_time: datetime = Field(
        ...,
        description="The target time that the forecast is produced for",
    )
    expected_power_generation_megawatts: float = Field(
        ..., ge=0, description="The forecasted value in MW"
    )

    _normalize_target_time = validator("target_time", allow_reuse=True)(datetime_must_have_timezone)

    def to_orm(self) -> ForecastValueSQL:
        """Change model to ForecastValueSQL"""
        return ForecastValueSQL(
            target_time=self.target_time,
            expected_power_generation_megawatts=self.expected_power_generation_megawatts,
        )


########
# 4. Input data status
########
class InputDataLastUpdatedSQL(Base, CreatedMixin):
    """Information about the input data that was used to create the forecast"""

    __tablename__ = "input_data_last_updated"

    id = Column(Integer, primary_key=True)
    gsp = Column(DateTime(timezone=True))
    nwp = Column(DateTime(timezone=True))
    pv = Column(DateTime(timezone=True))
    satellite = Column(DateTime(timezone=True))

    forecast = relationship("ForecastSQL", back_populates="input_data_last_updated")

    Index("index_input_data", CreatedMixin.created_utc.desc())


class InputDataLastUpdated(EnhancedBaseModel):
    """Information about the input data that was used to create the forecast"""

    gsp: datetime = Field(..., description="The time when the input GSP data was last updated")
    nwp: datetime = Field(..., description="The time when the input NWP data was last updated")
    pv: datetime = Field(..., description="The time when the input PV data was last updated")
    satellite: datetime = Field(
        ..., description="The time when the input satellite data was last updated"
    )

    _normalize_gsp = validator("gsp", allow_reuse=True)(datetime_must_have_timezone)
    _normalize_nwp = validator("nwp", allow_reuse=True)(datetime_must_have_timezone)
    _normalize_pv = validator("pv", allow_reuse=True)(datetime_must_have_timezone)
    _normalize_satellite = validator("satellite", allow_reuse=True)(datetime_must_have_timezone)

    def to_orm(self) -> InputDataLastUpdatedSQL:
        """Change model to InputDataLastUpdatedSQL"""
        return InputDataLastUpdatedSQL(
            gsp=self.gsp,
            nwp=self.nwp,
            pv=self.pv,
            satellite=self.satellite,
        )


########
# 4. Forecasts
########
# TODO add model_name to forecast, or add model table #13
class ForecastSQL(Base, CreatedMixin):
    """Forecast SQL model"""

    __tablename__ = "forecast"

    id = Column(Integer, primary_key=True)
    forecast_creation_time = Column(DateTime(timezone=True))
    model_name = Column(String)

    # many (forecasts) to one (location)
    location = relationship("LocationSQL", back_populates="forecast")
    location_id = Column(Integer, ForeignKey("location.id"), index=True)

    # one (forecasts) to many (forecast_value)
    forecast_values = relationship("ForecastValueSQL", back_populates="forecast")

    # many (forecasts) to one (input_data_last_updated)
    input_data_last_updated = relationship("InputDataLastUpdatedSQL", back_populates="forecast")
    input_data_last_updated_id = Column(
        Integer, ForeignKey("input_data_last_updated.id"), index=True
    )

    Index("index_forecast", CreatedMixin.created_utc.desc())


class Forecast(EnhancedBaseModel):
    """A single Forecast"""

    location: Location = Field(..., description="The location object for this forecaster")
    model_name: str = Field(..., description="The name of the model that made this forecast")
    forecast_creation_time: datetime = Field(
        ..., description="The time when the forecaster was made"
    )
    forecast_values: List[ForecastValue] = Field(
        ...,
        description="List of forecasted value objects. Each value has the datestamp and a value",
    )
    input_data_last_updated: InputDataLastUpdated = Field(
        ...,
        description="Information about the input data that was used to create the forecast",
    )

    _normalize_forecast_creation_time = validator("forecast_creation_time", allow_reuse=True)(
        datetime_must_have_timezone
    )

    def to_orm(self) -> ForecastSQL:
        """Change model to ForecastSQL"""
        return ForecastSQL(
            model_name=self.model_name,
            forecast_creation_time=self.forecast_creation_time,
            location=self.location.to_orm(),
            input_data_last_updated=self.input_data_last_updated.to_orm(),
            forecast_values=[forecast_value.to_orm() for forecast_value in self.forecast_values],
        )


class ManyForecasts(EnhancedBaseModel):
    """Many Forecasts"""

    forecasts: List[Forecast] = Field(
        ...,
        description="List of forecasts for different GSPs",
    )
