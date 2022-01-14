""" Save forecasts to the database """
from typing import List

from sqlalchemy.orm.session import Session

from nowcasting_forecast.database.models import ForecastSQL, StatisticSQL
from nowcasting_forecast.models import Forecast


def save(forecasts: List[Forecast], session: Session):
    """
    Save forecast to database

    1. Change pydantic object to sqlalchemy objects
    2. Add sqlalchemy onjects to database

    :param forecasts: list of pydantic forecasts
    :param session: database session
    """

    # convert pydantic to sql objects
    forecasts_sql = []
    for forecast in forecasts:
        print(forecast)
        forecast_sql = ForecastSQL(
            t0_datetime_utc=forecast.t0_datetime_utc,
            target_datetime_utc=forecast.target_datetime_utc,
            gsp_id=forecast.gsp_id,
        )
        for stat in forecast.statistics:
            StatisticSQL(name=stat.name, value=stat.value, forecast=forecast_sql)

        forecasts_sql.append(forecast_sql)

    # save objects to database
    session.add_all(forecasts_sql)
    session.commit()
