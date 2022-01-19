""" Read databa functions """
import logging
from typing import List, Optional

from sqlalchemy.orm.session import Session

from nowcasting_forecast.database.models import (ForecastSQL, ForecastValueSQL,
                                                 LocationSQL)

logger = logging.getLogger(__name__)


def get_latest_forecast(
    session: Session,
    gsp_id: Optional[int] = None,
) -> ForecastSQL:
    """
    Read forecasts

    :param session: database session
    :param gsp_id: optional to gsp id, to filter query on
        If None is given then all are returned.
    :param session: database session

    return: List of forecasts objects from database
    """

    # start main query
    query = session.query(ForecastSQL)

    # filter on gsp_id
    if gsp_id is not None:
        query = query.join(LocationSQL)
        query = query.filter(LocationSQL.gsp_id == gsp_id)

    # this make the newest ones comes to the top
    query.order_by(ForecastSQL.created_utc.desc())

    # get all results
    forecasts = query.first()

    logger.debug(f"Found forecasts for {gsp_id}")

    return forecasts


def get_forecast_values(
    session: Session,
    gsp_id: Optional[int] = None,
) -> List[ForecastValueSQL]:

    # start main query
    query = session.query(ForecastValueSQL)

    # filter on gsp_id
    if gsp_id is not None:
        query = query.join(ForecastSQL)
        query = query.join(LocationSQL)
        query = query.filter(LocationSQL.gsp_id == gsp_id)

    # get all results
    forecasts = query.all()

    return forecasts
