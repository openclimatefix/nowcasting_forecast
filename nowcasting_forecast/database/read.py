from nowcasting_forecast.database.models import ForecastSQL, StatisticSQL
from sqlalchemy.orm.session import Session
from sqlalchemy.orm import joinedload, subqueryload, contains_eager
from typing import List, Union, Optional
from datetime import datetime

import logging

logger = logging.getLogger(__name__)


def get_forecast(
    session: Session,
    gsp_id: Optional[int] = None,
    t0_datetime_utc: Optional[datetime] = None,
    target_datetime_utc: Optional[datetime] = None,
    statistic_names: Optional[Union[List[str], str]] = None,
) -> List[ForecastSQL]:

    if statistic_names is None:
        statistic_names = "all"

    distinct_columns = [ForecastSQL.gsp_id, ForecastSQL.t0_datetime_utc, ForecastSQL.target_datetime_utc]
    order_by_cols = distinct_columns + [ForecastSQL.created_utc.desc()]

    # start main query
    query = session.query(ForecastSQL)

    # select distinct on columns, so only lates forecast is read
    query = query.distinct(*distinct_columns)

    # filter on gsp_id
    if gsp_id is not None:
        query = query.filter(ForecastSQL.gsp_id == gsp_id)

    # filter on gsp_id
    if t0_datetime_utc is not None:
        query = query.filter(ForecastSQL.t0_datetime_utc == t0_datetime_utc)

    # filter on gsp_id
    if target_datetime_utc is not None:
        query = query.filter(ForecastSQL.target_datetime_utc == target_datetime_utc)

    if statistic_names != "all":
        # join with statistic table. We use outer join so that ....
        query = query.outerjoin(StatisticSQL)

        # only select statistics that we want
        query = query.filter(StatisticSQL.name.in_(statistic_names))

        # this makes sure the child table of statistics is filtered as we want it
        query = query.options(contains_eager(ForecastSQL.statistics))

    # order, this makes sure only the most recent forecast is collected
    query = query.order_by(*order_by_cols)

    # get all results
    forecasts = query.all()

    # forecasts = session.query(ForecastSQL.gsp_id).distinct().all()

    logger.debug(f"Found forecasts for {gsp_id}")

    return forecasts
