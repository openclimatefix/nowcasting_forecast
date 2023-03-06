import json
from datetime import datetime, timedelta

import boto3
import plotly.graph_objects as go
from nowcasting_datamodel.connection import DatabaseConnection
from nowcasting_datamodel.models.base import Base_Forecast
from nowcasting_datamodel.models.gsp import GSPYieldSQL, LocationSQL
from nowcasting_datamodel.models.models import ForecastSQL, ForecastValueSQL

client = boto3.client("secretsmanager")
response = client.get_secret_value(
    SecretId="development/rds/forecast/",
)
secret = json.loads(response["SecretString"])
""" We have used a ssh tunnel to 'localhost' """
db_url = f'postgresql://{secret["username"]}:{secret["password"]}@localhost:5433/{secret["dbname"]}'

creation_utc = "2022-07-22 03:00:00"
gsp_id = 0
creation_utc_init = datetime(2022, 7, 22, 0)


# get forecast from database
forecasts = []
connection = DatabaseConnection(url=db_url, base=Base_Forecast, echo=True)
with connection.get_session() as session:
    # get forecast for 3 hours
    for minutes in range(0, 30 * 2 * 3, 30):
        creation_utc = creation_utc_init + timedelta(minutes=minutes)
        print(f"Getting forecast for {creation_utc}")

        query = session.query(ForecastValueSQL)

        query = query.distinct(ForecastValueSQL.target_time)

        query = query.join(ForecastSQL)
        query = query.join(LocationSQL)
        query = query.join(GSPYieldSQL)

        query = query.where(ForecastValueSQL.created_utc <= creation_utc)
        query = query.where(ForecastValueSQL.target_time >= creation_utc)
        query = query.where(LocationSQL.gsp_id == gsp_id)

        query = query.order_by(ForecastValueSQL.target_time, ForecastValueSQL.created_utc.desc())

        forecast_values = query.all()

        forecasts.append([forecast_values, creation_utc])

with connection.get_session() as session:
    creation_utc = creation_utc_init

    query = session.query(GSPYieldSQL)
    query = query.join(LocationSQL)
    query = query.where(GSPYieldSQL.datetime_utc >= creation_utc)
    query = query.where(LocationSQL.gsp_id == gsp_id)

    query = query.order_by(GSPYieldSQL.datetime_utc)

    estimates = query.all()

traces = []
for forecast_values, creation_utc in forecasts:
    datetimes = [value.target_time for value in forecast_values]
    forecasts = [value.expected_power_generation_megawatts for value in forecast_values]

    traces.append(go.Scatter(x=datetimes, y=forecasts, name=f"forecast ({creation_utc})"))

datetimes_pvlive = [value.datetime_utc for value in estimates]
estimates_pvlive = [value.solar_generation_kw / 1000 for value in estimates]
traces.append(go.Scatter(x=datetimes_pvlive, y=estimates_pvlive, name="pvlive"))

# Plot


fig = go.Figure(data=traces)
fig.show(renderer="browser")
