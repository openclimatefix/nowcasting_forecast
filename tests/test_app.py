
from click.testing import CliRunner
from nowcasting_forecast.app import run
from nowcasting_forecast.database.connection import DatabaseConnection
from nowcasting_forecast.database.models import ForecastSQL


def test_fake(db_connection: DatabaseConnection):
    from click.testing import CliRunner

    runner = CliRunner()
    response = runner.invoke(run, ["--db-url", db_connection.url, "--fake", "true"])
    assert response.exit_code == 0

    with db_connection.get_session() as session:
        forecasts = session.query(ForecastSQL).all()
        assert len(forecasts) == 338 * 4
