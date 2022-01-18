from nowcasting_forecast.database.save import save
from nowcasting_forecast.database.fake import make_fake_forecasts


def test_save(db_session):
    forecasts = make_fake_forecasts(session=db_session)

    save(session=db_session, forecasts=forecasts)
