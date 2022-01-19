from nowcasting_forecast.database.fake import make_fake_forecasts
from nowcasting_forecast.database.save import save


def test_save(db_session):
    forecasts = make_fake_forecasts(gsp_ids=range(0, 10))

    save(session=db_session, forecasts=forecasts)
