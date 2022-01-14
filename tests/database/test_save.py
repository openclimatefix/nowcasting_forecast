from nowcasting_forecast.database.save import save


def test_save(db_session, forecasts):

    save(session=db_session, forecasts=forecasts)

