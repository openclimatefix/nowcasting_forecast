from nowcasting_forecast.models.cnn.dataloader import get_cnn_data_loader


def test_get_cnn_data_loader():

    dataloader = get_cnn_data_loader()
    assert len(dataloader) == 11
