import click


def test_train_inference_models(dimension, mdrnn_size):
    # import tensorflow as tf
    from impsy import mdrnn
    from impsy.utils import mdrnn_config

    click.secho(f"MDRNN size: {mdrnn_size}", fg="blue")
    SEQ_LEN = 50
    # get the params
    model_config = mdrnn_config(mdrnn_size)
    mdrnn_units = model_config["units"]
    mdrnn_layers = model_config["layers"]
    mdrnn_mixes = model_config["mixes"]

    train_mdrnn = mdrnn.PredictiveMusicMDRNN(
        mode=mdrnn.NET_MODE_TRAIN,
        dimension=dimension,
        n_hidden_units=mdrnn_units,
        n_mixtures=mdrnn_mixes,
        sequence_length=SEQ_LEN,
        layers=mdrnn_layers,
    )

    inference_mdrnn = mdrnn.PredictiveMusicMDRNN(
        mode=mdrnn.NET_MODE_RUN,
        dimension=dimension,
        n_hidden_units=mdrnn_units,
        n_mixtures=mdrnn_mixes,
        sequence_length=1,
        layers=mdrnn_layers,
    )

    assert train_mdrnn.model.count_params() == inference_mdrnn.model.count_params()


def test_train_function(
    dimension, dataset_file, dataset_location, models_location, mdrnn_size
):
    import os
    from impsy import train

    assert os.path.isfile(dataset_file)
    batch_size = 1
    epochs = 1

    # Train using that dataset
    train_output = train.train_mdrnn(
        dimension=dimension,
        dataset_location=dataset_location,
        model_size=mdrnn_size,
        early_stopping=False,
        patience=10,
        num_epochs=epochs,
        batch_size=batch_size,
        save_location=models_location,
        save_model=True,
        save_weights=False,
        save_tflite=False,
    )
    click.echo(train_output)
    assert "name" in train_output
    assert "history" in train_output


# def test_get_trained_model(trained_model):
#     assert 'name' in trained_model
#     assert 'history' in trained_model
#     assert 'keras_file' in trained_model
#     assert 'tflite_file' in trained_model
#     assert 'weights_file' in trained_model
