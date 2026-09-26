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


def test_dataset_dimension(dimension, dataset_file):
    from impsy import train

    assert train.dataset_dimension(dataset_file) == dimension


def test_find_dataset_file(tmp_path):
    import pytest
    from impsy import train

    npz = tmp_path / "some-data.npz"
    assert train.find_dataset_file(npz) == npz
    assert train.find_dataset_file(tmp_path, 4) == tmp_path / "training-dataset-4d.npz"
    with pytest.raises(ValueError):
        train.find_dataset_file(tmp_path)  # no .npz files
    npz.write_bytes(b"")
    assert train.find_dataset_file(tmp_path) == npz
    (tmp_path / "other.npz").write_bytes(b"")
    with pytest.raises(ValueError):
        train.find_dataset_file(tmp_path)  # two .npz files


def test_train_rejects_wrong_dimension(dimension, dataset_file, tmp_path):
    import pytest
    from impsy import train

    with pytest.raises(ValueError, match="dimension"):
        train.train_mdrnn(
            dimension=dimension + 1,
            dataset_location=dataset_file,
            model_size="xxs",
            early_stopping=False,
            patience=10,
            num_epochs=1,
            batch_size=1,
            save_location=tmp_path,
        )


def test_train_stopped_early_still_saves_tflite(dimension, dataset_file, tmp_path):
    """A callback that stops training (e.g., the web UI's stop button) still gets a model."""
    import tensorflow as tf
    from impsy import train

    class StopAfterOneBatch(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            self.model.stop_training = True

    output = train.train_mdrnn(
        dimension=None,
        dataset_location=dataset_file,
        model_size="xxs",
        early_stopping=False,
        patience=10,
        num_epochs=50,
        batch_size=1,
        save_location=tmp_path,
        callbacks=[StopAfterOneBatch()],
    )
    assert len(output["history"].history["loss"]) == 1
    assert output["tflite_file"].exists()
    assert output["keras_file"].exists()


def test_train_interrupted_still_saves_tflite(dimension, dataset_file, tmp_path):
    """Ctrl-C during training saves the model with the latest weights."""
    import tensorflow as tf
    from impsy import train

    class Interrupt(tf.keras.callbacks.Callback):
        def on_train_batch_end(self, batch, logs=None):
            raise KeyboardInterrupt

    output = train.train_mdrnn(
        dimension=None,
        dataset_location=dataset_file,
        model_size="xxs",
        early_stopping=False,
        patience=10,
        num_epochs=50,
        batch_size=1,
        save_location=tmp_path,
        callbacks=[Interrupt()],
    )
    assert output["tflite_file"].exists()


def test_train_command_with_dataset_argument(dimension, dataset_file, tmp_path):
    from click.testing import CliRunner
    from impsy import train

    result = CliRunner().invoke(
        train.train,
        [str(dataset_file), "-M", "xxs", "-N", "1", "-B", "1", "-O", str(tmp_path)],
    )
    assert result.exit_code == 0, result.output
    assert list(tmp_path.glob(f"*dim{dimension}*.tflite"))
