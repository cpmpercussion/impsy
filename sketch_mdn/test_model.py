from . import model
from . import sample_data


def train_epochs(num_epochs=1):
    print("Training Mixture RNN for", num_epochs, "epochs")
    net = model.MixtureRNN(mode=model.NET_MODE_TRAIN, n_hidden_units=128, n_mixtures=10, batch_size=100, sequence_length=120)
    x_t_log = sample_data.generate_data()
    loader = sample_data.SequenceDataLoader(num_steps=121, batch_size=100, corpus=x_t_log)
    losses = net.train(loader, num_epochs, saving=True)
    print("Training Done.")
    print("Mean Losses per Batch:")
    print(losses)


if __name__ == "__main__":
    train_epochs(30)
