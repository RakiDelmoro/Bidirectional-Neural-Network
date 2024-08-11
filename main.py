import torch
from mlp import mlp_network, model_runner
from data_utils import load_data_to_memory
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

def main():
    WIDTH = 28
    HEIGHT = 28
    BATCH_SIZE = 2098
    LEARNING_RATE = 0.01
    INPUT_FEATURE = WIDTH * HEIGHT
    NETWORK_FEATURE_SIZES = [2000, 2000]
    MLP_MODEL, MODEL_PARAMETERS = mlp_network(NETWORK_FEATURE_SIZES, 784, 10, "cuda")
    # Load training data into memory
    image_for_train, expected_for_training, image_for_validation, expected_for_validation = load_data_to_memory('./training-data/mnist.pkl.gz')

    # Dataloaders for network
    train_dataloader = DataLoader(TensorDataset(image_for_train, expected_for_training), batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(TensorDataset(image_for_validation, expected_for_validation), batch_size=BATCH_SIZE, shuffle=True)
    model_runner(MLP_MODEL, train_dataloader, validation_dataloader, 50, torch.nn.CrossEntropyLoss(), torch.optim.AdamW(MODEL_PARAMETERS, lr=LEARNING_RATE))

main()
