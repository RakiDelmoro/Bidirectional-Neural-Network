import torch
from data_utils import load_data_to_memory
from torch.utils.data import DataLoader

def main():
    WIDTH = 28
    HEIGHT = 28
    BATCH_SIZE = 4096
    LEARNING_RATE = 0.01
    INPUT_FEATURE = WIDTH * HEIGHT

    # Load training data into memory
    image_for_train, expected_for_training, image_for_validation, expected_for_validation = load_data_to_memory('./training-data/mnist.pkl.gz')

    # Dataloaders for network
    train_dataloader = DataLoader(image_for_train, expected_for_training, batch_size=BATCH_SIZE, shuffle=True)
    validation_dataloader = DataLoader(image_for_validation, expected_for_validation, batch_size=BATCH_SIZE, shuffle=True)

main()
