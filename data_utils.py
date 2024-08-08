import gzip
import torch
import random
import pickle

def load_data_to_memory(file_name: str):
    with (gzip.open(file_name, 'rb')) as file:
        ((training_image_array, training_label_array), (validation_image_array, validation_label_array), _) = pickle.load(file, encoding='latin-1')

    return training_image_array, training_label_array, validation_image_array, validation_label_array

