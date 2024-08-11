import torch
import gzip
import pickle

def prepare_label_data(expected_array):
    new_expected_array = []
    for each in range(expected_array.shape[0]):
        label = expected_array[each]
        expected_tensor = torch.zeros(10, device="cuda")
        expected_tensor[label] = 1
        new_expected_array.append(expected_tensor)

    return torch.stack(new_expected_array, dim=0)

def load_data_to_memory(file_name: str):
    with (gzip.open(file_name, 'rb')) as file:
        ((training_image_array, training_label_array), (validation_image_array, validation_label_array), _) = pickle.load(file, encoding='latin-1')

    return torch.tensor(training_image_array, device="cuda"), prepare_label_data(training_label_array), torch.tensor(validation_image_array, device="cuda"), prepare_label_data(validation_label_array)
