import os
import sys
import pickle
from PIL import Image
import numpy as np


def unpickle(file):
    with open(file, 'rb') as fo:
        dict_ = pickle.load(fo, encoding='bytes')
    return dict_


def create_dataset(foldername, filename):
    path = os.path.join(foldername, filename)
    dataset_details = unpickle(path)
    for key, val in dataset_details.items():
        print(key)

    try:
        os.mkdir(dataset_details[b'batch_label'])
    except OSError as error:
        print(error)

    for image_name, image in zip(dataset_details[b'filenames'], dataset_details[b'data']):
        im = Image.fromarray(np.transpose(np.reshape(image,(3, 32,32)), (1,2,0)))
        im.save(os.path.join(dataset_details[b'batch_label'].decode("utf-8"), image_name.decode("utf-8")))

    with open(os.path.join(dataset_details[b'batch_label'].decode("utf-8"), "labels.txt"), "w") as f:
        for label in dataset_details[b'labels']:
            f.write(str(label) + "\n")


if __name__ == "__main__":
    folder_name = sys.argv[1]
    for i in range(2, len(sys.argv)):
        file_name = sys.argv[i]
        create_dataset(folder_name, file_name)
