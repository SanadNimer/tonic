import os
import numpy as np
import torch

from tonic.io import read_mnist_file
from tonic.dataset import Dataset
import tonic.transforms as transforms
# from tonic.download_utils import extract_archive
import array


class NCALTECH101(Dataset):
    """`N-CALTECH101 <https://www.garrickorchard.com/datasets/n-caltech101>`_

    Events have (xytp) ordering.
    ::

        @article{orchard2015converting,
          title={Converting static image datasets to spiking neuromorphic datasets using saccades},
          author={Orchard, Garrick and Jayawant, Ajinkya and Cohen, Gregory K and Thakor, Nitish},
          journal={Frontiers in neuroscience},
          volume={9},
          pages={437},
          year={2015},
          publisher={Frontiers}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
    """

    # url = "https://data.mendeley.com/public-files/datasets/cy6cvx3ryv/files/36b5c52a-b49d-4853-addb-a836a8883e49/file_downloaded"
    # filename = "N-Caltech101-archive.zip"
    # file_md5 = "66201824eabb0239c7ab992480b50ba3"
    # data_filename = "Caltech101.zip"

    sensor_size = None  # all recordings are of different size
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, transform=None, target_transform=None, images_folder="images",
                 annotations_folder="annotations"):
        super(NCALTECH101, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        self.folder_name = images_folder
        self.annotations_folder_name = annotations_folder

        file_path = os.path.join(save_to, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            files.sort()
            for file in files:
                if file.endswith("bin"):
                    self.data.append(path + "/" + file)
                    # label_number = os.path.basename(path)
                    annotationFileName = 'annotation' + str(file)[5:]
                    self.targets.append(save_to + "/" + self.annotations_folder_name + "/" + annotationFileName)

    def __getitem__(self, index):
        """
        Returns:
            a tuple of (events, target) where target is the index of the target class.
        """
        events = read_mnist_file(self.data[index], dtype=self.dtype)
        target = self.read_annotation(self.targets[index])
        events["x"] -= events["x"].min()
        events["y"] -= events["y"].min()
        if self.transform is not None:
            events = self.transform(events)
        transformation = transforms.ToFrame(n_time_bins=5)
        events_transformed = transformation(events)
        print("###Sanad type of events_transformed: ", type(events_transformed))
        print("###Sanad type of events_transformed[0]: ", type(events_transformed[0]))
        print("###Sanad events_transformed: ", events_transformed)
        print("_____________________")
        if self.target_transform is not None:
            target = self.target_transform(target)
        print("index: ", index)
        print("filename: ", self.data[index])
        print("annotation name: ", self.targets[index])
        print("_____________________")
        print("###Sanad type of events: ", type(events[0]))
        print("###Sanad shape of events: ", np.shape(events))
        print("_____________________")
        print("###Sanad type of target: ", type(target[0]))
        print("###Sanad shape of target: ", np.shape(target))
        print("_____________________")
        print("###Sanad sample event: ", events)
        print("###Sanad sample target: ", target)
        return torch.tensor(events.astype(np.int32)), torch.tensor(target.astype(np.int32))

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return True
        # return (
        #     self._is_file_present()
        #     and self._folder_contains_at_least_n_files_of_type(800, ".bin")
        # )

    def read_annotation(self, annotationFileName):
        arr = array.array('h')
        with open(annotationFileName, 'rb') as file:
            arr.fromfile(file, 12)
            (x, y) = arr[2], arr[3]  # coordinates of top left corner
            w, h = arr[4] - arr[2], arr[7] - arr[3]
            return np.array([x, y, w, h])
