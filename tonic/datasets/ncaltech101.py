import os
import numpy as np

from tonic.io import read_mnist_file
from tonic.dataset import Dataset
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
    folder_name = "Caltech101"
    annotations_folder_name = "Caltech101_annotations"

    # save_to = /dataset
    # location_on_system = /dataset/NCALTECH101


    sensor_size = None  # all recordings are of different size
    dtype = np.dtype([("x", int), ("y", int), ("t", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, transform=None, target_transform=None):
        super(NCALTECH101, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        # if not self._check_exists():
        #     self.download()

        file_path = os.path.join(self.location_on_system, self.folder_name)
        for path, dirs, files in os.walk(file_path):
            dirs.sort()
            for file in files:
                if file.endswith("bin"):
                    self.data.append(path + "/" + file)
                    print("###Sanad file name: ", path + "/" + file)
                    annotationFileName = 'annotation' + str(file)[5:]
                    # label_number = os.path.basename(path)
                    print("####Sanad annotation name: ", self.annotations_folder_name + "/" + annotationFileName)
                    self.targets.append(self.annotations_folder_name + "/" + annotationFileName)

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
        if self.target_transform is not None:
            target = self.target_transform(target)
        return events, target

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
            return x, y, w, h
