import os
import numpy as np
import scipy.io as scio
from tonic.dataset import Dataset
from tonic.download_utils import (
    check_integrity,
    download_and_extract_archive,
    extract_archive,
)


class ASLDVS(Dataset):
    """ASL-DVS dataset <https://github.com/PIX2NVS/NVS2Graph>. Events have (txyp) ordering.
    ::

        @inproceedings{bi2019graph,
            title={Graph-based Object Classification for Neuromorphic Vision Sensing},
            author={Bi, Y and Chadha, A and Abbas, A and and Bourtsoulatze, E and Andreopoulos, Y},
            booktitle={2019 IEEE International Conference on Computer Vision (ICCV)},
            year={2019},
            organization={IEEE}
        }

    Parameters:
        save_to (string): Location to save files to on disk.
        download (bool): Choose to download data or verify existing files. If True and a file with the same
                    name and correct hash is already in the directory, download is automatically skipped.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.

    Returns:
        A dataset object that can be indexed or iterated over. One sample returns a tuple of (events, targets).
    """

    url = "https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=1"
    filename = "ASLDVS.zip"
    file_md5 = "20f1dbf961f9a45179f6e489e93c8f2c"

    classes = [chr(letter) for letter in range(97, 123)]  # generate alphabet
    int_classes = dict(zip(classes, range(len(classes))))
    sensor_size = (240, 180, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names

    def __init__(self, save_to, download=True, transform=None, target_transform=None):
        super(ASLDVS, self).__init__(
            save_to, transform=transform, target_transform=target_transform
        )

        self.location_on_system = save_to
        self.folder_name = "asl-dvs"
        self.data = []
        self.targets = []

        self.location_on_system = os.path.join(
            self.location_on_system, self.folder_name
        )
        if not os.path.exists(self.location_on_system):
            os.mkdir(self.location_on_system)

        if download:
            self.download()

        if not check_integrity(
            os.path.join(self.location_on_system, self.filename), self.file_md5
        ):
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.samples = []
        for path, dirs, files in os.walk(self.location_on_system):
            dirs.sort()
            files.sort()
            for file in files:
                if file.endswith("mat"):
                    self.samples.append(path + "/" + file)
                    self.targets.append(self.int_classes[path[-1]])

    def __getitem__(self, index):
        events, target = scio.loadmat(self.samples[index]), self.targets[index]
        data = (
            np.array(
                [
                    events["ts"].T,
                    events["x"].T,
                    self.sensor_size[1] - events["y"].T,
                    events["pol"].T,
                ],
                dtype=self.dtype,
            ),
            self.sensor_size,
        )
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def __len__(self):
        return len(self.samples)

    def download(self):
        download_and_extract_archive(
            self.url, self.location_on_system, filename=self.filename, md5=self.file_md5
        )
        for path, dirs, files in os.walk(self.location_on_system):
            dirs.sort()
            for file in files:
                if file.startswith("Yin") and file.endswith("zip"):
                    extract_archive(os.path.join(self.location_on_system, file))
