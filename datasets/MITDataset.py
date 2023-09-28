from torch.utils.data import Dataset
import os


BASE_PATH = "/datasets/mitbih_database"
ANNOT_ENDING = "annotations.txt"
data_files = [file for file in os.listdir(BASE_PATH) if file.endswith(".csv")]
data_files.sort()
__names__ = [name for name, _ in (os.path.splitext(file) for file in data_files)]  # [100, 101, ...]
__annots__ = [name + ANNOT_ENDING for name in __names__]  # [100annotation.txt, ...]


def segment(name):
    """
    cut ECG into 3-beat segments, centred on beat of interest.
    save the output along with its label as csv.

    -use annot file to get locs of peaks
    -ignore first & last peaks
    -find locs of midpoints between (3n)th and (3n+1)th
    -cut the data file in these locations
    -any label other than 'N'ormal takes priority
    -multilabels allowed: refine implementation at a later stage
    """
    ...


class MITBIHDataset(Dataset):
    """
    OG arrhythmia dataset from MIT-BIH
    """

    def __getitem__(self, item):
        ...

    def __len__(self):
        ...