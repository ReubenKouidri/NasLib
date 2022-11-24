from my_datasets.ArrhythmiaDataset import ArrhythmiaDataset, ArrhythmiaDataset2D
import matplotlib.pyplot as plt

DATA_PATH = "../my_datasets/Train300"
REF_PATH = "../my_datasets/reference300.csv"

d2 = ArrhythmiaDataset2D(data_path=DATA_PATH, reference_path=REF_PATH)

for i in range(5):
    img, tgt = d2[i]
    plt.matshow(img, cmap='Greys')
    plt.show()



