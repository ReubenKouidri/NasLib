import csv
import torch
from ArrhythmiaDataset2D import ArrhythmiaDataset
from torch.utils.data import DataLoader
import numpy as np


def test(record_base_path, model, model_state_dict):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    m = torch.load(model).to(device)  # check
    m.load_state_dict(torch.load(model_state_dict))
    m.eval()
    test_set_path = '/content/gdrive/MyDrive/test_data'
    test_ref_path = '/content/gdrive/MyDrive/REFERENCE.csv'
    test_data = ArrhythmiaDataset(test_set_path, test_ref_path, leads=3, normalize=True,
                                  smoothen=True, wavelet='mexh')
    test_loader = DataLoader(test_data, batch_size=10, shuffle=False, num_workers=1)
    # need to send this to device!
    with open('../../reports/answers.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Recording', 'Result'])
        for batch in test_loader:
            images = batch[0].to(device)
            names = batch[1]
            preds = model(images)
            result = torch.argmax(preds, dim=1) + 1  # check dim!!!!
            for i in range(len(names)):
                answer = [names[i], result[i].item()]
                if answer[1] > 9 or answer[1] < 1 or np.isnan(answer[1]):
                    answer[1] = 1
                writer.writerow(answer)
    csvfile.close()
