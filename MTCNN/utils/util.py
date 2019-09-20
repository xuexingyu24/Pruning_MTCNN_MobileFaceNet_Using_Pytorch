import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

class ListDataset(Dataset):
    def __init__(self, list_path):
        with open(list_path, 'r') as file:
            self.img_files = file.readlines()

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, index):

        annotation = self.img_files[index % len(self.img_files)].strip().split(' ')
        #---------
        #  Image
        #---------
        img = cv2.imread(annotation[0])
        img = img[:,:,::-1]
        img = np.asarray(img, 'float32')
        img = img.transpose((2, 0, 1))
        img = (img - 127.5) * 0.0078125
        input_img = torch.FloatTensor(img)
        #---------
        #  Label
        #---------
        label = int(annotation[1])
        bbox_target = np.zeros((4,))
        landmark = np.zeros((10,))

        if len(annotation[2:]) == 4:
            bbox_target = np.array(annotation[2:6]).astype(float)
        if len(annotation[2:]) == 14:
            bbox_target = np.array(annotation[2:6]).astype(float)
            landmark = np.array(annotation[6:]).astype(float)

        sample = {'input_img': input_img, 'label': label, 'bbox_target': bbox_target, 'landmark': landmark}

        return sample

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█'):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = '\r')
    # Print New Line on Complete
    if iteration == total:
        print()

def total_num_filters(model):
    filters = 0
    for name, module in model.features._modules.items():
        if isinstance(module, torch.nn.modules.conv.Conv2d):
            filters = filters + module.out_channels
    return filters