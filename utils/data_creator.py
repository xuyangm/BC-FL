from torchvision.transforms import transforms
from torch.utils.data import Dataset
from PIL import Image
import csv
import os


class CustomizedData(Dataset):

    def __init__(self, root_dir, fn):
        self.dataset_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.root_dir = root_dir
        data_path = os.path.join(root_dir, "client_data_mapping")
        data_path = os.path.join(data_path, fn)
        self.dataset = {}

        reader = csv.reader(open(data_path))
        is_first_row = True
        for row in reader:
            if is_first_row:
                is_first_row = False
                continue
            self.dataset[row[1]] = row[3]

    def __getitem__(self, idx):
        key_set = list(self.dataset.keys())
        img_path = os.path.join(self.root_dir, key_set[idx])
        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = self.dataset_transform(img)
        label = int(self.dataset[key_set[idx]])
        return img, label

    def __len__(self):
        return len(self.dataset)
