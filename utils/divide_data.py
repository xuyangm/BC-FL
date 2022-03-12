from torchvision import datasets, transforms
from utils.data_creator import CustomizedData
from random import Random
from torch.utils.data import DataLoader
import numpy as np
import csv
import os


class Partition(object):
    """ Dataset partitioning helper """

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]


class DataPartitioner(object):
    """Partition data by trace or random"""

    def __init__(self, data, seed=10):
        self.partitions = []
        self.rng = Random()
        self.rng.seed(seed)
        self.data = data
        np.random.seed(seed)
        self.data_len = len(self.data)

    def __len__(self):
        return self.data_len

    def trace_partition(self, data_map_file):
        """Read data mapping from data_map_file. Format: <client_id, sample_name, sample_category, category_id>"""

        clientId_maps = {}
        unique_clientIds = {}
        # load meta data from the data_map_file
        with open(data_map_file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            read_first = True
            sample_id = 0

            for row in csv_reader:
                if read_first:
                    read_first = False
                else:
                    client_id = row[0]

                    if client_id not in unique_clientIds:
                        unique_clientIds[client_id] = len(unique_clientIds)

                    clientId_maps[sample_id] = unique_clientIds[client_id]
                    sample_id += 1

        # Partition data given mapping
        self.partitions = [[] for _ in range(len(unique_clientIds))]

        for idx in range(sample_id):
            self.partitions[clientId_maps[idx]].append(idx)

    def partition_data_helper(self, num_clients, data_map_file=None):
        # read mapping file to partition trace
        if data_map_file is not None:
            self.trace_partition(data_map_file)
        else:
            self.uniform_partition(num_clients=num_clients)

    def uniform_partition(self, num_clients):
        # random partition
        data_len = self.data_len

        indexes = list(range(data_len))
        self.rng.shuffle(indexes)

        for _ in range(num_clients):
            part_len = int(1./num_clients * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        resultIndex = self.partitions[partition]
        self.rng.shuffle(resultIndex)

        return Partition(self.data, resultIndex)


def select_dataset(rank, partition, batch_size, num_loaders):
    """Load data given client Id"""
    partition = partition.use(rank - 1)
    dropLast = True
    if num_loaders == 0:
        time_out = 0
    else:
        time_out = 60

    return DataLoader(partition, batch_size=batch_size, shuffle=True, pin_memory=True, timeout=time_out, num_workers=num_loaders, drop_last=dropLast)


def get_dataset(num_clients, dataset_type, is_test):
    if dataset_type == 'cifar':
        train_transform, test_transform = get_data_transform(dataset_type)
        if not is_test:
            training_dataset = datasets.CIFAR10(
                root=dataset_type,
                train=True,
                download=True,
                transform=train_transform
            )
            training_sets = DataPartitioner(training_dataset)
            training_sets.partition_data_helper(num_clients=num_clients)
            return training_sets

        else:
            test_dataset = datasets.CIFAR10(
                root=dataset_type,
                train=False,
                download=True,
                transform=test_transform
            )
            test_sets = DataPartitioner(test_dataset)
            test_sets.partition_data_helper(num_clients=1)
            return test_sets
    elif dataset_type == 'femnist':
        train_transform, test_transform = get_data_transform(dataset_type)
        if not is_test:
            data_map_file = os.path.join('femnist', 'client_data_mapping', 'train.csv')
            training_dataset = CustomizedData('femnist', 'train.csv', train_transform)
            training_sets = DataPartitioner(training_dataset)
            training_sets.partition_data_helper(num_clients=num_clients, data_map_file=data_map_file)
            return training_sets

        else:
            test_dataset = CustomizedData('femnist', 'test.csv', test_transform)
            test_datasets = DataPartitioner(test_dataset)
            test_datasets.partition_data_helper(num_clients=1)  # the test data size is 1/20 now
            return test_datasets


def get_data_transform(data):
    if data == 'cifar':
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        test_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
        ])
    elif data == 'femnist':
        train_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        test_transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    return train_transform, test_transform
