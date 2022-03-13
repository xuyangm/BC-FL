import random
from itertools import combinations

import job_api_pb2_grpc
import job_api_pb2
import grpc
import pickle
import copy
import torch
from utils.divide_data import *
from selector import Selector
import time
import gc

from utils.model_creator import average_model, test, cal_shapley_values


class Group(object):
    """
    Define a group of clients which connect to the same edge server.
    """

    def __init__(self, group_id, ids, train_loaders, batch_sz=50, lr=1e-2, momentum=0.9, weight_decay=5e-4, local_steps=5):
        self.channel = None
        self.stub = None
        self.group_id = group_id
        self.client_ids = ids
        self.train_loaders = train_loaders
        self.global_model = None
        self.local_models = []
        self.device = torch.device("cuda")
        # torch.backends.cudnn.benchmark = True // if enabling it, the GPU memory usage increases a lot!
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.local_steps = local_steps
        self.accuracy = 0.0

        test_dataset = get_dataset(0, 'cifar', True)
        self.test_loader = select_dataset(1, test_dataset, batch_sz, 0, is_test=True)
        self.test_data_size = len(test_dataset.partitions[0])
        self.set_env()

    def set_env(self, seed=1):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def connect(self, addr):
        # connect to a server
        MAX_MESSAGE_LENGTH = 90000000
        self.channel = grpc.insecure_channel(addr, options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])
        self.stub = job_api_pb2_grpc.JobServiceStub(self.channel)

        # get global model
        request = job_api_pb2.ConnectRequest()
        request.group_id = self.group_id
        response = self.stub.Connect(request)
        self.global_model = pickle.loads(response.global_model)
        # self.global_model = self.global_model.to(device=self.device)
        for idx, _ in enumerate(self.client_ids):
            self.local_models.append(copy.deepcopy(self.global_model))

    def train(self):
        # test
        self.accuracy = test(self.global_model, self.test_loader, self.test_data_size)
        # train
        for idx, client in enumerate(self.client_ids):
            print("The {}-th client is training. Client ID: {}.".format(idx+1, client))

            self.local_models[idx] = self.local_models[idx].to(device=self.device)
            self.local_models[idx].train()
            optimizer = torch.optim.SGD(self.local_models[idx].parameters(),
                                        lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
            criterion = torch.nn.CrossEntropyLoss().to(device=self.device)

            step = 0
            while step < self.local_steps:
                step += 1
                print("epoch {}".format(step))
                try:
                    for (X, y) in self.train_loaders[idx]:
                        X = X.to(device=self.device)
                        y = y.to(device=self.device)
                        output = self.local_models[idx](X)
                        loss = criterion(output, y)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                except Exception as ex:
                    print(ex)
                    break
            self.local_models[idx] = self.local_models[idx].to('cpu')
            torch.cuda.empty_cache()

        # average models
        index = [_ for _ in range(len(self.local_models))]
        self.global_model = average_model(self.local_models, index)

        # create request
        request = job_api_pb2.TrainRequest()
        request.updates = pickle.dumps(self.global_model)
        _ = self.stub.Train(request)

    def close(self):
        self.channel.close()

    def get_shapley_values(self):
        shapely_values = {}
        value_list = cal_shapley_values(self.local_models, self.test_loader, self.test_data_size, self.accuracy)
        for idx, client_id in enumerate(self.client_ids):
            shapely_values[client_id] = value_list[idx]
        return shapely_values


def run():
    batch_sz = 50
    candidates = [_ for _ in range(1, 101)]
    rounds = 50
    selector = Selector(candidates)
    train_dataset = get_dataset(len(candidates), 'cifar', False, method='dirichlet', alpha=0.5)

    for r in range(rounds):
        t = time.time()
        # select participants
        participants = selector.select_participants(sample_size=4, method='Bandit')

        # prepare dataloader
        train_loaders = []
        for participant in participants:
            train_loaders.append(select_dataset(participant, train_dataset, batch_sz, 0))

        server_addr = 'localhost:12345'

        group = Group(r+1, participants, train_loaders)
        group.connect(server_addr)
        group.train()
        selector.update_contribution(group.get_shapley_values())
        group.close()

        # release memory when a round ends
        del group
        gc.collect()
        torch.cuda.empty_cache()
        print("Round {} finished, using {} min".format(r+1, (time.time()-t)/60))


if __name__ == '__main__':
    run()

