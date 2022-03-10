import job_api_pb2_grpc
import job_api_pb2
import grpc
import pickle
import copy
import torch
from utils.divide_data import *
from selector import Selector
import time


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
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.local_steps = local_steps

        test_dataset = get_dataset(0, 'cifar', True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=batch_sz)
        self.test_data_size = len(test_dataset)

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
                    print("error type: {}".format(ex))
                    break
            self.local_models[idx] = self.local_models[idx].to('cpu')

        # average models
        scale = 1.0/len(self.local_models)
        print("scale={}".format(scale))
        global_state = self.local_models[0].state_dict()

        for var in global_state:
            global_state[var] = global_state[var]*scale

        for i in range(1, len(self.local_models)):
            local_state = self.local_models[i].state_dict()
            for var in global_state:
                global_state[var] = global_state[var] + local_state[var]*scale

        self.global_model.load_state_dict(global_state)

        self.test()

        self.global_model = self.global_model.cpu()

        # create request
        request = job_api_pb2.TrainRequest()
        request.updates = pickle.dumps(self.global_model)
        _ = self.stub.Train(request)

    def close(self):
        self.channel.close()

    def test(self):
        # test
        self.global_model = self.global_model.to(device=self.device)

        total_loss = 0
        total_accuracy = 0
        criterion = torch.nn.CrossEntropyLoss().to(device=self.device)

        self.global_model.eval()
        test_step = 0
        with torch.no_grad():
            for (X, y) in self.test_loader:
                X = X.to(device=self.device)
                y = y.to(device=self.device)
                output = self.global_model(X)
                total_accuracy += (output.argmax(1) == y).sum()
                loss = criterion(output, y)
                total_loss += loss.item()
                test_step += 1

        print("total loss: {}".format(total_loss))
        print("total accuracy: {}%".format(total_accuracy / self.test_data_size * 100))


def run():
    batch_sz = 50
    candidates = [_ for _ in range(1, 101)]
    rounds = 200
    selector = Selector(candidates)
    train_dataset = get_dataset(len(candidates), 'cifar', False)
    for r in range(rounds):
        t = time.time()
        # select participants
        participants = selector.select_participants(sample_size=10)

        # prepare datasets
        train_loaders = []
        for participant in participants:
            train_loaders.append(select_dataset(participant, train_dataset, batch_sz, 0))

        server_addr = 'localhost:12345'

        group = Group(r+1, participants, train_loaders)
        group.connect(server_addr)
        group.train()
        group.close()
        del group
        print("Round {} finished, using {} min".format(r+1, (time.time()-t)/60))


if __name__ == '__main__':
    run()

