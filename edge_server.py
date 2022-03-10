import grpc
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import job_api_pb2_grpc
import job_api_pb2
import pickle
from concurrent import futures
import time

from utils.divide_data import get_dataset
from utils.model_creator import init_model


class EdgeServer(job_api_pb2_grpc.JobServiceServicer):
    """
    Edge server
    """

    def __init__(self, batch_sz):
        self.round = 0
        self.addr = "localhost:12345"

        self.global_model = init_model()
        self.device = torch.device("cuda")

        test_dataset = get_dataset(0, 'cifar', True)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=batch_sz)
        self.test_data_size = len(test_dataset)

    def Connect(self, request, context):
        print("------------Round {}------------".format(request.group_id))
        response = job_api_pb2.ConnectResponse()
        response.global_model = pickle.dumps(self.global_model)
        self.round = request.group_id
        return response

    def Train(self, request, context):
        self.global_model = pickle.loads(request.updates)

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

        writer = SummaryWriter("./testing_logs")
        writer.add_scalar("total loss", total_loss, self.round)
        writer.add_scalar("total accuracy(%)", total_accuracy/self.test_data_size*100, self.round)
        writer.close()

        response = job_api_pb2.TrainResponse()
        response.round = 1
        return response

    def run(self):
        MAX_MESSAGE_LENGTH = 90000000
        server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options=[
            ('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
            ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH),
        ])
        job_api_pb2_grpc.add_JobServiceServicer_to_server(self, server)

        print('Starting server. Listening for clients...')
        server.add_insecure_port(self.addr)
        server.start()

        try:
            while True:
                time.sleep(86400)
        except KeyboardInterrupt:
            server.stop(0)


if __name__ == '__main__':
    s = EdgeServer(50)
    s.run()
