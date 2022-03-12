from itertools import combinations

import torch
import torchvision


def init_model():
    model = torchvision.models.shufflenet_v2_x2_0(num_classes=10)
    return model


def average_model(model_list, index):
    scale = 1.0 / len(index)
    global_state = model_list[index[0]].state_dict()

    for var in global_state:
        global_state[var] = global_state[var] * scale

    for i in range(1, len(index)):
        local_state = model_list[index[i]].state_dict()
        for var in global_state:
            global_state[var] = global_state[var] + local_state[var] * scale

    model = init_model()
    model.load_state_dict(global_state)
    return model


def test(model, test_loader, test_data_size):
    model = model.to('cuda')

    total_loss = 0
    total_accuracy = 0
    criterion = torch.nn.CrossEntropyLoss().to('cuda')

    model.eval()
    test_step = 0
    with torch.no_grad():
        for (X, y) in test_loader:
            X = X.to('cuda')
            y = y.to('cuda')
            output = model(X)
            total_accuracy += (output.argmax(1) == y).sum()
            loss = criterion(output, y)
            total_loss += loss.item()
            test_step += 1

    model = model.to('cpu')
    torch.cuda.empty_cache()

    return total_accuracy / test_data_size


def cal_shapley_values(idx, model_list, test_loader, test_data_size, accuracy):
    l = [_ for _ in range(len(model_list))]
    l.remove(idx)
    num = len(model_list)
    counter = 1
    value = test(model_list[idx], test_loader, test_data_size) - accuracy

    for i in range(1, num):
        partner_list = list(combinations(l, i))
        for partners in partner_list:
            base_model = average_model(model_list, partners)
            base_acc = test(base_model, test_loader, test_data_size)
            full_tup = partners + (idx,)
            full_model = average_model(model_list, full_tup)
            full_acc = test(full_model, test_loader, test_data_size)
            value = value + full_acc - base_acc
            counter += 1

    return float(value / counter)


