from itertools import combinations

import torch
import torchvision


def init_model():
    model = torchvision.models.shufflenet_v2_x2_0(num_classes=10)
    return model


def average_model(model_list, index):
    if len(index) == 1:
        return model_list[index[0]]

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

    return total_accuracy.item() / test_data_size


def cal_shapley_values(model_list, test_loader, test_data_size, accuracy):
    value_list = []
    accuracy_dict = {}
    l = [_ for _ in range(len(model_list))]

    for i in range(len(model_list)):
        cmbs = list(combinations(l, i+1))
        for cmb in cmbs:
            model = average_model(model_list, cmb)
            accuracy_dict[cmb] = test(model, test_loader, test_data_size)

    for i in range(len(model_list)):
        nl = [x for x in l if x != i]
        counter = 1
        value = accuracy_dict[(i,)] - accuracy

        for j in range(1, len(model_list)):
            partner_list = combinations(nl, j)
            for partners in partner_list:
                full_tup = tuple(sorted(partners + (i,)))
                value = value + accuracy_dict[full_tup] - accuracy_dict[partners]
                counter += 1

        value_list.append(float(value / counter))

    return value_list


