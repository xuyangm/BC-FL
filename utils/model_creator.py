import torchvision


def init_model():
    model = torchvision.models.shufflenet_v2_x2_0(num_classes=62)
    return model
