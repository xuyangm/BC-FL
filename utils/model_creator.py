import torchvision


def init_model():
    model = torchvision.models.densenet201(num_classes=10)
    return model
