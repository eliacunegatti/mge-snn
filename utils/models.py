from models.models import Conv6
from models.resnet import resnet20, resnet32
from models.wide_resnet import WideResNet

def get_model(model_name, channels, classes):
    if model_name == 'CONV-6':
        model = Conv6(channels,classes)          
    elif model_name == 'Resnet-20':
        model = resnet20(classes)
    elif model_name == 'Resnet-32':
        model = resnet32(classes)
    elif model_name == 'Wide-Resnet-28-2':
        model = WideResNet(28,classes,2)
    else:
        raise Exception('Model not available')

    return model