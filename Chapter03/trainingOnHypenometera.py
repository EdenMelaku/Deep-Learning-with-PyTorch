from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder


def loadingData():
    imageTransform = transforms.Compose(transforms.Scale([224, 224]),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                      )
    train = ImageFolder("", imageTransform)
    test = ImageFolder("", imageTransform)



train_data_gen=DataLoader(train, batch_size=64, num_workers=3)
test_data_gen=DataLoader(test, batch_size=64, num_workers=3)


class EdenNet1(nn.Module()):

    Layer1=nn.Linear(in_features=224, out_features=)