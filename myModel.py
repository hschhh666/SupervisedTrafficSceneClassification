import torch
import torch.nn as nn
from torchvision.models import resnet50

class myResnet50(nn.Module):
    def __init__(self, num_classes = 4):
        super(myResnet50, self).__init__()
        self.model = resnet50(num_classes=365)
        places365_pre_trained_model_file = '/home/hsc/Research/TrafficSceneClassification/code/testExperiment/places365PreTrained/resnet50_places365.pth.tar'
        checkpoint = torch.load(places365_pre_trained_model_file, map_location=lambda storage, loc: storage)
        state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(state_dict)
        self.model.fc = nn.Linear(512 * 4, num_classes)
        self.model = nn.DataParallel(self.model)
    
    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    x = torch.rand([1,3,400,224])
    model = myResnet50()
    print(model(x).shape)