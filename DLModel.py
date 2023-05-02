import torch
import torch.nn as nn
import torch.nn.functional as F

class onlyCNNModel(nn.Module):
    def __init__(
            self, 
            numSequence: int, 
            numFeatures: int, 
            numTarget: int, 
            numCNNLayer: int, 
            numDecoderLayer: int, 
            kernel_size: int, 
            numStride: int
    ):
        super(onlyCNNModel, self).__init__()

        """
        numFeatures: The number of features.
        numCNNLayers: The number of CNN layers.
        numDecoderLayer: The number of Decoder layers. 
        kernel_size: The size of kernel
        """

        self.featureMapList = [int(numFeatures // i) for i in range(1, numCNNLayer+2)]
        self.CNNModel = [
            nn.Conv1d(in_channels = self.featureMapList[i], 
                      out_channels = self.featureMapList[i+1],
                      kernel_size = kernel_size,
                      stride = numStride) for i in range(self.featureMapList.__len__()-1)
        ]
        self.CNNModel = nn.Sequential(
            *[
                i
                for oneCNN in self.CNNModel for i in [oneCNN, nn.ReLU()]
            ] 
        )
        self.CNNOutputShape = numSequence
        for oneCNN in range(self.featureMapList.__len__()-1):
            self.CNNOutputShape = int((self.CNNOutputShape - kernel_size ) // numStride) + 1
        self.CNNOutputShape = self.CNNOutputShape * self.featureMapList[-1]
        self.nodesList = [int(i // 1) for i in reversed(torch.linspace(numTarget, self.CNNOutputShape, numDecoderLayer+2).tolist())]
        
        self.DecoderModel = nn.Sequential(*[
            nn.Linear(in_features = self.nodesList[i], 
                      out_features = self.nodesList[i+1]) 
            for i in range(self.nodesList.__len__() - 1)
        ])
        return 
    
    def forward(self, audioData, metaData):
        audioData = audioData.permute(0, 2, 1)
        yhat = self.CNNModel(audioData)
        yhat = yhat.view(yhat.size()[0], -1)
        yhat = self.DecoderModel(yhat)
        return nn.Softmax(dim = 1)(yhat)
    
class OnlyRNNModel(nn.Module):
    def __init__(self):
        super(OnlyRNNModel, self).__init__()
        return 
    
    def forward(self):
        return 

class CNNandRNNModel(nn.Module):
    def __init__(self):
        super(CNNandRNNModel, self).__init__()
        return 
    
    def forward(self):
        return 
