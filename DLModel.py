import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *

class onlyCNNModel(nn.Module):
    def __init__(
            self, 
            numSequence: int, 
            numAudioFeature: int, 
            numMetaDataFeature: int, 
            numTarget: int, 
            numCNNLayer: int, 
            numMetaDataLayer: int, 
            numDecoderLayer: int, 
            kernel_size: tuple, 
            numStride: int,
            device
    ):
        super(onlyCNNModel, self).__init__()

        """
        numFeatures: The number of features.
        numCNNLayers: The number of CNN layers.
        numMetaDataLayers: The number of NN layers for metaData. 
        numDecoderLayer: The number of Decoder layers. 
        kernel_size: The size of kernel.
        numStride: . 
        """
        self.device = device
        self.featureMapList = [
            int(1 // i) for i in list(torch.linspace(1, 0.1, numCNNLayer // 2 + 1))
        ] + [
            int((1 / 0.1) // i) for i in list(torch.linspace(1, 10, numCNNLayer // 2 + 1))
        ]
        self.CNNModel = [
            nn.Conv2d(in_channels = self.featureMapList[i], 
                      out_channels = self.featureMapList[i+1],
                      kernel_size = kernel_size,
                      stride = numStride) for i in range(self.featureMapList.__len__()-1)
        ]
        self.CNNModel = nn.Sequential(
            *[
                i
                for oneCNN in self.CNNModel for i in [oneCNN, nn.Tanh()]
            ] 
        )

        if numMetaDataLayer > 0:
            metaDataFeatureList = [numMetaDataFeature // int(i) for i in reversed(torch.linspace(1, numMetaDataFeature, numMetaDataLayer+1).tolist())]
            self.metaDataModel = nn.Sequential(*[
                nn.Linear(in_features = metaDataFeatureList[i],
                          out_features = metaDataFeatureList[i+1]) 
                for i in range(metaDataFeatureList.__len__()-1)
            ])    
        else:
            metaDataFeatureList = [0]
        
        # The calculation of the shape of the output of CNN 
        self.CNNOutputSequence = numSequence
        for oneCNN in range(numCNNLayer // 2 + 2):
            self.CNNOutputSequence = int((self.CNNOutputSequence - kernel_size[0] ) // numStride) + 1
        
        self.CNNOutputFeature = numAudioFeature
        for oneCNN in range(numCNNLayer // 2 + 2):
            self.CNNOutputFeature = int((self.CNNOutputFeature - kernel_size[1] ) // numStride) + 1
    
        self.CNNOutputShape = self.CNNOutputSequence * self.featureMapList[-1] * self.CNNOutputFeature
        self.CNNplusMetaDataFeatures = self.CNNOutputShape + metaDataFeatureList[-1]
        self.nodesList = [int(i // 1) for i in reversed(torch.linspace(numTarget, self.CNNplusMetaDataFeatures, numDecoderLayer+2).tolist())]
        self.DecoderModel = nn.Sequential(*[
            nn.Linear(in_features = self.nodesList[i], 
                      out_features = self.nodesList[i+1]) 
            for i in range(self.nodesList.__len__() - 1)
        ])
        return 
    
    def forward(self, audioData, metaData):

        audioData = audioData.to(self.device)
        metaData = metaData.to(self.device)

        audioData = audioData.unsqueeze(dim = -1)
        audioData = audioData.permute(0, 3, 1, 2)
        Audioyhat = self.CNNModel(audioData)

        if "self.metaDataModel" in globals():
            MetaDatayhat = self.metaDataModel(metaData)
        else:
            MetaDatayhat = metaData.clone()
        Audioyhat = Audioyhat.view(Audioyhat.size()[0], -1)
        yhat = torch.concat([Audioyhat, MetaDatayhat], dim = 1)

        yhat = self.DecoderModel(yhat)
        return nn.Softmax(dim = 1)(yhat)
    
class onlyRNNModel(nn.Module):
    def __init__(self, 
        numSequence: int, 
        numAudioFeature: int, 
        numMetaDataFeature: int, 
        numTarget: int, 
        numRNNLayer: int, 
        numMetaDataLayer: int, 
        numDecoderLayer: int, 
        rnnModel: str = "GRU", 
    ):
        super(onlyRNNModel, self).__init__()

        # Audio Data
        self.numRNNLayer = numRNNLayer
        self.numSequence = numSequence
        self.numAudioFeature = numAudioFeature
        if rnnModel == "RNN":
            self.RNNModel = nn.RNN(input_size = numAudioFeature,
                                   hidden_size = numAudioFeature // numRNNLayer,
                                   num_layers = numRNNLayer,
                                   batch_first = True)
        elif rnnModel == "LSTM":
            self.RNNModel = nn.LSTM(input_size = numAudioFeature,
                                   hidden_size = numAudioFeature // numRNNLayer,
                                   num_layers = numRNNLayer,
                                   batch_first = True)
        elif rnnModel == "GRU":
            self.RNNModel = nn.GRU(input_size = numAudioFeature,
                                   hidden_size = numAudioFeature // numRNNLayer,
                                   num_layers = numRNNLayer,
                                   batch_first = True)

        if numMetaDataLayer > 0:
            metaDataFeatureList = [numMetaDataFeature // int(i) for i in reversed(torch.linspace(1, numMetaDataFeature, numMetaDataLayer+1).tolist())]
            self.metaDataModel = nn.Sequential(*[
                nn.Linear(in_features = metaDataFeatureList[i],
                          out_features = metaDataFeatureList[i+1]) 
                for i in range(metaDataFeatureList.__len__()-1)
            ])    
        else:
            metaDataFeatureList = [0]

        # The calculation of the shape of the output of CNN 
        self.RNNOutputShape = (numAudioFeature // numRNNLayer) * numSequence
        self.RNNplusMetaDataFeatures = self.RNNOutputShape + metaDataFeatureList[-1]
        print(self.RNNOutputShape, metaDataFeatureList)
        self.nodesList = [int(i // 1) for i in reversed(torch.linspace(numTarget, self.RNNplusMetaDataFeatures, numDecoderLayer+2).tolist())]

        self.DecoderModel = nn.Sequential(*[
            nn.Linear(in_features = self.nodesList[i], 
                        out_features = self.nodesList[i+1]) 
            for i in range(self.nodesList.__len__() - 1)
        ])

        return 
    
    def forward(self, audioData, metaData):

        self.hidden_state = torch.randn(size = (self.numRNNLayer, audioData.size()[0], self.numAudioFeature // self.numRNNLayer))    
        audioYhat, _ = self.RNNModel(audioData, self.hidden_state)

        if "self.metaDataModel" in globals():
            metaDataYhat = self.metaDataModel(metaData)
        else:
            metaDataYhat = metaData.clone()

        audioYhat = audioYhat.reshape((audioYhat.size()[0], -1))
        yhat = torch.concat([audioYhat, metaDataYhat], dim = 1)

        yhat = self.DecoderModel(yhat)
        return nn.Softmax(dim = 1)(yhat)

class CNNandRNNModel(nn.Module):
    def __init__(self, 
        numSequence: int, 
        numAudioFeature: int, 
        numMetaDataFeature: int, 
        numTarget: int, 
        numCNNLayer: int, 
        numRNNLayer: int, 
        numMetaDataLayer: int, 
        numDecoderLayer: int, 
        kernel_size: int, 
        numStride: int,
        rnnModel: str = "GRU"
    ):
        super(CNNandRNNModel, self).__init__()

        """
        Overview: The combination of CNN and RNN model. First step is CNN and the second one is RNN. 
        """

        # Audio Data
        self.numRNNLayer = numRNNLayer
        self.numSequence = numSequence
        self.numAudioFeature = numAudioFeature
        self.featureMapList = [int(numAudioFeature // i) for i in range(1, numCNNLayer+2)]
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

        if rnnModel == "RNN":
            self.RNNModel = nn.RNN(input_size = self.featureMapList[-1],
                                   hidden_size = self.featureMapList[-1] // numRNNLayer,
                                   num_layers = numRNNLayer,
                                   batch_first = True)
        elif rnnModel == "LSTM":
            self.RNNModel = nn.LSTM(input_size = self.featureMapList[-1],
                                   hidden_size = self.featureMapList[-1] // numRNNLayer,
                                   num_layers = numRNNLayer,
                                   batch_first = True)
        elif rnnModel == "GRU":
            self.RNNModel = nn.GRU(input_size = self.featureMapList[-1],
                                   hidden_size = self.featureMapList[-1] // numRNNLayer,
                                   num_layers = numRNNLayer,
                                   batch_first = True)

        # MetaData
        if numMetaDataLayer > 0:
            metaDataFeatureList = [
                numMetaDataFeature // int(i) for i in reversed(torch.linspace(1, numMetaDataFeature, numMetaDataLayer+1).tolist())
            ]
            self.metaDataModel = nn.Sequential(*[
                nn.Linear(in_features = metaDataFeatureList[i],
                          out_features = metaDataFeatureList[i+1]) 
                for i in range(metaDataFeatureList.__len__()-1)
            ])    
        else:
            metaDataFeatureList = [0]
        
        # The calculation of the shape of the output of CNN and RNN
        self.CNNOutputShape = numSequence
        for oneCNN in range(self.featureMapList.__len__()-1):
            self.CNNOutputShape = int((self.CNNOutputShape - kernel_size ) // numStride) + 1
        self.RNNOutputShape = (self.featureMapList[-1] // numRNNLayer) * self.CNNOutputShape
        self.RNNplusMetaDataFeatures = self.RNNOutputShape + metaDataFeatureList[-1]
        print(self.RNNOutputShape, metaDataFeatureList)
        self.nodesList = [
            int(i // 1) for i in reversed(torch.linspace(numTarget, self.RNNplusMetaDataFeatures, numDecoderLayer+2).tolist())
        ]

        self.DecoderModel = nn.Sequential(*[
            nn.Linear(in_features = self.nodesList[i], 
                        out_features = self.nodesList[i+1]) 
            for i in range(self.nodesList.__len__() - 1)
        ])
        return 
    
    def forward(self, audioData, metaData):

        # CNN
        audioData = audioData.permute(0, 2, 1)
        audioYhat = self.CNNModel(audioData)
        audioYhat = audioYhat.permute(0, 2, 1)

        self.hidden_state = torch.randn(size = (self.numRNNLayer, audioYhat.size()[0], self.featureMapList[-1] // self.numRNNLayer))    
        audioYhat, _ = self.RNNModel(audioYhat, self.hidden_state)

        if "self.metaDataModel" in globals():
            metaDataYhat = self.metaDataModel(metaData)
        else:
            metaDataYhat = metaData.clone()

        audioYhat = audioYhat.reshape((audioYhat.size()[0], -1))
        print(audioYhat.size(), metaDataYhat.size())
        yhat = torch.concat([audioYhat, metaDataYhat], dim = 1)

        yhat = self.DecoderModel(yhat)
        return nn.Softmax(dim = 1)(yhat)
    
class CNNandAttentionModel(nn.Module):
    def __init__(self, 
        numSequence: int, 
        numAudioFeature: int, 
        numMetaDataFeature: int, 
        numTarget: int, 
        numCNNLayer: int, 
        numMetaDataLayer: int, 
        numDecoderLayer: int, 
        kernel_size: int, 
        numStride: int,
        device, 
        rnnModel: str = "GRU"
    ):
        super(CNNandAttentionModel, self).__init__()

        """
        Overview: The combination of CNN and RNN model. First step is CNN and the second one is RNN. 
        """

        # Audio Data
        self.numSequence = numSequence
        self.numAudioFeature = numAudioFeature
        self.featureMapList = [int(i) for i in [numAudioFeature, *torch.linspace(numAudioFeature*2, numAudioFeature*0.5, numCNNLayer).tolist()]]
        print(self.featureMapList, self.featureMapList.__len__())
        self.CNNModel = [
            nn.Conv1d(in_channels = self.featureMapList[i], 
                      out_channels = self.featureMapList[i+1],
                      kernel_size = kernel_size,
                      stride = numStride) for i in range(self.featureMapList.__len__()-1)
        ]
        self.CNNModel = nn.Sequential(
            *[
                i
                for oneCNN in self.CNNModel for i in [oneCNN, nn.Tanh()]
            ] 
        )

        self.SelfAttention = nn.MultiheadAttention(embed_dim = self.featureMapList[-1],
                                                   num_heads = 1)

        # MetaData
        if numMetaDataLayer > 0:
            metaDataFeatureList = [
                numMetaDataFeature // int(i) for i in reversed(torch.linspace(1, numMetaDataFeature, numMetaDataLayer+1).tolist())
            ]
            self.metaDataModel = nn.Sequential(*[
                nn.Linear(in_features = metaDataFeatureList[i],
                          out_features = metaDataFeatureList[i+1]) 
                for i in range(metaDataFeatureList.__len__()-1)
            ])    
        else:
            metaDataFeatureList = [0]
        
        # The calculation of the shape of the output of CNN and RNN
        self.CNNOutputShape = numSequence
        for oneCNN in range(numCNNLayer):
            self.CNNOutputShape = int((self.CNNOutputShape - kernel_size ) // numStride) + 1
            print(self.CNNOutputShape)
        print(self.featureMapList[-1], self.CNNOutputShape)
        self.RNNOutputShape = self.featureMapList[-1] * self.CNNOutputShape
        self.RNNplusMetaDataFeatures = self.RNNOutputShape + metaDataFeatureList[-1]
        self.nodesList = [
            int(i // 1) for i in reversed(torch.linspace(numTarget, self.RNNplusMetaDataFeatures, numDecoderLayer+2).tolist())
        ]

        self.DecoderModel = nn.Sequential(*[
            nn.Linear(in_features = self.nodesList[i], 
                        out_features = self.nodesList[i+1]) 
            for i in range(self.nodesList.__len__() - 1)
        ])
        return 
    
    def forward(self, audioData, metaData):

        # CNN
        audioData = audioData.permute(0, 2, 1)
        audioYhat = self.CNNModel(audioData)
        audioYhat = audioYhat.permute(0, 2, 1)

        
        audioYhat, _ = self.SelfAttention(query = audioYhat, key = audioYhat, value = audioYhat)

        if "self.metaDataModel" in globals():
            metaDataYhat = self.metaDataModel(metaData)
        else:
            metaDataYhat = metaData.clone()

        audioYhat = audioYhat.reshape((audioYhat.size()[0], -1))
        yhat = torch.concat([audioYhat, metaDataYhat], dim = 1)

        yhat = self.DecoderModel(yhat)
        return nn.Softmax(dim = 1)(yhat) 

class pretrainedModel(nn.Module):
    def __init__(
            self, 
            numSequence: int, 
            numAudioFeature: int, 
            numMetaDataFeature: int, 
            numTarget: int, 
            pretrainedModelName: str,
            numMetaDataLayer: int,
            numDecoderLayer: int, 
            device
    ):
        super(pretrainedModel, self).__init__()
        """
        Reference: https://pytorch.org/vision/main/models.html#classification
        """

        self.device = device
        if pretrainedModelName == "resnet18":
            self.audioModel = resnet18(pretrained = True)
        elif pretrainedModelName == "resnet34":
            pass
        elif pretrainedModelName == "resnet50":
            pass
        elif pretrainedModelName == "vit_b_16":
            pass
        elif pretrainedModelName == "vit_l_16":
            pass

        if numMetaDataLayer > 0:
            metaDataFeatureList = [numMetaDataFeature // int(i) for i in reversed(torch.linspace(1, numMetaDataFeature, numMetaDataLayer+1).tolist())]
            self.metaDataModel = nn.Sequential(*[
                nn.Linear(in_features = metaDataFeatureList[i],
                          out_features = metaDataFeatureList[i+1]) 
                for i in range(metaDataFeatureList.__len__()-1)
            ])    
        else:
            metaDataFeatureList = [0]
           
        self.CNNplusMetaDataFeatures = 1000 + metaDataFeatureList[-1]
        self.nodesList = [int(i // 1) for i in reversed(torch.linspace(numTarget, self.CNNplusMetaDataFeatures, numDecoderLayer+2).tolist())]
        self.DecoderModel = nn.Sequential(*[
            nn.Linear(in_features = self.nodesList[i], 
                      out_features = self.nodesList[i+1]) 
            for i in range(self.nodesList.__len__() - 1)
        ])
        return
    
    def forward(self, audioData, metaData):

        audioData = audioData.to(self.device)
        metaData = metaData.to(self.device)

        audioData = audioData.unsqueeze(dim = -1)

        # 把 Audio Data 的 Channel 複製成三個Channel
        audioData = torch.cat((audioData, audioData, audioData), dim = -1)

        audioData = audioData.permute(0, 3, 1, 2)
        audioYhat = self.audioModel(audioData)

        if "self.metaDataModel" in globals():
            metaDataYhat = self.metaDataModel(metaData)
        else:
            metaDataYhat = metaData.clone()
        audioYhat = audioYhat.view(audioYhat.size()[0], -1)
        yhat = torch.concat([audioYhat, metaDataYhat], dim = 1)

        yhat = self.DecoderModel(yhat)
        return nn.Softmax(dim = 1)(yhat)