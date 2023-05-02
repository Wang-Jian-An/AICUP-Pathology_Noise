import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from DLModel import *
from multi_class_model_evaluation import model_evaluation

def transformDataIntoTensor(audioData, metaData, batch_size, targetData = None):

    """
    將資料轉換成 Tensor 格式
    oneData: The audio data or metadata
    """


    if type(audioData[0]) == list: # 若每一筆資訊資料的格式為 list
        audioTensor = torch.FloatTensor(
            audioData
        )
    else: # 否則為 np.ndarray
        audioData = [i.tolist() for i in audioData]
        audioTensor = torch.FloatTensor(
            audioData
        )
    # print(metaData)
    metaDataTensor = torch.FloatTensor(
        metaData.values.tolist()
    )

    if targetData is not None:
        oneHotMapping = {
            i: [1 if j == 1 else 0 for j in range(list(set(targetData)).__len__())]
            for i in sorted(list(set(targetData)))
        } # 將分類結果轉換成 One-Hot Encoding
        targetData = torch.FloatTensor([oneHotMapping[i] for i in targetData])
        datasets = TensorDataset(audioTensor, metaDataTensor, targetData)
    else:
        datasets = TensorDataset(audioTensor, metaDataTensor)
    return DataLoader(datasets, batch_size = batch_size)

def DLPrediction(model, predDataLoader):

    """
    簡介：使用已訓練的模型與預測資料進行資料預測
    
    """
    yhatList = list()
    yhatProbList = list()
    for onePred in predDataLoader:
        yhatProb = model(*onePred)
        yhat = torch.argmax(yhatProb, dim = 1).tolist()
        yhatList.extend(yhat)
        yhatProbList.extend(yhatProb.tolist())
    return {
        "yhat": yhatList,
        "yhatProb": yhatProbList
    }

class DLTrainingFlow():
    def __init__(
            self, 
            trainAudio: list,
            trainMetaData: dict, 
            trainTarget: list,
            valiAudio: list,
            valiMetaData: dict,
            valiTarget: list,  
            testAudio: list,
            testMetaData: dict,
            testTarget: list, 
            modelName, 
            modelParams,
            batch_size, 
            lr,
            device, 
            refit = False, 
            lossFuncName = None, 
            optimizerName = None,
            finalModelExportPath = None
    ):
        
        """
        trainAudio: 
        trainMetaData: 
        valiAudio: 
        valiMetaData: 
        testAudio: 
        testMetaData: 
        
        """

        self.trainAudio = trainAudio
        self.valiAudio = valiAudio
        self.trainValiAudio = self.trainAudio + self.valiAudio if self.valiAudio is not None else None
        self.testAudio = testAudio

        self.trainMetaData = trainMetaData
        self.valiMetaData = valiMetaData
        self.trainValiMetaData = {
            trainKey: trainValue + valiValue
            for (trainKey, trainValue), (valiKey, valiValue) in zip(trainMetaData.items(), valiMetaData.items())
        } if self.valiMetaData is not None else None
        self.testMetaData = testMetaData

        assert tuple(trainMetaData.keys()) == tuple(testMetaData.keys()), "The features in the train and test set must be same. "
        self.metaDataInputFeatures = trainMetaData.keys()

        self.trainTarget = trainTarget
        self.valiTarget = valiTarget
        self.testTarget = testTarget

        self.refit = refit
        if refit:
            if self.valiAudio is not None:
                self.allAudio = self.trainAudio + self.valiAudio + self.testAudio
            else:
                self.allAudio = self.trainAudio + self.testAudio

            if self.valiMetaData is not None:
                self.allMetaData = {
                    trainKey: trainValue + valiValue + testValue
                    for (trainKey, trainValue), (valiKey, valiValue), (testKey, testValue) in zip(
                        self.trainMetaData.items(), self.valiMetaData.items(), self.testMetaData.items()
                    )
                }
            else:
                self.allMetaData = {
                    trainKey: trainValue + testValue
                    for (trainKey, trainValue), (testKey, testValue) in zip(
                        self.trainMetaData.items(), self.testMetaData.items()
                    )
                }

        self.modelName = modelName
        self.lossFuncName = lossFuncName
        self.optimizerName = optimizerName
        self.modelParams = modelParams
        self.batch_size = batch_size
        self.lr = lr

        self.finalModelExportPath = finalModelExportPath
        self.device = device
        return 

    def fit(self):

        # Step1. 把資料轉換成 DataLoader
        trainDataLoader = transformDataIntoTensor(
            audioData = self.trainAudio, 
            metaData = self.trainMetaData, 
            batch_size = self.batch_size, 
            targetData = self.trainTarget
        )
        if self.valiAudio:
            valiDataLoader = transformDataIntoTensor(
                audioData = self.valiAudio, 
                metaData = self.valiMetaData, 
                batch_size = self.batch_size, 
                targetData = self.valiTarget
            )
        else:
            valiDataLoader = None

        testDataLoader = transformDataIntoTensor(
            audioData = self.testAudio,
            metaData = self.testMetaData,
            batch_size = self.batch_size,
        )

        # Step2. Define Model
        model = self.defineModel(modelName = self.modelName).to(self.device)
        self.loss_func = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.Adam(model.parameters(), lr = self.lr)

        # Step2. 針對訓練資料進行模型訓練、驗證資料進行 Overfitting 檢測
        model, trainLossList, valiLossList = self.DLTraining(
            model = model, 
            trainDataLoader = trainDataLoader, 
            epochs = 20, 
            valiDataLoader = valiDataLoader
        )

        # Step3. 針對測試資料進行模型預測
        testYhat = DLPrediction(
            model = model, 
            predDataLoader = testDataLoader
        )

        # Step4. 模型評估
        evaluationResult = model_evaluation(
            ytrue = self.testTarget,
            ypred = testYhat["yhat"],
            ypred_proba = testYhat["yhatProb"]
        )

        # Step5. 如需要 Refit，則開始進行
        if self.refit:
            pass

        return {
            "Evaluation": evaluationResult
        }

    def defineModel(self, modelName):

        if modelName == "onlyCNN":
            model = onlyCNNModel(
                **self.modelParams
            )
            pass 
        elif modelName == "onlyRNN-RNN":
            pass
        elif modelName == "onlyRNN-LSTM":
            pass
        elif modelName == "onlyRNN-GRU":
            pass
        elif modelName == "CNNandRNN":
            pass
        return model

    def DLTraining(self, model, trainDataLoader, epochs, valiDataLoader = None):

        """
        model
        loss_func
        optimizer
        trainDataLoader
        valiDataLoader
        """
        trainLossList = list()
        valiLossList = list() if valiDataLoader else None
        for epoch in range(epochs):
            trainLoss = list()
            valiLoss = list() if valiDataLoader else None
            for audioData, metaData, target in trainDataLoader:
                self.optimizer.zero_grad()
                yhat = model(
                    audioData = audioData.to(self.device),
                    metaData = metaData.to(self.device)
                )
                loss = self.loss_func(yhat, target)
                loss.backward()
                trainLoss.append(loss.cpu() / audioData.size()[0])
                break
            trainLossList.append(torch.mean(torch.FloatTensor(trainLoss)))
            if valiDataLoader:
                with torch.no_grad():
                    for audioData, metaData, target in valiDataLoader:
                        yhat = model(
                            audioData = audioData.to(self.device),
                            metaData = metaData.to(self.device)
                        )
                        loss = self.loss_func(yhat, target)
                        valiLoss.append(loss.cpu() / audioData.size()[0])
                        break
                valiLossList.append(torch.mean(torch.FloatTensor(valiLoss)))
            break
        return model, trainLossList, valiLossList

    def DLEvaluation(self):
        return 