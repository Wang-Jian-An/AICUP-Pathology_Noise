import os, json, gzip, pickle
import torch
from datetime import datetime
import pandas as pd
from functions import *
from DLFunction import *

def defineModel(modelName, modelParams, device):
        modelParams = {
            "device": device,
            **modelParams
        }
        if modelName == "onlyCNN":
            model = onlyCNNModel(
                **modelParams
            )
        elif modelName == "onlyRNN-RNN":
            modelParams = {
                "rnnModel": "RNN",
                **modelParams
            }
            model = onlyRNNModel(
                **modelParams
            )
        elif modelName == "onlyRNN-LSTM":
            modelParams = {
                "rnnModel": "LSTM",
                **modelParams
            }
            model = onlyRNNModel(
                **modelParams
            )
        elif modelName == "onlyRNN-GRU":
            modelParams = {
                "rnnModel": "GRU",
                **modelParams
            }
            model = onlyRNNModel(
                **modelParams
            )
        elif modelName == "CNNandRNNModel":
            modelParams = {
                "rnnModel": "GRU",
                **modelParams
            }
            model = CNNandRNNModel(
                **modelParams
            )
        elif modelName == "CNNandAttentionModel":
            model = CNNandAttentionModel(
                **modelParams
            )
        elif modelName == "pretrainedModel":
            modelParams = {
                "device": device,
                **modelParams
            }
            audioModel = audioPretrainedModel(
                pretrainedModelName = modelParams["pretrainedModelName"],
                pretrained = modelParams["pretrained"],
                device = device
            ).to(device)
            metaDataModel1 = metaDataModel(
                numMetaDataFeature = modelParams["numMetaDataFeature"],
                numResidualLayer = 1,
                numEachResidualLayer = modelParams["numEachResidualLayer"],
                device = device
            ).to(device)
            combinedModel = audio_metaData_combined(
                numAudioFeature = 1000,
                numMetaDataFeature = metaDataModel1.featureList[-1],
                numTarget = modelParams["numTarget"],
                numDecoderLayer = modelParams["numDecoderLayer"],
                combinedMethod = modelParams["combinedMethod"],
                device = device
            ).to(device)
            return audioModel, metaDataModel1, combinedModel
        return model

if __name__ == "__main__":
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Step1. 讀取資料
    rawData, audioDict = load_file(r"D:\OneDrive - tmu.edu.tw\AI_Competition\多模態病理嗓音分類競賽\Public Testing Dataset.zip", train_or_predict = "predict")
    dataID = rawData["ID"].tolist() # For submission

    # Step2. 讀取 Target Mapping
    with open(os.path.join(mainPath, "Target_Mapping.json"), "r") as f:
        target_mapping = json.load(f)
    inverse_target_mapping = {
        index: str(i) for i, index in target_mapping.items()
    } 

    # Step3. 配對每個資料ID與音訊檔案
    predAudioDict = {
        oneID: audioDict[oneID] for oneID in rawData["ID"]
    }

    # Step4. 特徵工程
    predAudioDict = feature_engineering(oneAudioDict = predAudioDict, FEID = 5)
    rawData = FE_for_metaData(oneData = rawData.copy(), jsonPath = mainPath)

    # Step5. 讀取模型
    modelName = "pretrainedModel"
    with open("128_87_36_5_resnext101_32x8d_True_1_1_inner_product.json") as f:
        modelParams = json.load(f)
    audioModelPath = os.path.join("audioModel_128_87_36_5_resnext101_32x8d_True_1_1_inner_product.pth")
    metaDataModelPath = os.path.join("metaDataModel_128_87_36_5_resnext101_32x8d_True_1_1_inner_product.pth")
    combinedModelPath = os.path.join("combinedModel_128_87_36_5_resnext101_32x8d_True_1_1_inner_product.pth")
    model = defineModel(modelName = modelName, 
                        modelParams = modelParams,
                        device = device)
    if modelName == "pretrainedModel":
        audioModel, metaDataModel, combinedModel = model
        del model

    if ".gzip" in audioModelPath:
        with gzip.GzipFile(audioModelPath, "r") as f:
            model = pickle.load(f)
    elif ".pkl" in audioModelPath:
        pass
    elif ".pt" in audioModelPath or ".pth" in audioModelPath:
        audioModel= torch.load(audioModelPath).to(device)
        metaDataModel1= torch.load(metaDataModelPath).to(device)
        combinedModel = torch.load(combinedModelPath).to(device)

    # Step6. 模型預測
    if ".gzip" in audioModelPath or ".pkl" in audioModelPath:
        yhat = model.predict(predAudioDict)
    else:
        predDataLoader = transformDataIntoTensor(
            audioData = predAudioDict,
            metaData = rawData,
            batch_size = rawData.shape[0]
        )
        predAudio, predmetaData = next(iter(predDataLoader))
        audioYhat = audioModelPath(predAudio, mode = "combined")
        metaDataYhat = metaDataModel1(predmetaData, mode = "combined")
        yhat = combinedModel(audioYhat, metaDataYhat)
        yhat = model(predAudio, predmetaData)
    
    # Step7. 儲存結果
    yhatData = {
        oneKey: oneYhat for oneKey, oneYhat in zip(dataID, yhat)
    }
    pd.DataFrame(yhatData).to_csv(os.path.join(mainPath, "Submission", "Submission_{}".format(str(datetime.today()).split(" ")[0])))