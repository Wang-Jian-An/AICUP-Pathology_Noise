import os, json, gzip, torch
import pandas as pd
from functions import *
from Variable import *
from Model_Training_and_Evaluation_Flow import modelTrainingFlow
from DLFunction import DLTrainingFlow

modelMode = "DL"
device = "gpu" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    # Step1. 讀取資料
    rawData, audioDict = load_file(r"D:\OneDrive - tmu.edu.tw\AI_Competition\多模態病理嗓音分類競賽\Training Dataset.zip")
    
    # Step2. 把類別轉換成從 0 開始
    target_mapping = {
        i: index
        for index, i in enumerate(np.sort(rawData[mainTarget].unique()).tolist())
    }
    with open(os.path.join(mainPath, "Target_Mapping.json"), "w") as f:
        json.dump(target_mapping, f)
    rawData[mainTarget] = rawData[mainTarget].apply(lambda x: target_mapping[x])

    # Step2. 切割訓練、驗證與測試資料
    trainID, valiID, testID = split_train_validation_test_data(IDList = rawData["ID"].tolist()[:100],
                                                               targetList = rawData.loc[:100, mainTarget])
    trainData, valiData, testData = [
        rawData.query("ID in @oneIDList").reset_index(drop = True) for oneIDList in [trainID, valiID, testID]
    ]
    trainAudioDict, valiAudioDict, testAudioDict = [
        {
            oneID: audioDict[oneID]
            for oneID in oneIDList
        } for oneIDList in [trainID, valiID, testID]
    ]

    # Step3. 特徵工程
    trainAudioDataFrame, valiAudioDataFrame, testAudioDataFrame = [
        feature_engineering(oneAudioDict = oneData, FEID = 4) for oneData in [trainAudioDict, valiAudioDict, testAudioDict]
    ]

    # Step4. Model Training
    finalResult = list()
    if modelMode == "ML":
        inputFeatures = list(trainAudioDataFrame.columns)
        trainAudioDataFrame[mainTarget] = trainData[mainTarget].tolist()
        valiAudioDataFrame[mainTarget] = valiData[mainTarget].tolist()
        testAudioDataFrame[mainTarget] = testData[mainTarget].tolist()
        totalResult = modelTrainingFlow(trainData = trainAudioDataFrame,
                                        valiData = valiAudioDataFrame,
                                        testData = testAudioDataFrame,
                                        inputFeatures = inputFeatures, 
                                        target = mainTarget, 
                                        ml_methods = ["None"],
                                        targetType = "classification",
                                        mainMetric = "recall", 
                                        featureSelection = None, 
                                        featureImportance = None,
                                        modelFileName = None).fit(permutationImportanceMethod = None)
    else:
        for mainModel, allModelParams in modelParamsList.items():
            for oneModelParams in allModelParams:
                totalResult = DLTrainingFlow(
                    trainAudio = trainAudioDataFrame,
                    valiAudio = valiAudioDataFrame,
                    testAudio = testAudioDataFrame,
                    trainMetaData = trainData[metaDatainputFeatures],
                    valiMetaData = valiData[metaDatainputFeatures],
                    testMetaData = testData[metaDatainputFeatures],
                    trainTarget = trainData[mainTarget],
                    valiTarget = valiData[mainTarget],
                    testTarget = testData[mainTarget],
                    batch_size = 2,
                    device = device,
                    modelName = mainModel,
                    modelParams = {
                        "numSequence": trainAudioDataFrame[0].__len__(), 
                        "numAudioFeature": trainAudioDataFrame[0][0].__len__(), 
                        "numMetaDataFeature": metaDatainputFeatures.__len__(), 
                        **oneModelParams
                    },
                    lr = 1e-3
                ).fit()
                basicInformation = {
                    "Main-Model": mainModel,
                    **oneModelParams, 
                    "batch_size": 2
                }
                finalResult.append({
                    **basicInformation,
                    **totalResult["Evaluation"]
                })
    print(finalResult)
    pd.DataFrame(finalResult).to_excel(os.path.join(mainPath, "result", "test.xlsx"), index = None)