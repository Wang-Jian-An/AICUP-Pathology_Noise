import os, json, gzip, torch
import pandas as pd
from functions import *
from Variable import *
from AutoML_Flow.Model_Training_and_Evaluation_Flow import modelTrainingFlow
from DLFunction import DLTrainingFlow

REMOVE_HEAD_TAIL_TIME = 0.2
FIX_AUDIO_TIME = 2.5
BATCH_SIZE = 8
MODEL_MODE = "ML"
device = "cuda" if torch.cuda.is_available() else "cpu"

if __name__ == "__main__":

    # Step1. 讀取資料
    print("Start loading data")
    rawData, audioDict = load_file(r"D:\OneDrive - tmu.edu.tw\AI_Competition\多模態病理嗓音分類競賽\Training Dataset.zip", train_or_predict = "train")
    
    # Step2. 把類別轉換成從 0 開始
    print("Start re-defining target")
    target_mapping = {
        i: index
        for index, i in enumerate(np.sort(rawData[mainTarget].unique()).tolist())
    }
    with open(os.path.join(mainPath, "Target_Mapping.json"), "w") as f:
        json.dump(target_mapping, f)
    rawData[mainTarget] = rawData[mainTarget].apply(lambda x: target_mapping[x])

    # Step3. 儲存 Multiclass 於 One-Hot Encoding 的對照表
    print("Start one-hot encoding for target")
    for oneFeature in metaDataMulticlassFeatures:
        uniqueFeature = sorted(rawData[oneFeature].unique().tolist())
        mappingList = {
            oneFeature: [1 if i == index else 0 for i in range(uniqueFeature.__len__())]
            for index, oneFeature in enumerate(uniqueFeature)
        }
        with open(os.path.join(mainPath, f"{oneFeature}_mapping.json"), "w") as f:
            json.dump(mappingList, f)

    # Step4. 統一將音訊資料移除前後各數秒
    print("Start removing the head and tail of the audio")
    audioDict = {
        oneID: removeHeadTailAudio(oneAudio, removeOneSideTime = REMOVE_HEAD_TAIL_TIME / 2)
        for oneID, oneAudio in audioDict.items()
    }

    # Step5. 切割訓練、驗證與測試資料
    print("Start splitting data")
    trainID, valiID, testID = split_train_validation_test_data(IDList = rawData["ID"].tolist(),
                                                               targetList = rawData[mainTarget])
    rawData = rawData.set_index("ID")
    trainData, valiData, testData = [
        rawData.loc[oneIDList, :].reset_index("ID") for oneIDList in [trainID, valiID, testID]
    ]
    trainAudioDict, valiAudioDict, testAudioDict = [
        {
            oneID: audioDict[oneID]
            for oneID in oneIDList
        } for oneIDList in [trainID, valiID, testID]
    ]

    # Step6. 資料增量
    print("Start data augmentation")
    trainAudioDict, trainData = augmentationFlow(
        totalAudioData = trainAudioDict,
        totalMetaData = trainData, 
        augmentMode = augmentMode
    )

    # Step7. 統一所有音訊資料至相同長度
    print("Start normalizing the length of the audio")
    trainAudioDict, valiAudioDict, testAudioDict = [
        normalizeAudioLength(audioDict = oneAudioDict, fixLength = FIX_AUDIO_TIME)
        for oneAudioDict in [trainAudioDict, valiAudioDict, testAudioDict]
    ]

    # Step8. 將音訊轉換成頻譜圖之特徵工程、將 MetaData 進行特徵工程處理
    trainData, valiData, testData = [
        FE_for_metaData(oneData = oneData.reset_index(drop = True).copy(), jsonPath = mainPath) 
        for oneData in [trainData, valiData, testData]
    ]    
    metaDatainputFeatures = [i for i in trainData.columns if i != mainTarget] # For training_flow.py
    for oneAudioFE in FEAudio:
        trainAudioDataFrame, valiAudioDataFrame, testAudioDataFrame = [
            audioFeatureEngineer(
                oneAudioDict = oneData, 
                **oneAudioFE
            ) 
            for oneData in [trainAudioDict, valiAudioDict, testAudioDict]
        ]

        # Step9. Model Training
        finalResult = list()
        if MODEL_MODE == "ML":
            inputFeatures = list(trainAudioDataFrame.columns)
            trainAudioDataFrame = pd.concat([
                trainAudioDataFrame.reset_index(drop = True), trainData
            ], axis = 1)
            valiAudioDataFrame = pd.concat([
                valiAudioDataFrame.reset_index(drop = True), valiData
            ], axis = 1)
            testAudioDataFrame = pd.concat([
                testAudioDataFrame.reset_index(drop = True), testData
            ], axis = 1)
            inputFeatures = [*inputFeatures, *metaDatainputFeatures]
            for oneFE in featureEngineerFlowList:
                totalResult = modelTrainingFlow(trainData = trainData,
                                                valiData = valiData,
                                                testData = testData,
                                                inputFeatures = inputFeatures, 
                                                target = mainTarget, 
                                                targetType = "classification",
                                                num_baggings = 1, 
                                                ml_methods = ["None"],
                                                hyperparameter_tuning_method = "TPESampler", 
                                                HTMetric = "cross_entropy", 
                                                thresholdMetric = "f1_1", 
                                                featureSelection = oneFE["FeatureSelection"],
                                                modelNameList = ["Random Forest with Entropy"], 
                                               fitBestModel = False,
                                               modelFilePath = os.path.join("./")).fit(permutationImportanceMethod=PERMUTATIONIMPORTANCE)
                basicInformation = {
                    **oneAudioFE,
                    **oneFE
                }
                finalResult.append({**basicInformation, **totalResult})
        elif MODEL_MODE == "SiameseNetwork_DL":
            pass
        else:
            print(trainAudioDataFrame.shape)
            if type(trainAudioDataFrame) == pd.DataFrame: trainAudioDataFrame = trainAudioDataFrame.values.tolist()
            if type(valiAudioDataFrame) == pd.DataFrame: valiAudioDataFrame = valiAudioDataFrame.values.tolist()
            if type(testAudioDataFrame) == pd.DataFrame: testAudioDataFrame = testAudioDataFrame.values.tolist()
            for mainModel, allModelParams in list(modelParamsList.items())[-1:]:
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
                        batch_size = BATCH_SIZE,
                        device = device,
                        modelName = mainModel,
                        trainBatchforEpoch = False,
                        balancedTrain = False, 
                        epochs = 2, 
                        modelParams = {
                            "numSequence": trainAudioDataFrame.__len__(), 
                            "numAudioFeature": 1, 
                            "numMetaDataFeature": metaDatainputFeatures[0].__len__(), 
                            **oneModelParams
                        },
                        lr = 1e-5,
                        refit = False,
                        finalModelExportPath = os.path.join(mainPath, "Final_Model")
                    ).fit()
                    basicInformation = {
                        "Main-Model": mainModel,
                        **oneModelParams, 
                        "batch_size": BATCH_SIZE,
                        **oneAudioFE
                    }
                    finalResult.append({
                        **basicInformation,
                        **totalResult["Evaluation"]
                    })
                    pd.DataFrame(finalResult).to_excel(os.path.join(mainPath, "result", "test.xlsx"), index = None)
        break