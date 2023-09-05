import os, zipfile, librosa, json, tqdm
import numpy as np
import pandas as pd
from Variable import *
from sklearn.model_selection import train_test_split

def load_file(zipFilePath, train_or_predict = "train"):

    zipRawData = zipfile.ZipFile(zipFilePath)
    dataPath = [i for i in zipRawData.namelist() if "datalist" in i][0]    
    rawData = pd.read_csv(zipRawData.open(dataPath))
    if train_or_predict == "train":
        audioDict = {
            oneAudio: librosa.load(zipRawData.open(os.path.join(trainVoicePath, f"{oneAudio}.wav")))[0].tolist()
            for oneAudio in rawData["ID"].tolist()
        }
    else:
        audioDict = {
            oneAudio: librosa.load(zipRawData.open(os.path.join(publicPath, f"{oneAudio}.wav")))[0].tolist()
            for oneAudio in rawData["ID"].tolist()
        }

    # audioDict = {
    #     oneKey: oneAudio / np.abs(oneAudio)
    #     for oneKey, oneAudio in audioDict.items()
    # }
    return rawData, audioDict

def split_train_validation_test_data(IDList: list, targetList: list, stratify_baseline = mainTarget):

    """
    根據 ID 將資料切割成訓練、驗證與測試資料
    """
    print(IDList.__len__(), targetList.shape)
    tempData = pd.DataFrame({"ID": IDList, mainTarget: targetList})
    trainID, testID = train_test_split(tempData, test_size = 0.2, stratify = tempData[mainTarget], shuffle = True)
    trainID, valiID = train_test_split(trainID, test_size = 0.25, stratify = trainID[mainTarget], shuffle = True)
    return trainID["ID"].tolist(), valiID["ID"].tolist(), testID["ID"].tolist()

def onehotencoding_for_multiclass(oneSeries: pd.Series, import_json: str = None):

    """
    簡介：針對單一特徵進行 One-Hot Encoding
    """

    with open(import_json) as f:
        mappingList = json.load(f)
    featureName = oneSeries.name
    onehotencodingData = [mappingList[str(i)] for i in oneSeries.tolist()]
    return pd.DataFrame(onehotencodingData, columns = [f"{featureName}_{i}" for i in mappingList.keys()])

def FE_for_metaData(oneData, jsonPath):

    """
    簡介：將 metaData 中的類別變數轉換成 One-Hot Encoding、Age 轉換成 0 至 1 之間
    """
    oneData["Age"] = (oneData["Age"] - min(oneData["Age"])) / (max(oneData["Age"]) - min(oneData["Age"]))
    FESelectFeatures = [*metaDataNumericalFeatures, *metaDataTwoclassFeatures]
    FESelectFeatures = [*FESelectFeatures, mainTarget] if mainTarget in oneData.columns.tolist() else FESelectFeatures
    FEData = pd.concat([
        oneData[FESelectFeatures],
        *[
            onehotencoding_for_multiclass(oneSeries = oneData[oneFeature], import_json = os.path.join(jsonPath, f"{oneFeature}_mapping.json"))
            for oneFeature in metaDataMulticlassFeatures
        ]
    ], axis = 1)
    return FEData

def augmentationFlow(totalAudioData, totalMetaData, augmentMode):

    """
    簡介：所有資料增量的流程
    Params
    {
        totalAudioData: ,
        totalMetaData: ,
        augmentMode: 設定哪些類別需要資料增量，以及增量的模式。type: list
            {
                target: 欲資料增量的目標。,
                augmentMethod: 資料增量方法。
            }
    }
    """

    finishAugmentAudioData = dict()
    finishAugmentMetaData = list()
    totalMetaDataDict = totalMetaData.to_dict("records")
    for oneAugmentMode in augmentMode:
        oneTarget = oneAugmentMode["target"]
        wantAugmentMetaData = totalMetaData[totalMetaData[mainTarget] == oneTarget]
        wantAugmentID = wantAugmentMetaData["ID"].tolist()
        wantAugmentMetaData = wantAugmentMetaData.to_dict("records")
        wantAugmentAudio = {
            oneID: oneAudio
            for oneID, oneAudio in totalAudioData.items() if oneID in wantAugmentID
        }
        for (oneAudioID, oneAudio), oneMetaData in zip(wantAugmentAudio.items(), wantAugmentMetaData):
            augmentAudio = augmentOneData(oneAudio = oneAudio, augmentMethod = oneAugmentMode["augmentMethod"])
            finishAugmentAudioData = {
                **finishAugmentAudioData, 
                "{}_{}".format(oneAudioID, oneAugmentMode["augmentMethod"]): augmentAudio
            }
            oneMetaData["ID"] = "{}_{}".format(oneMetaData["ID"], oneAugmentMode["augmentMethod"])
            finishAugmentMetaData.append(oneMetaData)
    finishAugmentAudioData = {
        **totalAudioData,
        **finishAugmentAudioData
    }
    finishAugmentMetaData = [*totalMetaDataDict, *finishAugmentMetaData]
    return finishAugmentAudioData, pd.DataFrame(finishAugmentMetaData)

def augmentOneData(
        oneAudio: np.ndarray,
        augmentMethod = "shift_1"
):

    """
    簡介：一筆資料增量的流程
    """

    # print(augmentMethod)
    assert "shift" in augmentMethod or "noise" in augmentMethod or type(augmentMethod) == list, "There is no valid method for audio augmentation. "

    # Augment Audio Data
    if type(augmentMethod) == list:
        shiftMethod = [i for i in augmentMethod if "shift" in i]
        noiseMethod = [i for i in augmentMethod if "noise" in i]
        seconds = eval(shiftMethod[0].split("_")[-1])
        noiseFactor = eval(noiseMethod[0].split("_")[-1])
        augmentAudio = audioShift(oneAudio = oneAudio, seconds = seconds)
        augmentAudio = addAudioNoise(oneAudio = augmentAudio, noiseFactor = noiseFactor)
    elif "shift" in augmentMethod:
        seconds = eval(augmentMethod.split("_")[-1])
        augmentAudio = audioShift(oneAudio = oneAudio, seconds = seconds)
    elif "noise" in augmentMethod:
        noiseFactor = eval(augmentMethod.split("_")[-1])
        augmentAudio = addAudioNoise(oneAudio = oneAudio, noiseFactor = noiseFactor)
    return augmentAudio

def audioShift(oneAudio, seconds):

    """
    簡介：將音訊資訊往左或往右移動
    """

    shiftNum = int(22050 * seconds)
    return np.roll(oneAudio, shiftNum)

def addAudioNoise(oneAudio, noiseFactor):

    """
    簡介：將音訊資料加入白噪音。
    """

    noise = np.random.randn(oneAudio.shape[0]) if type(oneAudio) == np.ndarray else np.random.randn(oneAudio.__len__())
    return oneAudio + noiseFactor * noise

def normalizeAudioLength(
    audioDict: dict,
    fixLength: int or float,
    sr: int = 22050
):
    audioDict = {
        oneID: normalizeOneAudioLength(oneAudio = oneAudio, fixLength = fixLength)
        for oneID, oneAudio in tqdm.tqdm(list(audioDict.items()))
    }
    return audioDict

def normalizeOneAudioLength(
    oneAudio: list or np.ndarray,
    fixLength: int or float,
    sr = 22050
):
    if type(oneAudio) == np.ndarray:
        oneAudio = oneAudio.tolist()
    
    targetDataLength = int(sr * fixLength)
    halfLength = oneAudio.__len__()
    if oneAudio.__len__() < targetDataLength:
        oneNormalizeAudio = librosa.util.fix_length(
            data = np.array(oneAudio),
            size = targetDataLength,
            mode = "reflect"
        )
    elif oneAudio.__len__() > targetDataLength:
        differenceHalf = oneAudio.__len__() - targetDataLength // 2
        oneNormalizeAudio = oneAudio[halfLength-differenceHalf:-halfLength+differenceHalf]
        if oneNormalizeAudio.__len__() != targetDataLength:
            difference = oneNormalizeAudio.__len__() - targetDataLength 
            if difference > 0:
                oneNormalizeAudio = [*oneNormalizeAudio, *oneAudio[-differenceHalf:-differenceHalf+difference]]
            else:
                oneNormalizeAudio = [*oneAudio[differenceHalf+difference: differenceHalf], *oneNormalizeAudio]
    else:
        oneNormalizeAudio = oneAudio.copy()
    return oneNormalizeAudio if type(oneNormalizeAudio) else oneNormalizeAudio.tolist()

def removeHeadTailAudio(
    oneAudio: np.ndarray or list,
    removeOneSideTime: int or float,
    sr: int = 22050
):
    removeDataLength = int(sr * removeOneSideTime)
    removeAudio = oneAudio[removeDataLength:-removeDataLength]
    return removeAudio

def audioMelSpectrogram(
    oneAudio: np.ndarray,
    n_mels = 128,
    hop_length = 512,
    **kwargs
):
    S = librosa.feature.melspectrogram(
        y = oneAudio,
        n_mels = n_mels,
        hop_length = hop_length
    )
    return S

def audioMFCC(
    oneAudio: np.ndarray,
    n_mfcc = 40,
    hop_length = 512,
    **kwargs
):
    S = librosa.feature.mfcc(
        y = oneAudio,
        n_mfcc = n_mfcc,
        hop_length = hop_length
    )
    return S

def oneAudioFeatureEngineer(
    oneAudio,
    audioTransformMethod,
    statistics,
    domain,
    **kwargs
):
    if type(oneAudio) == list: oneAudio = np.array(oneAudio)    
    if audioTransformMethod == "melspectrogram":
        transformedAudio = audioMelSpectrogram(oneAudio, **kwargs)
    elif audioTransformMethod == "mfcc":
        transformedAudio = audioMFCC(oneAudio, **kwargs)
    if statistics.__len__() > 0:
        FEOutput = list()
        featureList = list()
        if "time" in domain:
            if "mean" in statistics:
                statisticsFeatures = np.mean(transformedAudio, axis = 1).tolist()
                FEOutput.extend(statisticsFeatures)
                featureList.extend([f"{audioTransformMethod}_time_mean_{i}" for i in range(1, len(statisticsFeatures)+1)])
            if "std" in statistics:
                statisticsFeatures = np.std(transformedAudio, axis = 1).tolist()
                FEOutput.extend(statisticsFeatures)
                featureList.extend([f"{audioTransformMethod}_time_std_{i}" for i in range(1, len(statisticsFeatures)+1)])
            if "median" in statistics:
                statisticsFeatures = np.median(transformedAudio, axis = 1).tolist()
                FEOutput.extend(statisticsFeatures)
                featureList.extend([f"{audioTransformMethod}_time_median_{i}" for i in range(1, len(statisticsFeatures)+1)])
            if "maximum" in statistics:
                statisticsFeatures = np.maximum(transformedAudio, axis = 1).tolist()
                FEOutput.extend(statisticsFeatures)
                featureList.extend([f"{audioTransformMethod}_time_maximum_{i}" for i in range(1, len(statisticsFeatures)+1)])
        if "frequency" in domain:
            if "mean" in statistics:
                statisticsFeatures = np.mean(transformedAudio, axis = 0).tolist()
                FEOutput.extend(statisticsFeatures)
                featureList.extend([f"{audioTransformMethod}_frequency_mean_{i}" for i in range(1, len(statisticsFeatures)+1)])
            if "std" in statistics:
                statisticsFeatures = np.std(transformedAudio, axis = 0).tolist()
                FEOutput.extend(statisticsFeatures)
                featureList.extend([f"{audioTransformMethod}_frequency_std_{i}" for i in range(1, len(statisticsFeatures)+1)])
            if "median" in statistics:
                statisticsFeatures = np.median(transformedAudio, axis = 0).tolist()
                FEOutput.extend(statisticsFeatures)
                featureList.extend([f"{audioTransformMethod}_frequency_median_{i}" for i in range(1, len(statisticsFeatures)+1)])
            if "maximum" in statistics:
                statisticsFeatures = np.maximum(transformedAudio, axis = 0).tolist()
                FEOutput.extend(statisticsFeatures)
                featureList.extend([f"{audioTransformMethod}_frequency_maximum_{i}" for i in range(1, len(statisticsFeatures)+1)])
        FEOutput = pd.Series(FEOutput, index = featureList) 
    else:
        FEOutput = transformedAudio.tolist()
    return FEOutput

def audioFeatureEngineer(
    oneAudioDict, 
    audioTransformMethod, 
    statistics, 
    domain = None,
    **kwargs
):

    """
    簡介：針對音訊資料進行特徵工程（Hints: 每個音訊已經處理成統一長度）
    Params: 
        oneAudioDict: 由資料ID與音訊資料組合而成的Dictionary
        audioTransformMethod: 
        statistics: 
        domain: 
        **kwargs
            n_mels: 
            n_mfcc: 
            hop_length: 
    Return:
    
    """
    
    assert (statistics.__len__() > 0 and domain is not None) or (statistics.__len__() == 0 or domain is None), "domain must be time or frequency when there have statistics."
    FEOutput = {
        oneID: oneAudioFeatureEngineer(
            oneAudio = oneAudio, 
            audioTransformMethod = audioTransformMethod, 
            statistics = statistics, 
            domain = domain, 
            **kwargs
        )
        for oneID, oneAudio in oneAudioDict.items()
    }
    if statistics.__len__() > 0:
        return pd.DataFrame(FEOutput).T
    else:
        return FEOutput