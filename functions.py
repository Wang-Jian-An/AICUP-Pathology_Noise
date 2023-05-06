import os, zipfile, librosa
import numpy as np
import pandas as pd
from Variable import *
from sklearn.model_selection import train_test_split

def load_file(zipFilePath):

    zipRawData = zipfile.ZipFile(zipFilePath)
    dataPath = [i for i in zipRawData.namelist() if ".csv" in i][0]    
    rawData = pd.read_csv(zipRawData.open(dataPath)).iloc[:100, :]
    audioDict = {
        oneAudio: librosa.load(zipRawData.open(os.path.join(trainVoicePath, f"{oneAudio}.wav")))[0].tolist()
        for oneAudio in rawData["ID"].tolist()[:100]
    }
    return rawData, audioDict

def split_train_validation_test_data(IDList: list, targetList: list, stratify_baseline = mainTarget):

    """
    根據 ID 將資料切割成訓練、驗證與測試資料
    """
    tempData = pd.DataFrame({"ID": IDList, mainTarget: targetList})
    trainID, testID = train_test_split(tempData, test_size = 0.2, stratify = tempData[mainTarget], shuffle = True)
    trainID, valiID = train_test_split(trainID, test_size = 0.25, stratify = trainID[mainTarget], shuffle = True)
    return trainID["ID"], valiID["ID"], testID["ID"]

def stft_and_statistics(oneAudioData):
    
    """
    簡介：將音訊資料提取 STFT，並依照時間順序取出各式統計量
    輸入；
        oneAudioData：音訊資料。type: np.ndarray
    輸出：
        featuredAudioData：特徵萃取後之音訊資料。type: np.ndarray
    """
    oneAudioData = np.array(oneAudioData) if type(oneAudioData) == list else oneAudioData
    D_stft = librosa.stft(oneAudioData)
    D_stft_db = librosa.amplitude_to_db(np.abs(D_stft))

    D_stft_db_mean = np.mean(D_stft_db, axis = 1)
    D_stft_db_std = np.std(D_stft_db, axis = 1)
    return np.array(D_stft_db_mean.tolist() + D_stft_db_std.tolist())


def feature_engineering(oneAudioDict, FEID = 1):

    """
    簡介：建立特徵工程流程
    Params: 
        oneAudioDict: 由資料ID與音訊資料組合而成的Dictionary
        FEID: 特徵工程方法之ID
    Return:
    

    FEID = 1：透過 STFT 搭配統計量計算出 2D 資料結構
    FEID = 2：透過 Mel Spectrogram 搭配統計量計算出 2D 資料結構
    FEID = 3：將 STST、Mel Spectrogram 結果結合起來計算出 2D 資料結構
    FEID = 4：透過 STFT 搭配 sequence padding 技巧求得 3D 資料結構
    FEID = 5：透過 Mel Spectrogram 搭配 sequence padding 技巧求得 3D 資料結構
    FEID = 6：透過 MFCC 搭配 sequence padding 技巧求得 3D 資料結構
    """

    if FEID == 1:
        oneAudioDict = {
            oneKey: stft_and_statistics(oneAudioData = oneValue)
            for oneKey, oneValue in oneAudioDict.items()
        }
        oneAudioDataFrame = pd.DataFrame(list(oneAudioDict.values()))
        FEOutput = pd.DataFrame(oneAudioDataFrame.values, columns = [f"F_{j}" for j in oneAudioDataFrame.columns])
        pass
    elif FEID == 2:
        pass
    elif FEID == 3:
        pass
    elif FEID == 4:
        longest_audio_length = max([i.__len__() for i in oneAudioDict.values()])
        oneAudioDict = {
            oneKey: oneValue + ([0] * (longest_audio_length - oneValue.__len__()))
            for oneKey, oneValue in oneAudioDict.items()
        }
        FEOutput = {
            oneKey: librosa.amplitude_to_db(np.abs(librosa.stft( np.array(oneValue) )))
            for oneKey, oneValue in oneAudioDict.items()
        }
        FEOutput = list(FEOutput.values())
    elif FEID == 5:
        FEOutput = {
            oneKey: librosa.power_to_db( librosa.feature.melspectrogram( np.array(oneValue) ))
            for oneKey, oneValue in oneAudioDict.items()
        }
        FEOutput = list(FEOutput.values())
    elif FEID == 6:
        FEOutput = {
            oneKey: librosa.feature.mfcc( np.array(oneValue) )
            for oneKey, oneValue in oneAudioDict.items()
        }
        FEOutput = list(FEOutput.values())
    return FEOutput