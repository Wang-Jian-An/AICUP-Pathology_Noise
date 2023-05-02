import os, json, gzip, pickle
from datetime import datetime
import pandas as pd
from functions import *

if __name__ == "__main__":
    
    # Step1. 讀取資料
    rawData, audioDict = load_file(r"D:\OneDrive - tmu.edu.tw\AI_Competition\多模態病理嗓音分類競賽\Training Dataset.zip")

    # Step2. 讀取 Target Mapping
    with gzip.open(os.path.join(mainPath, "Target_Mapping.json.gz"), "wt") as f:
        target_mapping = json.load(f)
    inverse_target_mapping = {
        index: i for i, index in target_mapping.items()
    }

    # Step3. 配對每個資料ID與音訊檔案
    predAudioDict = {
        oneID: audioDict[oneID] for oneID in rawData["ID"]
    }

    # Step4. 特徵工程
    predAudioDict = feature_engineering(oneAudioDict = predAudioDict, FEID = 1)

    # Step5. 讀取模型
    modelFileName = ""
    modelPath = os.path.join(mainPath, "Final_Model", modelFileName)
    if ".gzip" in modelFileName:
        with gzip.GzipFile(modelPath, "r") as f:
            model = pickle.load(f)
    elif ".pkl" in modelFileName:
        pass
    elif ".pt" in modelFileName or ".pth" in modelFileName:
        pass

    # Step6. 模型預測
    if ".gzip" in modelFileName or ".pkl" in modelFileName:
        yhat = model.predict(predAudioDict)
    else:
        yhat = model(predAudioDict)
    
    # Step7. 儲存結果
    yhatData = {
        oneKey: oneYhat for oneKey, oneYhat in zip(rawData["ID"], yhat)
    }
    pd.DataFrame(yhatData).to_csv(os.path.join(mainPath, "Submission", "Submission_{}".format(str(datetime.today()).split(" ")[0])))