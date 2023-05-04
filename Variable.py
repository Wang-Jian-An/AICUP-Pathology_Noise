import os

mainPath = r"D:/OneDrive - tmu.edu.tw/AI_Competition/多模態病理嗓音分類競賽"
trainVoicePath = "Training Dataset/training_voice_data/"
mainTarget = "Disease category"
metaDatainputFeatures = [
    "Sex",
    "Age",
    "Narrow pitch range",
    "Decreased volume",
    "Fatigue"
]

modelParamsList = {
    "CNNandRNNModel": [
        {
            "numTarget": 5, 
            "numCNNLayer": 2,
            "numRNNLayer": 2, 
            "numDecoderLayer": 1, 
            "numMetaDataLayer": 1, 
            "kernel_size": 15,
            "numStride": 2
        },
        {
            "numTarget": 5, 
            "numCNNLayer": 4,
            "numRNNLayer": 2, 
            "numDecoderLayer": 1, 
            "numMetaDataLayer": 1, 
            "kernel_size": 15,
            "numStride": 2
        },
        {
            "numTarget": 5, 
            "numCNNLayer": 6,
            "numRNNLayer": 2, 
            "numDecoderLayer": 1, 
            "numMetaDataLayer": 1, 
            "kernel_size": 15,
            "numStride": 2
        },
        {
            "numTarget": 5, 
            "numCNNLayer": 2,
            "numRNNLayer": 4, 
            "numDecoderLayer": 1, 
            "numMetaDataLayer": 1, 
            "kernel_size": 15,
            "numStride": 2
        },
        {
            "numTarget": 5, 
            "numCNNLayer": 2,
            "numRNNLayer": 6, 
            "numDecoderLayer": 1, 
            "numMetaDataLayer": 1, 
            "kernel_size": 15,
            "numStride": 2
        },
        {
            "numTarget": 5, 
            "numCNNLayer": 4,
            "numRNNLayer": 4, 
            "numDecoderLayer": 1, 
            "numMetaDataLayer": 1, 
            "kernel_size": 15,
            "numStride": 2
        },
    ]
}