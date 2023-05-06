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
    "pretrainedModel": [
        {
            "numTarget": 5, 
            "pretrainedModelName": "resnet18",
            "numDecoderLayer": 1, 
            "numMetaDataLayer": 1, 
        },
    ],
    "onlyCNN": [
        {
            "numTarget": 5, 
            "numCNNLayer": 3,
            "numDecoderLayer": 1, 
            "numMetaDataLayer": 1, 
            "kernel_size": (15, 7),
            "numStride": 2
        },
    ],
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
    ],
    "CNNandAttentionModel": [
        {
            "numTarget": 5, 
            "numCNNLayer": 20,
            "numDecoderLayer": 1, 
            "numMetaDataLayer": 1, 
            "kernel_size": 15,
            "numStride": 1
        }
    ],
}