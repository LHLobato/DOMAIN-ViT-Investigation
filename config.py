sample_feature_extraction_type = ["TFIDV", "BERT-PREPROCESSED", "TFIVD-fullfeatures"]
current_dataset = sample_feature_extraction_type[1]
DATA_DIR_0 = {
    "test":f"datasets/{current_dataset}/GASF0/test", 
    "train":f"datasets/{current_dataset}/GASF0/train",
    "val":f"datasets/{current_dataset}/GASF0/val",
}

DATA_DIR_1 = {
    "test":f"datasets/{current_dataset}/GASF1/test", 
    "train":f"datasets/{current_dataset}/GASF1/train",
    "val":f"datasets/{current_dataset}/GASF1/val",
}
DATA_DIR_2 = {
    "test":f"datasets/{current_dataset}/GASF2/test", 
    "train":f"datasets/{current_dataset}/GASF2/train",
    "val":f"datasets/{current_dataset}/GASF2/val",
}

DATA_DIR_3 = {
    "test":f"datasets/{current_dataset}/GADF0/test", 
    "train":f"datasets/{current_dataset}/GADF0/train",
    "val":f"datasets/{current_dataset}/GADF0/val",
}
DATA_DIR_4 = {
    "test":f"datasets/{current_dataset}/GADF1/test", 
    "train":f"datasets/{current_dataset}/GADF1/train",
    "val":f"datasets/{current_dataset}/GADF1/val",
}
DATA_DIR_5 = {
    "test":f"datasets/{current_dataset}/GADF2/test", 
    "train":f"datasets/{current_dataset}/GADF2/train",
    "val":f"datasets/{current_dataset}/GADF2/val",
}

DATA_DIR_6={
    "test":f"datasets/{current_dataset}/RPLOT0/test", 
    "train":f"datasets/{current_dataset}/RPLOT0/train",
    "val":f"datasets/{current_dataset}/RPLOT0/val",
}
DATA_DIR_7={
    "test":f"datasets/{current_dataset}/RPLOT1/test", 
    "train":f"datasets/{current_dataset}/RPLOT1/train",
    "val":f"datasets/{current_dataset}/RPLOT1/val",
}
DATA_DIR_8={
    "test":f"datasets/{current_dataset}/RPLOT2/test", 
    "train":f"datasets/{current_dataset}/RPLOT2/train",
    "val":f"datasets/{current_dataset}/RPLOT2/val",
}


DATA_DIRS_GASF = [DATA_DIR_0,DATA_DIR_1,DATA_DIR_2]
DATA_DIRS_GADF = [DATA_DIR_3,DATA_DIR_4,DATA_DIR_5]
DATA_DIRS_RPLOT = [DATA_DIR_6,DATA_DIR_7,DATA_DIR_8]

DATA_DIRS = [DATA_DIRS_GASF, DATA_DIRS_GADF, DATA_DIRS_RPLOT]

BATCH_SIZE = 128
P = 0.2
WEIGHT_DECAY = 0.05
NUM_SAMPLES = 61574
NUM_EPOCHS = 100
RESOLUTION = 24
NUM_DATASETS = 3
UNFROZEN_LAYERS = [0,4]
