import os 
os.environ['MPLCONFIGDIR'] = '.'
import configparser
import logging
import pathlib as path
from collections import defaultdict
import numpy as np
import pandas as pd
import glob
from idao.dataloader import DataGenerator
from tensorflow.keras import models
import tensorflow as tf
dict_pred = defaultdict(list)


def make_csv(mode, data_generator, model_path, prediction_files, cfg):
    logging.info("Loading checkpoint")
    model = models.load_model(model_path)

    if mode == "classification":
        logging.info("Classification model loaded")
    else:
        logging.info("Regression model loaded")
      
    preds = model.predict(data_generator, verbose = 1).reshape(-1)

    for i, pred in enumerate(preds):
        if mode == "classification":
            dict_pred["id"].append(prediction_files[i].split('/')[-1].split(".")[0])
            output = pred
            dict_pred["classification_predictions"].append(output)

        else:
            output = pred
            dict_pred["regression_predictions"].append(output)


def main(cfg):
    PATH = path.Path(cfg["DATA"]["DatasetPath"])
    PUBLIC_PATH = PATH / 'public_test'
    PRIVATE_PATH = PATH / 'private_test'
    

    image_paths = {
        'public': glob.glob(str(PUBLIC_PATH / '*.png')),
        'private': glob.glob(str(PRIVATE_PATH / '*.png'))
    }

    dataloaders = {
        'public': DataGenerator(images=image_paths['public'], batch_size=1, shuffle = False),
        'private': DataGenerator(images=image_paths['private'], batch_size=1, shuffle = False)
    }

    for dl_name in ['public', 'private']:
        for mode in ["regression", 'classification']:
            if mode == "classification":
                model_path = cfg["REPORT"]["ClassificationCheckpoint"]
            else:
                model_path = cfg["REPORT"]["RegressionCheckpoint"]
            make_csv(mode = mode, data_generator = dataloaders[dl_name], model_path = model_path, prediction_files = image_paths[dl_name], cfg=cfg)
    
    data_frame = pd.DataFrame(
        dict_pred,
        columns=["id", "classification_predictions", "regression_predictions"],
    )
    data_frame.to_csv("submission.csv", index=False, header=True)


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    config = configparser.ConfigParser()
    config.read("./config.ini")
    main(cfg=config)
