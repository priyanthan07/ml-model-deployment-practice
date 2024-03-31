import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation, DataTransformationConfig
import pandas as pd

from dataclasses import dataclass
from sklearn.model_selection import train_test_split


# create the config to provide all the inputs for DataIngestion
@dataclass  # this decorator is used to define variables without __init__
class DataIngestionConfig:
    train_data_path:str = os.path.join("artifacts","train.csv")
    test_data_path:str = os.path.join("artifacts","test.csv")
    raw_data_path:str = os.path.join("artifacts","raw.csv")


class DataIngestion:
    def __init__(self):
        # define variables
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered into data ingestion method or componenets")
        try:
            # reading a csv datafile
            df = pd.read_csv("notebooks\data\stud.csv")
            logging.info("Read the dataset as framework")

            # make directory 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # save as .csv file
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("train test split initiated")

            # split the dataset in to train set and test set
            train_set, test_set = train_test_split(df, test_size= 42, random_state=42)

            # save as .csv file
            df.to_csv(self.ingestion_config.train_data_path, index= 42, header= True)
            df.to_csv(self.ingestion_config.test_data_path, index= 42, header= True)

            logging.info("data ingestion completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e, sys)
        

if __name__ == "__main__":
    data_injestion = DataIngestion()
    train_data, test_data = data_injestion.initiate_data_ingestion()

    data_transform = DataTransformation()
    train_arr, test_arr,_ = data_transform.initiate_data_transformation(train_data, test_data)
