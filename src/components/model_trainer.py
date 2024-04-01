import os
import sys
from dataclasses import dataclass

import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor,)
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("split train and test input data")
            x_train, y_train, x_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest" : RandomForestRegressor(),
                "Decision Tree" :DecisionTreeRegressor(),
                "Gradient Boosting" :GradientBoostingRegressor(),
                "Linear Regression" : LinearRegression(),
                "K-Neighbors Regressor" : KNeighborsRegressor(),
                "XGBClassifier" : XGBRegressor(),
                "CatBoosting Regressor" : CatBoostRegressor(),
                "AdaBoost Regressor" : AdaBoostRegressor()
            }
            model_report:dict = evaluate_models(
                X_train = x_train, y_train = y_train, X_test=x_test, y_test = y_test, models = models
            )

            # highest model score in the report
            highest_score = max(sorted(model_report.values()))

            best_model_name = list(models.keys())[
                list(model_report.values()).index(highest_score)
            ]

            best_model = models[best_model_name]

            if highest_score < 0.6:
                raise CustomException("No best model found")
            
            logging.info("best model found on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )

            prediction = best_model.predict(x_test)

            r2score = r2_score(y_test, prediction)

            return r2score  



        except Exception as e:
            raise CustomException(e, sys)
