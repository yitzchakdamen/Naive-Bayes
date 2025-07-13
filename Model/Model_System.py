from Model.Naive_Bayes import model_training, model_testing, Prediction
from Model.model_info import ModelInfo
from typing import cast
from Model.Upload import UploadData
from Model.Clean import Clean
import pandas as pd
import os


class ModelSystem:
    
    def __init__(self) -> None:
        pass
        
    def upload_data(self, file, target_variable:str, str_yes:str, str_no:str, columns=None):
        data:pd.DataFrame = cast(pd.DataFrame, UploadData.upload(file))
        self.data_dict: dict = Clean(df=data, target_variable=target_variable, str_yes=str_yes, str_no=str_no, columns=columns).activation()
    
    def upload_model(self, file):
        data:dict = cast(dict, UploadData.upload(file))
        self.nmodel = data

    def upload_prepared(self):
        if hasattr(self, "data_dict" ):
            if self.data_dict:
                self.data_all = self.data_dict["all"]
                self.data_train_df = self.data_dict["train_df"]
                self.data_test_df = self.data_dict["test_df"]
                return True
        return False
    
    def training(self, name:str):
        if self.upload_prepared():
            model_training(df=self.data_all).activation(f"{name}_training_all")
            model_training(df=self.data_train_df).activation(f"{name}_training_75")
    
    def testing(self):
        if self.upload_prepared() and hasattr(self, "nmodel"):
            training = model_testing(df=self.data_test_df, model=self.nmodel).run()
            return training

    def prediction(self, values=[], parameters={}):
        if hasattr(self, "nmodel"):
            return Prediction( modl=self.nmodel, values=values, parameters=parameters).activation()

    def get_info(self):
        model_info = ModelInfo()
        return [
            model_info.get_model_info(cast(dict,UploadData.upload(os.path.join(model_info.MODELS_DIR, file)))) 
            for file in os.listdir(model_info.MODELS_DIR) 
            if file.endswith(".json")]

