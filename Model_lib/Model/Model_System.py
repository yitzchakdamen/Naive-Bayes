from .Naive_Bayes import model_training, model_testing, Prediction
from .Model_info import ModelInfo
from typing import cast
from .Upload import Upload
from .Clean import Clean
import pandas as pd
import os
import requests
import json
from io import BytesIO


class ModelSystem:
    
    MODELS_DIR = "./models"
    UPLOAD_URL = "http://main:8888/upload/"
    GET_FILE_URL = "http://main:8888/download/"
    GET_INFO_URL = "http://main:8888/models_info/"

    def __init__(self) -> None:
        pass
        
    def upload_data(self, file, target_variable:str, str_yes:str, str_no:str, columns=None):
        data:pd.DataFrame = cast(pd.DataFrame, Upload.upload(file))
        self.data_dict: dict = Clean(df=data, target_variable=target_variable, str_yes=str_yes, str_no=str_no, columns=columns).activation()
    
    def upload_model(self, file):
        data:dict = cast(dict, Upload.upload(file))
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
            training_all = model_training(df=self.data_all).activation()
            training_75 = model_training(df=self.data_train_df).activation()
            self.saving_model_file(training_all, name=f"{name}_training_all", upload_url=self.UPLOAD_URL)
            self.saving_model_file(training_75, name=f"{name}_training_75", upload_url=self.UPLOAD_URL)
            return {"training_all":training_all, "training_75": training_75}

    def testing(self):
        if self.upload_prepared() and hasattr(self, "nmodel"):
            training = model_testing(df=self.data_test_df, model=self.nmodel).run()
            return training

    def prediction(self, values=[], parameters={}):
        if hasattr(self, "nmodel"):
            return Prediction( modl=self.nmodel, values=values, parameters=parameters).activation()

    def get_info(self):
        model_info = ModelInfo()
        list_model_info = [] 
        
        for file in os.listdir(self.MODELS_DIR):
            if file.endswith(".json"):
                model:dict = cast(dict,Upload.upload(os.path.join(self.MODELS_DIR, file)))
                info = model_info.get_model_info(model)
                list_model_info.append(info)
        
        return list_model_info
    
    def saving_model_file(self, model, name: str, upload_url: str):
        model["name"] = name
        json_bytes = json.dumps(model, indent=4).encode("utf-8")
        files = {"file": (f"{name}.json", BytesIO(json_bytes), "application/json")}

        try:
            response = requests.post(upload_url, files=files)
            response.raise_for_status()
            print(f"Тhe model was successfully uploaded: {response.json()}")
            return response.json()
        except requests.RequestException as e:
            print(f"Error sending model to server: {e}")
            return {"error": str(e)}
            