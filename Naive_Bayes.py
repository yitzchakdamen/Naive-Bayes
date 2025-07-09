import pandas as pd
from sklearn.model_selection import train_test_split
import json
from typing import cast
import streamlit as st

class UploadData:
    
    def _upload_csv(self, file_pat):
        if isinstance(file_pat, str):
            return pd.read_csv(file_pat)
        else:
            # תמיכה ב־UploadedFile או קובץ פתוח
            return pd.read_csv(file_pat)
    
    def _upload_json(self, file_pat):
        try:
            if isinstance(file_pat, str):
                with open(file_pat, "r", encoding="utf-8") as file:
                    return json.load(file)
            else:
                # תמיכה ב־UploadedFile או קובץ פתוח
                return json.load(file_pat)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return {}

    @staticmethod
    def upload(file):
        # קובץ כמחרוזת
        if isinstance(file, str):
            if file.endswith(".csv"):
                return UploadData()._upload_csv(file)
            elif file.endswith(".json"):
                return UploadData()._upload_json(file)
        
        # קובץ של Streamlit או אובייקט עם .name (כמו open(...))
        if hasattr(file, "name"):
            name = file.name.lower()
            if name.endswith(".csv"):
                return UploadData()._upload_csv(file)
            elif name.endswith(".json"):
                return UploadData()._upload_json(file)

        raise ValueError("Unsupported file type. Please upload a CSV or JSON file.")



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
            training_all = model_training(df=self.data_all).activation(f"{name}_training_all")
            training_75 = model_training(df=self.data_train_df).activation(f"{name}_training_75")
    
    def testing(self):
        if self.upload_prepared() and hasattr(self, "nmodel"):
            training = model_testing(df=self.data_test_df, model=self.nmodel).run()
            return training

    def prediction(self, values):
        if hasattr(self, "nmodel"):
            return Prediction( modl=self.nmodel, values=values).activation()


        
class model_training:
    
    def __init__(self, df:pd.DataFrame, target_variable:str="target", str_yes:str="yes", str_no:str="no") -> None:
        self.target_variable = target_variable
        self.df = df
        self.yes = str_yes
        self.no = str_no

    def _target_variable_definition(self) -> None:
        self.num_target_variable = self.df[self.target_variable].size
        self.yes_count = sum(self.df[self.target_variable] == self.yes)
        self.no_count = sum(self.df[self.target_variable] == self.no)
        # self.a = self.df[self.target_variable].unique()[0]
    
    def _training(self) -> dict:
        model = {
            "yes":{},
            "no":{} ,
            "target_variable_Percentage_Yes": self.yes_count / self.num_target_variable,
            "target_variable_Percent_No": self.no_count  / self.num_target_variable,
            "columns": list(self.df.columns)
            }

        for category in self.df:
            if category == self.target_variable:
                continue 
            
            model["yes"][category], model["no"][category]= {}, {}
            
            k = self.df[category].nunique()
            
            for parameter in self.df[category].unique():
                df_parameter: pd.Series = self.df[category] == parameter
                model["yes"][category][parameter] = (sum((df_parameter) &  (self.df[self.target_variable] == self.yes)) + 1) / (self.yes_count + k)
                model["no"][category][parameter] = (sum((df_parameter) &  (self.df[self.target_variable] == self.no)) + 1) / (self.no_count + k)
        return model
    
    def activation(self, name:str="model") -> dict:
        self._target_variable_definition()
        model = self._training()
        with open(file= f"{name}.json", mode="w", encoding="utf-8") as file:
            json.dump(obj=model, fp=file, indent=4)
        
        return model



class Prediction:
    
    def __init__(self, modl:dict, parameters: dict={}, values:list=[]) -> None:
        self.modl = modl
        self.parameters: dict = self.preparation(parameters, values)
        
    def preparation(self, parameters: dict={}, values:list=[]) -> dict:
        if parameters: return parameters
        elif values: return dict(zip([col for col in self.modl["columns"] ], values))
        else: raise
        
    def activation(self) -> bool:
        yes = no = 1
        for category in self.parameters:
            if category == "target":
                continue
            try:
                yes *= self.modl["yes"][category][self.parameters[category]]
                no *= self.modl["no"][category][self.parameters[category]]
            except:
                pass
            
        yes *= self.modl["target_variable_Percentage_Yes"]
        no *= self.modl["target_variable_Percent_No"]
        return yes > no
    

class Clean:

    def __init__(self, df:pd.DataFrame, target_variable:str, str_yes:str, str_no:str, columns=None) -> None:
        self.target_variable = target_variable.strip()
        self.df:pd.DataFrame = df
        self.yes = str_yes.strip()
        self.no = str_no.strip()
        self.columns = columns
        
    def activation(self) -> dict:
        if self._clean():
            return self._df_split()
        else: return {}
        
    def _clean(self):
        self.df = self.df.fillna("missing")
        
        if self.columns and len(self.columns) == self.df.shape[1]:
            self.df.columns = self.columns
            
        if self.target_variable in self.df.columns:
            self.df.rename(columns={self.target_variable:"target"}, inplace=True)
            self.df['target'] = self.df.pop('target')
            
            if sorted(self.df["target"].unique().tolist()) == sorted([self.no, self.yes]):
                self.df["target"] = self.df["target"].map({self.no: "no", self.yes: "yes"})
                return True
            
        return False
    
    def _df_split(self):
        train_df, test_df = train_test_split(
            self.df,
            test_size=0.25,         # 25% לבדיקה
            random_state=42,        # כדי שהתוצאה תהיה תמיד זהה אם מריצים שוב
            shuffle=True,           # ערבוב הנתונים לפני הפיצול
            stratify=self.df["target"]  # לשמור על יחס yes/no גם באימון וגם בבדיקה
        )
        return {
            "all": self.df,         # כל הדאטה (לא חובה להשתמש בזה)
            "train_df": train_df,         # הדאטה לאימון (75%)
            "test_df": test_df           # הדאטה לבדיקה (25%)
        }

class model_testing:
    
    def __init__(self, df:pd.DataFrame, model:dict) -> None:
        self.df = df
        self.model = model
        
    def run(self):    
        TP = TN = FP = FN = 0

        for _, row in self.df.iterrows():
            dict_val = row.to_dict()
            pred = Prediction(parameters=dict_val, modl=self.model).activation()
            actual = dict_val["target"] == "yes"

            if pred and actual:
                TP += 1  # True Positive
            elif pred and not actual:
                FP += 1  # False Positive
            elif not pred and actual:
                FN += 1  # False Negative
            elif not pred and not actual:
                TN += 1  # True Negative
                
        accuracy = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return {
            "num_yes": TP + TN,
            "num_no": FP + FN,
            "result": accuracy * 100,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


