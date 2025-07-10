import pandas as pd
import json


        
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


