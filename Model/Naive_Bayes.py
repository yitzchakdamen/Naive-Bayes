import pandas as pd
import json


        
class model_training:
    
    def __init__(self, df:pd.DataFrame, target_variable:str="target") -> None:
        self.target_variable: str = target_variable
        self.df: pd.DataFrame = df

    def _target_variable_definition(self) -> None:
        self.num_target_variable: int = self.df[self.target_variable].size
        self.list_target_variable = list(self.df[self.target_variable].unique())

    def _training(self) -> dict:
        model = {
            "columns": list(self.df.columns),
            "target_variable": self.list_target_variable,
            }
        
        for variable in self.list_target_variable:
            model[variable] = {}
            model[F"target_variable_Percent_{variable}"] = sum(self.df[self.target_variable] == variable) / self.num_target_variable

        for category in self.df:
            if category == self.target_variable:
                continue 
            
            for variable in self.list_target_variable:
                model[variable][category] = {}
            
            k = self.df[category].nunique()
            
            for parameter in self.df[category].unique():
                df_parameter: pd.Series = self.df[category] == parameter
                for target_variable in self.list_target_variable:
                    model[target_variable][category][parameter] = (sum((df_parameter) &  (self.df[self.target_variable] == target_variable)) + 1) / (sum(self.df[self.target_variable] == target_variable) + k)

        return model
    
    def activation(self, name:str="model") -> dict:
        self._target_variable_definition()
        model = self._training()
        with open(file= f"Files_model\\{name}.json", mode="w", encoding="utf-8") as file:
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
            
        yes *= self.modl["target_variable_Percent_yes"]
        no *= self.modl["target_variable_Percent_no"]
        
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


