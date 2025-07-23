import pandas as pd
import json
import requests
from io import BytesIO

class model_training:
    """
    Class for training a Naive Bayes model on a pandas DataFrame.
    """

    def __init__(self, df: pd.DataFrame, target_variable: str = "target") -> None:
        """
        Initialize with a DataFrame and the name of the target variable.
        """
        self.target_variable: str = target_variable
        self.df: pd.DataFrame = df

    def _target_variable_definition(self) -> None:
        """
        Define the target variable: count and unique values.
        """
        self.num_target_variable: int = self.df[self.target_variable].size
        self.list_target_variable = list(self.df[self.target_variable].unique())

    def _training(self) -> dict:
        """
        Train the Naive Bayes model and return it as a dictionary.
        """
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

    
    def activation(self,upload_url:str,  name: str = "model") -> dict:
        """
        Run the training, save the model as a JSON file.
        """
        self._target_variable_definition()
        model = self._training()
        self.saving_model_file(model, name, upload_url)
        return model



class Prediction:
    """
    Class for making predictions using a trained Naive Bayes model.
    """

    def __init__(self, modl: dict, parameters: dict = {}, values: list = []) -> None:
        """
        Initialize with a model and parameters or values for prediction.
        """
        self.modl = modl
        self.parameters: dict = self.preparation(parameters, values)
        
    def preparation(self, parameters: dict = {}, values: list = []) -> dict:
        """
        Prepare the parameters dictionary for prediction.
        """
        if parameters: return parameters
        elif values: return dict(zip([col for col in self.modl["columns"] ], values))
        else: raise
        
    def activation(self) -> dict:
        """
        Perform prediction and return the probabilistic result.
        """
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
        
        return {"preparation":yes > no, "yes":yes, "no":no}
    

class model_testing:
    """
    Class for evaluating the performance of a Naive Bayes model on a dataset.
    """

    def __init__(self, df: pd.DataFrame, model: dict) -> None:
        """
        Initialize with a DataFrame and a model for testing.
        """
        self.df = df
        self.model = model
        
    def run(self) -> dict:
        """
        Run model evaluation and return performance metrics (accuracy, precision, recall, F1).
        """
        TP = TN = FP = FN = 0

        for _, row in self.df.iterrows():
            dict_val = row.to_dict()
            pred = Prediction(parameters=dict_val, modl=self.model).activation()
            actual = dict_val["target"] == "yes"

            if pred["preparation"] and actual: TP += 1 
            elif pred["preparation"] and not actual: FP += 1  
            elif not pred["preparation"] and actual: FN += 1 
            elif not pred["preparation"] and not actual: TN += 1  
                
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
            "f1": f1,
            "TP": TP,
            "TN": TN,
            "FP": FP,
            "FN": FN
        }