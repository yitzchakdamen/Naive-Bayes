import pandas as pd
import json
import csv



class UploadFile:
    pass

class ModelSystem:
    pass

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
    
    def _training(self) -> dict:
        model = {
            "yes":{},
            "no":{} ,
            "target_variable_Percentage_Yes": self.yes_count / self.num_target_variable,
            "target_variable_Percent_No": self.no_count  / self.num_target_variable}

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
        elif values: return dict(zip([col for col in self.modl["yes"] ], values))
        else: raise
        
    def activation(self) -> bool:
        yes = no = 1
        
        for category in self.parameters:
            yes *= self.modl["yes"][category][self.parameters[category]]
            no *= self.modl["no"][category][self.parameters[category]]
            
        yes *= self.modl["target_variable_Percentage_Yes"]
        no *= self.modl["target_variable_Percent_No"]
        return yes > no
    


class Clean:

    def __init__(self, df:pd.DataFrame, target_variable:str, str_yes:str, str_no:str, columns=None) -> None:
        self.target_variable = target_variable.strip()
        self.df:pd.DataFrame = df
        self.yes = str_yes
        self.no = str_no
        self.columns = columns
        
    def activation(self) -> dict:
        self._clean()
        return self._df_split()
        
    def _clean(self):
        if self.columns and len(self.columns) == self.df.shape[1]:
            self.df.columns = self.columns
        
        self.df.rename({self.target_variable:"target"})
        self.df['target'] = self.df.pop('target')
        self.df["target"] = self.df["target"].map({self.no: "no", self.yes: "yes"})
    
    def _df_split(self):
        return {
            "all": self.df,
            "75": self.df.iloc[:int(self.df.shape[0] * 0.25),:],
            "25": self.df.iloc[int(self.df.shape[0] * 0.25):,:]
        }
        

class model_testing:
    
    def __init__(self, df:pd.DataFrame, model:dict) -> None:
        self.df = df
        self.model = model
        
    def run(self):
        yes = no = 0
        for i in self.df.index:
            dict_val = self.df.iloc[i].to_dict()
            if Prediction(parameters=dict_val, modl=self.model).activation():
                if dict_val["target"] == 'yes': yes += 1
                else: no += 1
            else:
                if dict_val["target"] == 'no': yes += 1
                else: no += 1
                
        return {"num_yes": yes, "num_no": no, "result": yes / self.df.shape[0] }


# df = pd.read_csv("buy_computer_data.csv")

# model_training(df=df, target_variable="buys_computer").activation(name="buy_computer_data")

# with open("buy_computer_data.json", "r", encoding="utf-8") as file:
#     data = json.load(file)
    
# p = {"income":"high","student":"yes", "credit_rating":"fair", "age_group":"Very Young"} 
# r = Prediction(parameters=p, modl=data).activation()
# print(r)



# columns = [
#     "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
#     "gill-attachment", "gill-spacing", "gill-size", "gill-color",
#     "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
#     "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
#     "ring-number", "ring-type", "spore-print-color", "population", "habitat"
# ]

# df = pd.read_csv("agaricus-lepiota.csv", header=None, names=columns)
# df["class"] = df["class"].map({"p": "no", "e": "yes"})


# model_training(df=df, target_variable="class").activation(name="agaricus-lepiota")

# with open("agaricus-lepiota.json", "r", encoding="utf-8") as file:
#     data = json.load(file)

        
# keys = [
#     "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
#     "gill-attachment", "gill-spacing", "gill-size", "gill-color",
#     "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
#     "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
#     "ring-number", "ring-type", "spore-print-color", "population", "habitat"]

