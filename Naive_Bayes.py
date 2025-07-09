import pandas as pd
from sklearn.model_selection import train_test_split
import json



class UploadData:
    
    def _upload_csv(self, file_pat:str):
        return pd.read_csv(file_pat)
    
    def _upload_json(self, file_pat:str):
        with open(file_pat, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data
    
    @staticmethod
    def upload(file:str):
        if ".csv" in file: return UploadData()._upload_csv(file)
        if ".json" in file: return UploadData()._upload_json(file)


class ModelSystem:
    
    def __init__(self) -> None:
        pass
        
    def upload_data(self, data_pat:str, target_variable:str, str_yes:str, str_no:str, columns=None):
        data:pd.DataFrame = UploadData.upload(data_pat)
        self.data_dict: dict = Clean(df=data, target_variable=target_variable, str_yes=str_yes, str_no=str_no, columns=columns).activation()
        self.data_all = self.data_dict["all"]
        self.data_train_df = self.data_dict["train_df"]
        self.data_test_df = self.data_dict["test_df"]
    
    def training(self):
        if hasattr(self, "data_dict"):
            training_all = model_training(df=self.data_all).activation("training_all")
            training_75 = model_training(df=self.data_train_df).activation("training_75")
    
    def testing(self, model_pat):
        if hasattr(self, "data_dict"):
            model: dict = UploadData.upload(model_pat)
            training = model_testing(df=self.data_test_df, model=model).run()
            self.print_results( training=training)
        
    def print_results(self, training: dict):
        print()
        print("=== תוצאות על דאטה של בדיקה (25%)  ===")
        print(f"אחוז הצלחה: {training['result']:.2f}%")
        print(f"מספר הצלחות (yes): {training['num_yes']}")
        print(f"מספר כישלונות (no): {training['num_no']}")

        
    def prediction(self):
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
        # self.a = self.df[self.target_variable].unique()[0]
    
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
        self._clean()
        return self._df_split()
        
    def _clean(self):
        if self.columns and len(self.columns) == self.df.shape[1]:
            self.df.columns = self.columns
        
        self.df.rename(columns={self.target_variable:"target"}, inplace=True)
        self.df['target'] = self.df.pop('target')
        self.df["target"] = self.df["target"].map({self.no: "no", self.yes: "yes"})
    
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
        yes = no = 0

        for _, row in self.df.iterrows():
            dict_val = row.to_dict()
            if Prediction(parameters=dict_val, modl=self.model).activation():
                if dict_val["target"] == 'yes': yes += 1
                else: no += 1
            else:
                if dict_val["target"] == 'no': yes += 1
                else: no += 1
                
        return {"num_yes": yes, "num_no": no, "result": yes / self.df.shape[0] * 100 }



if __name__ == "__main__":
    print("=================================")
    columns = [
    "class", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
    "gill-attachment", "gill-spacing", "gill-size", "gill-color",
    "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
    "ring-number", "ring-type", "spore-print-color", "population", "habitat"
    ]
    
    model_system = ModelSystem()
    model_system.upload_data(data_pat="agaricus-lepiota.csv", target_variable="class", str_yes="e", str_no="p", columns=columns)
    model_system.training()
    print(" ========= מודל אימון על הכול =========")
    model_system.testing("training_all.json")
    print(" ========= מודל אימון על 75% =========")
    model_system.testing("training_75.json")
    

# =================================
#  ========= מודל אימון על הכול =========
# === תוצאות על כל הדאטה (אימון + בדיקה) ===
# אחוז הצלחה: 95.64%
# מספר הצלחות (yes): 7769
# מספר כישלונות (no): 354

# === תוצאות על דאטה של בדיקה (25%)  ===
# אחוז הצלחה: 97.00%
# מספר הצלחות (yes): 5910
# מספר כישלונות (no): 183
#  ========= מודל אימון על 75% =========
# === תוצאות על כל הדאטה (אימון + בדיקה) ===
# אחוז הצלחה: 55.58%
# מספר הצלחות (yes): 4515
# מספר כישלונות (no): 3608

# === תוצאות על דאטה של בדיקה (25%)  ===
# אחוז הצלחה: 40.80%
# מספר הצלחות (yes): 2486
# מספר כישלונות (no): 3607