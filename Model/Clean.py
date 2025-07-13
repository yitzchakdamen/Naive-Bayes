import pandas as pd
from sklearn.model_selection import train_test_split


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
        # self.df = self.df.fillna("missing")
        
        self.df = self.df.astype(str)
        if "Index" in self.df.columns:
            self.df.drop(columns=["Index"], inplace=True)
        if "index" in self.df.columns:
            self.df.drop(columns=["index"], inplace=True)

        
        if self.columns and len(self.columns) == self.df.shape[1]:
            self.df.columns = self.columns
            
        if self.target_variable in self.df.columns:
            
            self.df.rename(columns={self.target_variable:"target"}, inplace=True)
            self.df['target'] = self.df.pop('target')
            self.df['target'] = self.df['target'].apply(str)

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
        return {"all": self.df, "train_df": train_df, "test_df": test_df}
