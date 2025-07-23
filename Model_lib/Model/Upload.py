import pandas as pd
import json


class UploadData:
    
    def _upload_csv(self, file_pat):
        return pd.read_csv(file_pat)
    
    def _upload_json(self, file_pat):
        try:
            if isinstance(file_pat, str):
                with open(file_pat, "r", encoding="utf-8") as file:
                    return json.load(file)
            else:
                return json.load(file_pat)
        except Exception as e:
            print(f"Error loading JSON file: {e}")
            return {}

    @staticmethod
    def upload(file):
        if isinstance(file, dict):
            return file
        
        if isinstance(file, str):
            if file.endswith(".csv"):
                return UploadData()._upload_csv(file)
            elif file.endswith(".json"):
                return UploadData()._upload_json(file)

        if hasattr(file, "name"):
            name = file.name.lower()
            if name.endswith(".csv"):
                return UploadData()._upload_csv(file)
            elif name.endswith(".json"):
                return UploadData()._upload_json(file)

        raise ValueError("Unsupported file type. Please upload a CSV or JSON file.")
