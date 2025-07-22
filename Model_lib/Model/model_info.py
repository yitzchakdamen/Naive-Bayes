
class ModelInfo:
    """
    Class to handle model information and predictions.
    """
    MODELS_DIR = "./app/Files_model"

    def get_model_info(self, nmodel: dict):
        """
        Returns the model information.
        """
        columns_all = {}
        target_class = nmodel.get("target_variable", [])[0]

        if target_class and target_class in nmodel:
            for col in nmodel.get("columns", []):
                if col in nmodel[target_class]:
                    columns_all[col] = list(nmodel[target_class][col].keys())
                    
        information = {
            "name": nmodel.get("name"),
            "columns":nmodel.get("columns", []),
            "columns_all": columns_all,
            "target_variable": nmodel.get("target_variable"),
        }
        return information
    
