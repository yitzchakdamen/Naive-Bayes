
class ModelInfo:
    """
    Class to handle model information and predictions.
    """
    
    def __init__(self, nmodel: dict):
        self.nmodel = nmodel

    def get_model_info(self):
        """
        Returns the model information.
        """
        information = {
            "name": self.nmodel.get("name"),
            "columns": [
                {col: [col_a for col_a in self.nmodel.get("target_variable", [])]} for col in self.nmodel.get("columns", [])
                ],
            "target_variable": self.nmodel.get("target_variable"),
        }
        return information
    