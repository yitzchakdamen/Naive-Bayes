
class ModelInfo:
    """Handle model information and predictions."""

    def get_model_info(self, nmodel: dict):
        """Extract and return model info as dict."""
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
    
