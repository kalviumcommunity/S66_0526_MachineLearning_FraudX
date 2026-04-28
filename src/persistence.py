import joblib
import os

def save_artifacts(model, pipeline, model_path: str, pipeline_path: str) -> None:
    """
    Serialize and save model and preprocessing pipeline.
    
    Args:
        model: Trained model object.
        pipeline: Fitted preprocessing pipeline.
        model_path: Path to save the model.
        pipeline_path: Path to save the pipeline.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(pipeline, pipeline_path)

def load_artifacts(model_path: str, pipeline_path: str) -> tuple:
    """
    Load saved model and preprocessing pipeline.
    
    Args:
        model_path: Path to the saved model.
        pipeline_path: Path to the saved pipeline.
        
    Returns:
        tuple: (model, pipeline)
    """
    model = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    return model, pipeline
