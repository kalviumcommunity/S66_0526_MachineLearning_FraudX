import joblib
from sklearn.ensemble import RandomForestClassifier
from src.config import RANDOM_STATE, MODEL_PATH

def train_model(X_train, y_train) -> RandomForestClassifier:
    """
    Train a Random Forest model on the training data.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model

def save_model(model: RandomForestClassifier, filepath: str = MODEL_PATH) -> None:
    """
    Serialize the trained model to disk.
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_saved_model(filepath: str = MODEL_PATH) -> RandomForestClassifier:
    """
    Load a serialized model from disk.
    """
    return joblib.load(filepath)
