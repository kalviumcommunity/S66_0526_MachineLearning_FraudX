from sklearn.ensemble import RandomForestClassifier
from src.config import RANDOM_STATE

def train_model(X_train, y_train, random_state: int = RANDOM_STATE) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.
    
    Args:
        X_train: Processed training features.
        y_train: Training labels.
        random_state: Seed for reproducibility.
        
    Returns:
        RandomForestClassifier: Trained model object.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train, y_train)
    return model
