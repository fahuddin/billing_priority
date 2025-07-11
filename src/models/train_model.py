from ..data import load_raw_data
from ..features import feature_builder
from sklearn.ensemble import RandomForestClassifier


def train():
    X, y = load_raw_data()
    X_feat = feature_builder(X)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_feat, y)

    evaluate_model(model, X_feat, y)
    return model

if __name__ == "__main__":
    train()
