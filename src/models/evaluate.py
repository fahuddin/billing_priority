from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X, y_true):
    y_pred = model.predic(X)
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    
    print("\nClassification Report:")
    report = classification_report(y_true, y_pred, output_dict=False)
    print(report)
    
    return report