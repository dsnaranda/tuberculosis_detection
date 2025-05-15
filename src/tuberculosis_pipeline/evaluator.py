import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
from config import MODEL_DIR


def evaluate_model(model, val_gen):
    # Genera métricas y gráficas de validación.
    probs = model.predict(val_gen).reshape(-1)
    preds = (probs > 0.5).astype(int)
    true  = val_gen.classes
    labels = list(val_gen.class_indices.keys())

    # Reporte
    report = classification_report(true, preds, target_names=labels)
    tn, fp, fn, tp = confusion_matrix(true, preds).ravel()
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    auc_score = roc_auc_score(true, probs)

    # ROC
    plt.figure()
    fpr, tpr, _ = roc_curve(true, probs)
    plt.plot(fpr, tpr, label=f'AUC = {auc_score:.2f}')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.title('ROC Curve')
    plt.savefig(os.path.join(MODEL_DIR, 'roc_curve.png'))
    plt.close()

    # Matriz de confusión
    plt.figure()
    cm = confusion_matrix(true, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    plt.close()

    # Guardar reporte textual
    with open(os.path.join(MODEL_DIR, 'evaluation_report.txt'), 'w') as f:
        f.write(report)
        f.write(f"\nSensitivity: {sensitivity:.4f}\n")
        f.write(f"Specificity: {specificity:.4f}\n")
        f.write(f"AUC: {auc_score:.4f}\n")