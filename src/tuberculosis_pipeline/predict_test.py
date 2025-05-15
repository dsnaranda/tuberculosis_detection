import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
from config import MODEL_DIR

# Directorio de test (datos/test)
test_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'test'))

# Cargar modelo entrenado
model_path = os.path.join(MODEL_DIR, 'tuberculosis_model_finetuned.h5')
model = load_model(model_path)

# Generador solo con rescale para test
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    test_dir,
    target_size=(224, 224),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Evaluar pérdida y exactitud generales
test_loss, test_acc = model.evaluate(test_gen, verbose=1)
print(f"Test loss: {test_loss:.4f}, Test accuracy: {test_acc:.4f}")

# Predicciones y métricas detalladas
probs = model.predict(test_gen).reshape(-1)
preds = (probs > 0.5).astype(int)
true = test_gen.classes
labels = list(test_gen.class_indices.keys())


print(classification_report(true, preds, target_names=labels))

# Matriz de confusión
cm = confusion_matrix(true, preds)
plt.figure()
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Test')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix_test.png'))
plt.close()

# Curva ROC
tpr, fpr, _ = None, None, None  # placeholders
fpr, tpr, _ = roc_curve(true, probs)
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
plt.plot([0,1], [0,1], linestyle='--')
plt.title('ROC Curve - Test')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(MODEL_DIR, 'roc_curve_test.png'))
plt.close()

import numpy as np

# Cálculo de aciertos y errores
total_samples = len(true)
correct_predictions = np.sum(preds == true)
incorrect_predictions = total_samples - correct_predictions

accuracy_percent = (correct_predictions / total_samples) * 100
error_percent = (incorrect_predictions / total_samples) * 100

print(f"\nTotal de muestras de test: {total_samples}")
print(f"Aciertos: {correct_predictions} ({accuracy_percent:.2f}%)")
print(f"Fallos: {incorrect_predictions} ({error_percent:.2f}%)")
