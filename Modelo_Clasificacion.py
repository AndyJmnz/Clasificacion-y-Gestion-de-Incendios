import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# Cargar el archivo CSV
df = pd.read_csv(r"C:\Users\ap56j\OneDrive\Documentos\Samsumg\proyecto\BD_IncendiosSNIF_2015-2023_LIMPIOestesi.csv", encoding="latin-1")

# Eliminar registros de la clase 0
df = df[df['Tipo impacto'] != 0]

# Equilibrar la clase 1 reduciéndola a 2000 muestras
df_clase_1 = df[df['Tipo impacto'] == 1].sample(n=2000, random_state=50)
df_clase_2 = df[df['Tipo impacto'] == 2]
df_clase_3 = df[df['Tipo impacto'] == 3]

# Unir los datos reducidos
balanced_df = pd.concat([df_clase_1, df_clase_2, df_clase_3])

# Separar características y etiquetas
X = balanced_df[['Latitud', 'Longitud', 'longitud_grados', 'Duración días','Tamaño']]
y = balanced_df['Tipo impacto']

# Ajustar clases para empezar desde 0
y -= 1

# Dividir en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=50, stratify=y)


# Definir el espacio de búsqueda para los hiperparámetros de XGBoost
param_grid = {
    'n_estimators': [45, 50,60],
    'max_depth': [6, 7, 8],
    'learning_rate': [0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0],
    'gamma': [0],
    'scale_pos_weight': [{0: 1, 1: 1, 2: 10}, {0: 1, 1: 1, 2: 15}]
}

# Inicializar el modelo XGBoost
model = XGBClassifier(objective='multi:softmax', num_class=3, random_state=50)

# Inicializar GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, 
                           cv=3, scoring='accuracy', n_jobs=-1, verbose=2)

grid_search.fit(X_train, y_train)

# Obtener los mejores parámetros
print("Mejores parámetros encontrados:", grid_search.best_params_)

# Obtener el mejor modelo entrenado
best_model = grid_search.best_estimator_

# Predecir en el conjunto de prueba con el mejor modelo
y_pred_best = best_model.predict(X_test) + 1  # Restaurar las clases originales

# Evaluar el modelo optimizado
print("Exactitud (Accuracy) con XGBoost:", accuracy_score(y_test + 1, y_pred_best))
print("\nReporte de Clasificación (con XGBoost):\n", classification_report(y_test + 1, y_pred_best))

# Matriz de confusión con el modelo optimizado
cm_best = confusion_matrix(y_test + 1, y_pred_best, labels=[1, 2, 3])
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', xticklabels=["1", "2", "3"], yticklabels=["1", "2", "3"])
plt.xlabel('Predicho')
plt.ylabel('Real')
plt.title('Matriz de Confusión (Con XGBoost)')
plt.show()

# Visualizar la distribución de clases después del balanceo
plt.figure(figsize=(10, 5))
sns.countplot(x=y_train_resampled + 1, palette='viridis')
plt.title('Distribución de Clases en el Conjunto de Entrenamiento (Después de SMOTE)')
plt.xlabel('Clase')
plt.ylabel('Frecuencia')
plt.show()