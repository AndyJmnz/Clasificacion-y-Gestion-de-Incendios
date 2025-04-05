import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE

# Cargar los datos
df = pd.read_csv(r"C:\Users\ap56j\OneDrive\Documentos\Samsumg\NuevoProyecto\BD_IncendiosSNIF_2015-2023_LIMPIO.csv", encoding="latin-1", dtype=str)


# Verificar distribución inicial
print("Distribución antes del balanceo:")
print(df["Tipo impacto"].value_counts())

df["Tipo impacto"] = pd.to_numeric(df["Tipo impacto"])

# Separar por clases
df_clase_1 = df[df["Tipo impacto"] == 1].sample(n=2000, random_state=42)  # Reducir a 2000
df_clase_2 = df[df["Tipo impacto"] == 2]  # Mantener todos
df_clase_3 = df[df["Tipo impacto"] == 3]  # Mantener todos

# Aplicar SMOTE para generar datos sintéticos
X = df.drop(columns=["Tipo impacto"])
y = df["Tipo impacto"]

smote = SMOTE(sampling_strategy={2: 2000, 3: 2000}, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Convertir a DataFrame
df_sintetico = pd.DataFrame(X_resampled, columns=X.columns)
df_sintetico["Tipo impacto"] = y_resampled

# Unir con la clase 1 ya reducida
df_final = pd.concat([df_clase_1, df_sintetico], ignore_index=True)

# Guardar nuevo dataset balanceado
df_final.to_csv("dataset_balanceado.csv", index=False)
print("Dataset balanceado guardado con éxito.")