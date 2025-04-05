import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# 1. Cargar datasets
df_incendios = pd.read_csv('BD.csv', encoding='ISO-8859-1', low_memory=False)
df_temp = pd.read_csv('temperaturas-mexico.csv')

# 2. Generar Clusters geográficos
X_geo = df_incendios[['latitud_grados', 'longitud_grados']].dropna()
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
df_incendios.loc[X_geo.index, 'Cluster_geo'] = kmeans.fit_predict(X_geo)

# 3. Procesar fechas y calcular Mes
meses_abrev = {
    1:'Ene', 2:'Feb', 3:'Mar', 4:'Abr', 5:'May', 6:'Jun',
    7:'Jul', 8:'Ago', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dic'
}
df_incendios['Fecha Inicio'] = pd.to_datetime(df_incendios['Fecha Inicio'], errors='coerce')
df_incendios['Mes'] = df_incendios['Fecha Inicio'].dt.month

# 4. Convertir estados a minúsculas y limpiar
df_incendios['Estado'] = df_incendios['Estado'].str.lower().str.strip()
df_temp['Estado'] = df_temp['Estado'].str.lower().str.strip()

# 5. Función para asignar temperatura
def obtener_temperatura(row):
    estado = row['Estado']
    mes_abrev_local = meses_abrev.get(row['Mes'], 'Anual')
    if estado in df_temp['Estado'].values and mes_abrev_local in df_temp.columns:
        temp = df_temp.loc[df_temp['Estado'] == estado, mes_abrev_local]
        if not temp.empty:
            return temp.values[0]
    return np.nan

df_incendios['Temp_prom'] = df_incendios.apply(obtener_temperatura, axis=1)

# 6. Variable objetivo: Duración días
df_incendios['Duración días'] = pd.to_numeric(df_incendios['Duración días'], errors='coerce')
df_incendios.dropna(subset=['Duración días', 'Temp_prom'], inplace=True)

# Transformación logarítmica para Duración
y = np.log1p(df_incendios['Duración días'])

# 7. Selección de características
features = [
    'Año',
    'latitud_grados', 'longitud_grados',
    'Estado', 'Municipio', 'Región', 'Causa',
    'Tipo de incendio', 'Tipo Vegetación', 'Régimen de fuego',
    'Tipo impacto', 'Total hectáreas', 'Tamaño', 'Detección', 'Llegada',
    'Cluster_geo', 'x'
]

# Crear una copia para evitar warnings
X = df_incendios[features].copy()

# 8. Definir variables numéricas y categóricas
numerical_features = [
    'Año', 'latitud_grados', 'longitud_grados', 'Total hectáreas',
    'Detección', 'Llegada', 'Temp_prom'
]
categorical_features = [
    'Estado', 'Municipio', 'Región', 'Causa',
    'Tipo de incendio', 'Tipo Vegetación', 'Régimen de fuego',
    'Tipo impacto', 'Tamaño', 'Cluster_geo'
]

# Convertir a numérico las columnas definidas como numéricas
for col in numerical_features:
    X.loc[:, col] = pd.to_numeric(X[col], errors='coerce')

# Eliminar filas con NaN en cualquiera de las columnas
X.dropna(subset=numerical_features + categorical_features, inplace=True)
# Ajustar y para que coincida con el nuevo índice
y = y.loc[X.index]

# 9. Preprocesamiento y pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    (
        'regressor',
        XGBRegressor(
            random_state=42,
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6
        )
    )
])

# 10. Entrenamiento y evaluación
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_test_orig = np.expm1(y_test)
y_pred_orig = np.expm1(y_pred)

mse = mean_squared_error(y_test_orig, y_pred_orig)
r2 = r2_score(y_test_orig, y_pred_orig)

print(f'MSE: {mse}')
print(f'R²: {r2}')
