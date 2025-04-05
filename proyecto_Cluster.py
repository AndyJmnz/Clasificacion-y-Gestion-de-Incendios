import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import seaborn as sns

# Cargar datos
df = pd.read_csv('BD.csv', encoding='latin1').dropna()

# Preparación inicial (redondear coordenadas para regiones)
df['Latitud_round'] = df['Latitud'].round(1)
df['Longitud_round'] = df['Longitud'].round(1)

# Resumen regional para clusterización regional
region_summary = df.groupby(['Latitud_round', 'Longitud_round']).agg(
    frecuencia_incendios=('Año', 'count'),
    duracion_promedio=('Duración días', 'mean'),
    vegetacion_predominante=('Tipo Vegetación', lambda x: x.mode()[0])
).reset_index()

# Encoding vegetación predominante
veg_encoder = OneHotEncoder(sparse_output=False)
veg_encoded = veg_encoder.fit_transform(region_summary[['vegetacion_predominante']])
veg_df = pd.DataFrame(veg_encoded, columns=veg_encoder.get_feature_names_out())

region_features = pd.concat([
    region_summary[['frecuencia_incendios', 'duracion_promedio']], veg_df
], axis=1)

# Normalización regional
scaler = StandardScaler()
region_scaled = scaler.fit_transform(region_features)

# KMeans Regional
kmeans_region = KMeans(n_clusters=5, random_state=42, n_init=10)
region_summary['cluster_region'] = kmeans_region.fit_predict(region_scaled)

# Visualizar Clusters Regionales
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Longitud_round', y='Latitud_round', hue='cluster_region',
                size='frecuencia_incendios', sizes=(20, 200),
                palette='viridis', data=region_summary, alpha=0.7)
plt.title('Clusters Regionales de Probabilidad de Incendios')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.legend(title='Cluster Región')
plt.savefig('clusters_regionales.png')

# Clusterización Individual (incendios específicos)
categorical_cols = ['Causa', 'Tipo impacto', 'Tipo Vegetación']
numerical_cols = ['Duración días', 'Latitud', 'Longitud']

preprocessor = ColumnTransformer(transformers=[
    ('num', StandardScaler(), numerical_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
])

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=4, random_state=42, n_init=10))
])

df['cluster_incendio'] = pipeline.fit_predict(df)

# Resumen estadístico de clusters individuales
incendio_profiles = df.groupby('cluster_incendio').agg(
    num_incendios=('Año', 'count'),
    duracion_media=('Duración días', 'mean'),
    impacto_comun=('Tipo impacto', lambda x: x.mode()[0]),
    causa_comun=('Causa', lambda x: x.mode()[0]),
    vegetacion_comun=('Tipo Vegetación', lambda x: x.mode()[0])
).reset_index()



# Tabla resumen top 10 regiones más afectadas
print("\nTop 10 Regiones con Mayor Frecuencia de Incendios:\n")
top10_regiones = region_summary.sort_values(by='frecuencia_incendios', ascending=False).head(10)
print(top10_regiones[['Latitud_round', 'Longitud_round', 'frecuencia_incendios', 'vegetacion_predominante']])
top10_regiones.to_csv('top10_regiones_incendios.csv', index=False)

# Análisis explícito según ecosistema
print("\nAnálisis de incendios por ecosistema:\n")
ecosistema_summary = df.groupby('Tipo Vegetación').agg(
    frecuencia_incendios=('Año', 'count'),
    duracion_promedio=('Duración días', 'mean')
).sort_values(by='frecuencia_incendios', ascending=False).reset_index()

print(ecosistema_summary.head(10))


# Visualizar Clusters Individuales
plt.figure(figsize=(10, 8))
sns.scatterplot(x='Longitud', y='Latitud', hue='cluster_incendio',
                size='Duración días', sizes=(10, 100),
                palette='plasma', data=df, alpha=0.5)
plt.title('Clusters Individuales de Incendios')
plt.xlabel('Longitud')
plt.ylabel('Latitud')
plt.legend(title='Cluster Incendio')
plt.savefig('clusters_individuales.png')

# Matriz de Riesgo
risk_matrix = pd.crosstab(region_summary['cluster_region'], df['cluster_incendio'])
plt.figure(figsize=(8, 6))
sns.heatmap(risk_matrix, annot=True, fmt='g', cmap='YlOrRd')
plt.title('Matriz de Riesgo por Cluster Regional e Individual')
plt.xlabel('Cluster Incendio Individual')
plt.ylabel('Cluster Regional')
plt.savefig('matriz_riesgo.png')

print("Análisis y visualizaciones completos. Archivos generados exitosamente.")
