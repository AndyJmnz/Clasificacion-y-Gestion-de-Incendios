import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

def excel_serial_to_date(serial):
    if pd.notna(serial):  # Verificar si el valor no es NaN
        try:
            # Intentar convertir el valor a un número de serie de Excel
            serial = int(serial)
            base_date = datetime(1900, 1, 1)
            return base_date + timedelta(days=serial - 1)
        except (ValueError, TypeError):
            # Si falla, asumir que es una cadena de texto en formato de fecha
            try:
                # Intentar convertir la cadena a un objeto datetime
                return datetime.strptime(serial, "%d/%m/%Y")
            except ValueError:
                # Si no coincide el formato, devolver None
                return None
    else:
        return None  # Devolver None para valores NaN

pd.set_option('display.max_rows', None)

# Cargar el archivo CSV con la codificación correcta
df = pd.read_csv(r"C:\Users\ap56j\OneDrive\Documentos\Samsumg\proyecto\BD_IncendiosSNIF_2015-2023.csv", encoding="latin-1", dtype=str)

# Eliminar duplicados
df = df.drop_duplicates()

# Verifica que las columnas existen antes de eliminarlas
columnas_a_borrar = ['Clave Municipio','Arbolado Adulto', 'Renuevo', 'Arbustivo', 'Herbáceo', 'Hojarasca', 'Clave del incendio','CVE_ENT','CVE_MUN','CVEGEO']

# Elimina las columnas si existen en el DataFrame
df = df.drop(columns=[col for col in columnas_a_borrar if col in df.columns])

# Normalizar Causas
df['Causa'] = df['Causa'].str.strip().str.lower()

# Normalizar Predio
df['Predio'] = df['Predio'].str.strip().str.lower()

# Normalizar tipo de impacto
df['Tipo impacto'] = df['Tipo impacto'].str.strip().str.lower()

# Normalizar tipo de incendio
df['Tipo de incendio'] = df['Tipo de incendio'].str.strip().str.lower()

# Normalizar Regimen de fuego
df['Régimen de fuego'] = df['Régimen de fuego'].str.strip().str.lower()

# Normalizar Región
df['Región'] = df['Región'].str.strip().str.lower()

# Normalizar Municipio
df['Municipio'] = df['Municipio'].str.strip().str.lower()

# Normalizar Estado
df['Estado'] = df['Estado'].str.strip().str.lower()

# Normalizar tipo de incendio
df['Causa especifica'] = df['Causa especifica'].str.strip().str.lower()

# Normalizar tipo de incendio
df['Tipo Vegetación'] = df['Tipo Vegetación'].str.strip().str.lower()

# Normalizar Dias
df['Duración días'] = df['Duración días'].fillna('0 Días')
df['Duración días'] = df['Duración días'].str.extract(r'(\d+)', expand=False)

# Convertir dato de tipo de string a num 
# Mapeo de valores
mapping = {
    "impacto mínimo": 1,
    "impacto moderado": 2,
    "impacto severo": 3,
    0: 0  # Asegurar que los ceros se mantengan
}

# Aplicar el mapeo
df["Tipo impacto"] = df["Tipo impacto"].replace(mapping)

# Limpiar causas especificas
df["Causa especifica"] = df["Causa especifica"].replace(
    ["quema  para  pastoreo", "quema  para pastoreo", "quema de pastoreo", "quema para  pastoreo", "quema para \npastoreo"], 
    "quema para pastoreo"
)

df["Causa especifica"] = df["Causa especifica"].replace(
    ["ninguna", "ninguna / no aplica","ninguna/ no aplica","no aplica", "0"],
    "NA"
)

df["Causa especifica"] = df["Causa especifica"].replace(
    ["quema para preparación de siembra","quema para preparacion de siembra", "quema para siembra"],
    "quema para siembra"
)

df["Causa especifica"] = df["Causa especifica"].replace(
    ["fogatas de \npaseantes", "fogatas de paseantes"], 
    "fogatas"
)
df["Causa especifica"] = df["Causa especifica"].replace(
    ["hornos de carbón","horno de carbon"], 
    "hornos de carbon"
)

#Limpiar causas 
# Mapeo de valores
mappingCausas = {
    "actividades agrícolas": 1,
    "actividades ilícitas": 2,
    "actividades pecuarias": 3,
    "cazadores": 4,
    "desconocidas": 5,
    "festividades y rituales": 6,
    "fogatas": 7,
    "fumadores": 8,
    "intencional": 9,
    "limpias de derecho de vía": 10,
    "naturales": 11,
    "otras actividades productivas": 12,
    "quema de basureros": 13,
    "residuos de aprovechamiento forestal": 14,
    "transportes": 15,
    0: 0  # Asegurar que los ceros se mantengan
}

# Aplicar el mapeo
df["Causa"] = df["Causa"].replace(mappingCausas)

# Agrupar tipo vegetacion 
df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["bosque inducido","bosque inducido - bi","bosque de pino", "bosque de pino - bp", "bosque de \npino","bosque de \nencino", "bosque de ayarin","bosque de ayarín - bs","bosque de ayarín","bosque de cedro","bosque de cedro - bb","bosque de encino","bosque de encino - bq","bosque de encino-pino","bosque de encino-pino - bqp","bosque de galería","bosque de galería - bg","bosque de mezquite", "bosque de mezquite","bosque de oyamel","bosque de oyamel - ba","bosque de pino-encino","bosque de pino-encino - bpq","bosque de táscate","bosque de táscate - bj","bosque mesofilo - bm","bosque mesófilo","bosque mesófilo - bm","bosque mesófilo de montaña","bosque cultivado","bosque cultivado - bc"], 
    "bosque"
)

df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["chaparral - ml"],
    "chaparral"
)
df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["manglar - vm"],
    "manglar"
)
df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["palmar inducido - vpi","palmar natural - vpn","palmar natural","palmar inducido"],
    "palmar"
)
df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["petén","petén - pt"],
    "peten"
)
df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["popal - va"],
    "popal"
)
df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["pradera de alta montaña - vw"],
    "pradera de alta montaña"
)
df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["sabana - vs"],
    "sabana"
)
df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["sabanoide - vsi"],
    "sabanoide"
)
df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["tular - vt"],
    "tular"
)
df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["matorral crasicaule", "matorral crasicaule - mc", "matorral de coníferas", "matorral de coníferas - mj", "matorral desertico rosetófilo","matorral desértico micrófilo","matorral desertico micrófilo","matorral desértico micrófilo - mdm","matorral desértico rosetófilo","matorral desértico rosetófilo - mdr","matorral espinoso tamaulipeco","matorral espinoso tamaulipeco - met","matorral rosetófilo costero","matorral rosetófilo costero - mrc", "matorral sarco-crasicaule","matorral sarcocaule", "matorral sarcocaule - msc","matorral sarcocrasicaule","matorral sarcocrasicaule - mscc","matorral sarcocrasicaule de neblina - msn","matorral subtropical","matorral subtropical - mst","matorral submontano","matorral submontano - msm"], 
    "matorral"
)

df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["selva alta perennifolia","selva alta perennifolia - sap","selva alta subperennifolia","selva alta subperennifolia - saq","selva baja caducifolia","selva baja caducifolia - sbc","selva baja espinosa - sbc","selva baja espinosa - sbk","selva baja espinosa caducifolia","selva baja espinosa subperennifolia","selva baja perennifolia","selva baja perennifolia - sbp","selva baja subcaducifolia","selva baja subcaducifolia - sbs","selva baja subperennifolia","selva baja subperennifolia - sbq"],
    "selva baja"
)

df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["selva mediana caducifolia","selva mediana caducifolia - smc", "selva mediana perennifolia","selva mediana perennifolia - smp","selva mediana subcaducifolia","selva mediana subcaducifolia - sms","selva mediana subperennifolia","selva mediana subperennifolia - smq"],
    "selva mediana"
)

df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["vegetacion hidrofila", "vegetación de dunas costeras", "vegetación de dunas costeras - vu", "vegetación de galería", "vegetación de galería - vg","vegetación halofila hidrófila","vegetación halofila xerofila","vegetación halófila (hidrófila) - vhh","vegetación halófila (xerófila) - vh","vegetación halófila hidrófila","vegetación secundaria arbustiva de matorral de coníferas"], 
    "vegetacion"
)

df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["mezquital","mezquital (espinoso) - mke","mezquital (otros tipos) - mk","mezquital (xerófilo) - mkx","mezquital tropical","mezquital xerófilo"],
    "mezquital"
)

df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(
    ["pastizal gipsófilo", "pastizal gypsófilo", "pastizal halófilo", "pastizal halófilo - ph", "pastizal natural","pastizal natural - pn","pastizal inducido - pu"], 
    "pastizal"
)

# Mapeo de valores
mappingVege = {
    "bosque": 1,
    "chaparral": 2,
    "manglar": 3,
    "matorral": 4,
    "mezquital": 5,
    "palmar": 6,
    "pastizal": 7,
    "peten": 8,
    "popal": 9,
    "pradera de alta montaña": 10,
    "sabana": 11,
    "sabanoide": 12,
    "selva baja": 13,
    "selva mediana": 14,
    "tular": 15,
    "vegetacion": 16,
    0: 0  # Asegurar que los ceros se mantengan
}

# Aplicar el mapeo
df["Tipo Vegetación"] = df["Tipo Vegetación"].replace(mappingVege)


df["Tamaño"] = df["Tamaño"].replace(
    ["a"],
    "0"
)
# Mapeo de valores
mappingTamaño = {
    "0 a 5 Hectáreas": 1,
    "11 a 20 Hectáreas": 2,
    "21 a 50 Hectáreas": 3,
    "51 a 100 Hectáreas": 4,
    "6 a 10 Hectáreas": 5,
    "Mayor a 100 Hectáreas": 6,
    0: 0  # Asegurar que los ceros se mantengan
}

# Aplicar el mapeo
df["Tamaño"] = df["Tamaño"].replace(mappingTamaño)

valores_faltantes = df['Causa especifica'].isna().sum()

#print(f"La columna 'Causa especifica' tiene {valores_faltantes} valores faltantes.")

#print("\n",df.groupby("Tipo impacto")["Tipo impacto"].count())

df['Fecha Inicio'] = df['Fecha Inicio'].apply(excel_serial_to_date)

df['Fecha Termino'] = df['Fecha Termino'].apply(excel_serial_to_date)

print(df.columns)
#print(df["Fecha Termino"].head(5))

# Guardar el archivo CSV con la codificación correcta
df.to_csv("BD_IncendiosSNIF_2015-2023_LIMPIOestesi.csv", index=False, encoding="latin-1")

print("Archivo guardado exitosamente como 'BD_IncendiosSNIF_2015-2023_LIMPIO.csv'")



