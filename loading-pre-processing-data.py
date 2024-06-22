# %% Importações

import zipfile
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pickle

# %% Confrigurando o logging

logging.basicConfig(level=logging.INFO, filename='info-preprocessing.log',
                    format='%(asctime)s - %(levelname)s- %(message)s')

# %% Fazendo a extração do conteúdo

caminho_zip = 'air-quality-and-health-impact-dataset.zip'
caminho_extrair = 'C:/Users/kaiqu/OneDrive/Documentos/py-deep-learning-projects/health-impact'

# Fazendo a extração da pasta
with zipfile.ZipFile(caminho_zip, 'r') as zip_ref:
    zip_ref.extractall(caminho_extrair)

# %% Carregando o dataset

df = pd.read_csv('air_quality_health_impact_data.csv')

logging.info('As colunas desse dataframe é: ')
for col in df.columns:
    logging.info(col)

# Descrição do dataset
logging.info(df.describe())

# Informações do dataset
logging.info(df.info())

# Verificando a existência de valores nulos
logging.info('Verificando a existência de valores nulos: ')
logging.info(df.isnull().sum())

# Excluindo a coluna RecordID
if 'RecordID' in df.columns:
    df.drop('RecordID', axis=1, inplace=True)
else:
    logging.error('A coluna RecordID não existe na base de dados.')

# %% Plotando gráficos

# Mapa de correlação
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='crest', linewidth=1)
plt.title('Correlation Map', fontsize=14, fontweight='bold')
plt.savefig(
    'C:/Users/kaiqu/OneDrive/Documentos/py-deep-learning-projects/health-impact/plots/correlantionmap.png')
plt.show()

# Gráfico de distribuição
numerics_columns = df.select_dtypes(include='number')

for col in numerics_columns.columns:
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    sns.histplot(data=df, x=col, kde=True)
    plt.title(f'Distribuição da coluna {col}', fontsize=14, fontweight='bold')
    plt.xlabel(col, fontsize=12, fontweight='bold')

    plt.subplot(1, 2, 2)
    sns.boxplot(df[col])
    plt.title(f'Boxplot da coluna {col}', fontsize=14, fontweight='bold')
    plt.xlabel(col, fontsize=12, fontweight='bold')
    plt.tight_layout()

    plt.savefig(
        f'C:/Users/kaiqu/OneDrive/Documentos/py-deep-learning-projects/health-impact/plots/{col}.png')
    plt.show()

# %% Divisão de previsores e classe e normalização

target_columns = ['AQI', 'PM10', 'PM2_5', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity',
                  'WindSpeed', 'RespiratoryCases', 'CardiovascularCases',
                  'HospitalAdmissions']

x = df[target_columns]
y = df['HealthImpactClass']

logging.info(f'O shape da variável X é: {x.shape}')
logging.info(f'O shape da variável Y é: {y.shape}')

# Normalizando os previsores

scaler = StandardScaler()
x_scaled = scaler.fit_transform(X=x)

# Aplicando o padrão OneHot nas classes (requerido usando rede neural densa)
y_dummy = to_categorical(y)
logging.info(f'O shape das classes dummy é: {y_dummy.shape}')

# %% Dividindo as variáveis de treino e teste e salvando usando o pickle

X_train, X_test, y_train, y_test = train_test_split(
    x_scaled, y_dummy, test_size=0.2, random_state=42)

logging.info(
    f'O shape das variáveis de treino é: {X_train.shape}, {y_train.shape}')
logging.info(
    f'O shape das variáveis de teste é: {X_test.shape}, {y_test.shape}')

# Salvando as variáveis
with open('healthimpact.pkl', 'wb') as f:
    pickle.dump([X_train, X_test, y_train, y_test], f)
