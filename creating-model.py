# %% Importações

import pickle
from keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    Callback
)

from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Input
)

import logging
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix    
)

import seaborn as sns

# %% Configurando o logging


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='training.log',
    filemode='w'
)

logger = logging.getLogger()

# Adicionando um FileHandler para um novo arquivo de log específico para o treinamento
new_log_file = 'training_specific.log'
fh = logging.FileHandler(new_log_file)
fh.setLevel(logging.INFO)
fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(fh)

# %% Carregamento das variáveis

with open('dataset-variables/healthimpact.pkl', 'rb') as f:
    X_train, X_test, y_train, y_test = pickle.load(f)

# %% Criando um callbacks personalizado para armazenar o logging


class LoggingCallback(Callback):
    def on_epoch_end(self, epochs, logs=None):
        loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        accuracy = logs.get('accuracy')
        val_accuracy = logs.get('val_accuracy')

        log_message = f'Epoch {epochs+1}: '
        if loss is not None:
            log_message += f'loss={loss:.4f}, '
        if accuracy is not None:
            log_message += f'accuracy={accuracy:.4f}, '
        if val_loss is not None:
            log_message += f'val_loss={val_loss:.4f}, '
        if val_accuracy is not None:
            log_message += f'val_accuracy={val_accuracy:.4f}'

        logger.info(log_message.strip().strip(','))

# %% Criação das callbacks


es = EarlyStopping(monitor='loss', min_delta=1e-10, patience=10, verbose=1)
rlr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=5, verbose=1)
log = LoggingCallback()

# %% Criando modelo, compilando e treinando

model = Sequential()
model.add(Input(shape=(X_train.shape[1],)))
model.add(Dense(units=64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(units=32, activation='relu'))
model.add(Dense(units=5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['categorical_accuracy'])
history = model.fit(X_train, y_train, callbacks=[
                    es, rlr, log], validation_data=(X_test, y_test), epochs=100)

# %% Obtendo histórico de accuracy e loss e plotando figura

model_accuracy = history.history['categorical_accuracy']
model_val_accuracy = history.history['val_categorical_accuracy']
model_loss = history.history['loss']
model_val_loss = history.history['val_loss']

plt.figure(figsize= (15, 5))

plt.subplot(1, 2, 1)
plt.plot(model_accuracy, label= 'Accuracy')
plt.plot(model_val_accuracy, label= 'Validation Accuracy')
plt.title('Accuracy and Validation Accuracy', fontsize= 14, fontweight= 'bold')
plt.xlabel('Epochs', fontsize= 12, fontweight= 'bold')
plt.ylabel('Accuracy', fontsize= 12, fontweight= 'bold')
plt.legend(fontsize= 'large')

plt.subplot(1, 2, 2)
plt.plot(model_loss, label= 'Loss')
plt.plot(model_val_loss, label= 'Validation Loss')
plt.title('Loss and Validation Loss', fontsize= 14, fontweight= 'bold')
plt.xlabel('Epochs', fontsize= 12, fontweight= 'bold')
plt.ylabel('Loss', fontsize= 12, fontweight= 'bold')
plt.legend(fontsize= 'large')

plt.savefig('C:/Users/kaiqu/OneDrive/Documentos/py-deep-learning-projects/health-impact/plots/historico.png')
plt.show()

# %% Fazendo previsões de novas ocorrências

predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis= 1)
y_test_original = np.argmax(y_test, axis= 1)

# %% Métricas

# Accuracy score
accuracy = accuracy_score(y_test_original, predictions)
logger.info(f'A acurácia do modelo é: {accuracy}')

# Classification Report
logger.info(f'Classification Report: \n\n {classification_report(y_test_original, predictions)}')

# Matriz de Confusão
cm = confusion_matrix(y_test_original, predictions)
tags = ['Very High', 'High', 'Moderate', 'Low', 'Very Low'] # tags da matriz

plt.figure(figsize= (5, 5)) # criando figura e definindo tamanho
sns.heatmap(cm, annot= True, fmt= 'd', cmap= 'cubehelix', xticklabels= tags, yticklabels= tags) # criando matriz mostrando os números em inteiro, passando tags e mudando cor
plt.title('Confusion Matrix - Random Forest', fontsize= 16, fontweight= 'bold') # adicionado título
plt.xlabel('Predicted', fontsize= 12, fontweight= 'bold') # definindo eixo X
plt.ylabel('Real', fontsize= 12, fontweight= 'bold') # definindo eixo Y
plt.savefig('C:/Users/kaiqu/OneDrive/Documentos/py-deep-learning-projects/health-impact/plots/confusion.png')
plt.show() # plotando a figura