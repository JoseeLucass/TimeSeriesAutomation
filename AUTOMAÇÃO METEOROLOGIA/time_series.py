
#----------------IMPORTAÇÕES PARA O DOWNLOAD DO DATASET NO SITE DO INMET-----------
from selenium import webdriver
from selenium.webdriver.common.by import By
import time
import os

# Definir o caminho da pasta de download
pasta_download = os.path.join(os.getcwd(), 'DataBase')

# Criar a pasta se não existir
if not os.path.exists(pasta_download):
    os.makedirs(pasta_download)


chrome_options = webdriver.ChromeOptions()
prefs = {"download.default_directory": pasta_download}
chrome_options.add_experimental_option("prefs", prefs)
chrome_options.add_argument("--start-maximized") 


driver = webdriver.Chrome(options=chrome_options)


driver.get("https://portal.inmet.gov.br/")


time.sleep(3)


botao_menu = driver.find_element(By.XPATH, '/html/body/div[2]/div[2]/div/div/div/div/div/div/div/nav/div/ul/li[3]/a')
botao_menu.click()


time.sleep(2)


botao_dados = driver.find_element(By.XPATH, '//*[@id="navbarSupportedContent"]/ul/li[3]/ul/li[4]/a')
botao_dados.click()


time.sleep(3)

# Rolando a página até o final
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)


botao_download = driver.find_element(By.XPATH, '//*[@id="main"]/div/div/article[25]/a')
botao_download.click()

time.sleep(10)

time.sleep(120)
driver.quit()

print(f"Arquivo salvo na pasta {pasta_download}.")

#----------------EXTRAIR O ARQUIVO BAIXADO-----------

import zipfile


# Caminho da pasta onde o arquivo foi baixado
pasta_download = os.path.join(os.getcwd(), 'DataBase')


for arquivo in os.listdir(pasta_download):
    if arquivo.endswith('.zip'):
        caminho_arquivo = os.path.join(pasta_download, arquivo)
        
        # Extrair o arquivo ZIP
        with zipfile.ZipFile(caminho_arquivo, 'r') as zip_ref:
            zip_ref.extractall(pasta_download)
        print(f"Arquivo {arquivo} extraído com sucesso na pasta {pasta_download}.")
        break
else:
    print("Nenhum arquivo .zip encontrado na pasta de download.")

#--------------------------------------------TRATAMENTO DAS BASES-------------------------------------------

import chardet
import pandas as pd


arquivo = './DataBase/INMET_NE_PB_A320_JOAO PESSOA_01-01-2024_A_31-10-2024.csv'

with open(arquivo, 'rb') as f:
    resultado = chardet.detect(f.read())
    encoding = resultado['encoding']


base = pd.read_csv(arquivo, sep=";", encoding=encoding, decimal=",", skiprows=8)

#-------------------------------------------SERIE TEMPORAL-------------------------------------

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from fpdf import FPDF

print(base)


print(base.columns)

# Pegando a série que quero trabalhar
serie_temporal = base['PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)']

# Tentando converter as datas no formato %Y/%d/%m
base['Data'] = pd.to_datetime(base['Data'], format='%Y/%d/%m', errors='coerce')


base['Data'] = base['Data'].fillna(pd.to_datetime(base['Data'], format='%d/%m/%y', errors='coerce'))


serie_temporal.index = base['Data']

# Ajustando a data da série para poder realizar a decomposição
serie_temporal.index = pd.Series(pd.date_range(serie_temporal.index[0], periods=len(serie_temporal.index), freq="d"))
print(serie_temporal)


serie_temporal = serie_temporal.fillna(0)  


# Decomposição da série temporal
result = seasonal_decompose(serie_temporal, model='additive')
result.plot()
plt.savefig('decomposicao.png')
plt.close()


plot_acf(serie_temporal, lags=30)
plt.savefig('acf.png')
plt.close()

plot_pacf(serie_temporal, lags=30)
plt.savefig('pacf.png')
plt.close()

# Função para criar janelas de tempo
def Janela_tempo(serie, p):
    '''
    Metodo que transforma um vetor em uma matriz com os dados de entrada e um vetor com as respectivas saídas
    :param serie: série temporal que será remodelada
    :return: retorna duas variáveis, uma matriz com os dados de entrada e um vetor com os dados de saída: matriz_entrada, vetor_saida
    '''
    tamanho_matriz = len(serie) - p

    matriz_entrada = []
    for i in range(tamanho_matriz):
        matriz_entrada.append([0.0] * p)

    vetor_saida = []
    for i in range(len(matriz_entrada)):
        matriz_entrada[i] = serie[i:i+p]
        vetor_saida.append(serie[i+p])

    return np.array(matriz_entrada), np.array(vetor_saida)

# Função para normalizar a série
def Normalizar(serie):
    '''
    Metodo que normaliza a série temporal em um intervalo de [0, 1]
    :param serie: série temporal que será normalizada
    :return: retorna a série normalizada
    '''
    min = np.min(serie)
    max = np.max(serie)

    serie_norm = []
    for i in serie:
        valor = (i - min)/(max - min)
        serie_norm.append(valor)

    return np.array(serie_norm)


serie_normalizada = Normalizar(serie_temporal.values)


plt.plot(serie_normalizada)
plt.savefig('serie_normalizada.png')
plt.close()

# Criando as janelas de tempo
X, y = Janela_tempo(serie_normalizada, 9) 

# Dividindo em treinamento e teste
tamanho_treinamento = int(round(0.8 * len(serie_normalizada)))  
tamanho_teste = int(round(0.2 * len(serie_normalizada)))       

X_treinamento = X[:tamanho_treinamento]
y_treinamento = y[:tamanho_treinamento]

X_teste = X[tamanho_treinamento+1:]
y_teste = y[tamanho_treinamento+1:]

# Instanciando o modelo DecisionTreeRegressor
clf = DecisionTreeRegressor()

# Definindo os parâmetros para GridSearchCV
parameters = {'max_depth': [5,8,9,10,12, 15,20, 30], 'min_samples_split': [5,10,15, 20, 25, 30, 35], 'min_samples_leaf': [7, 8, 9, 10, 11,15,20]}

# Usando GridSearchCV para encontrar a melhor combinação de hiperparâmetros
grid_search = GridSearchCV(clf, parameters, cv=5)
grid_search.fit(X_treinamento, y_treinamento)


clf_params = grid_search.best_params_
print("Melhores parâmetros :", clf_params)

# Fazer previsões usando o modelo otimizado
previsoes_arvore = grid_search.predict(X_teste)


mse_arvore_de_decisao = mean_squared_error(y_teste, previsoes_arvore)
print("Erro quadrático médio (MSE) do modelo otimizado:", mse_arvore_de_decisao)

# Instanciando o modelo LinearRegression
linear_regression = LinearRegression()


parameters_lr = {
    'fit_intercept': [True, False],
    'copy_X': [True, False]
}

# Usando GridSearchCV para encontrar a melhor combinação de hiperparâmetros
grid_search_lr = GridSearchCV(linear_regression, parameters_lr, cv=5)
grid_search_lr.fit(X_treinamento, y_treinamento)

# Melhores parâmetros encontrados
lr_params = grid_search_lr.best_params_
print("Melhores parâmetros :", lr_params)

# Fazer previsões usando o modelo otimizado
previsoes_lr = grid_search_lr.predict(X_teste)


mse_lr = mean_squared_error(y_teste, previsoes_lr)
print("Erro quadrático médio (MSE) do modelo otimizado:", mse_lr)

# Plotar os valores reais e as previsões dos modelos
plt.figure(figsize=(12, 6))
plt.plot(y_teste[-50:], label='Real', linestyle='-', linewidth=3, marker='o', zorder=3)
plt.plot(previsoes_arvore[-50:], label='Decision Tree Regressor', linestyle='-', linewidth=3, alpha=0.7, zorder=1, color='red')
plt.plot(previsoes_lr[-50:], label='Linear Regression', linestyle='-', linewidth=3, alpha=0.7, zorder=1, color='blue')
plt.xlabel('Observações', fontsize=14)
plt.ylabel('Valores (série temporal)', fontsize=14)
plt.title('Comparação de Teste: Decision Tree Regressor vs Linear Regression', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=12)
plt.savefig('comparacao_modelos.png')
plt.close()



############################### CRANDO RELATORIO DE PDF COM AS IMAGENS SALVAS #################################################


pdf = FPDF()


pdf.add_page()


pdf.set_font("Arial", size=16)
pdf.cell(200, 10, txt="Relatório Final - Análise de Série Temporal", ln=True, align='C')


pdf.ln(10)


imagens = ['decomposicao.png', 'acf.png', 'pacf.png', 'serie_normalizada.png', 'comparacao_modelos.png']

for imagem in imagens:
    
    pdf.image(imagem, x=10, y=None, w=180)
    
    pdf.ln(10)

# Salvando o PDF
pdf.output("relatorio_final.pdf")

print("Relatório PDF criado com sucesso!")