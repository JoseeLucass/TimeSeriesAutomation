# **Aplicação de Análise de Séries Temporais com Dados Meteorológicos do INMET**

## **Descrição do Projeto**
Esta aplicação realiza o download de dados meteorológicos do site do INMET, processa esses dados, realiza análises de séries temporais e gera um relatório em PDF com gráficos e resultados de modelos preditivos. O objetivo principal é demonstrar como obter insights a partir de séries temporais meteorológicas, utilizando modelos de regressão e técnicas estatísticas.

---

## **Funcionalidades**
1. **Download dos Dados:**
   - Automatiza o download de arquivos do site do INMET via Selenium.
   - Cria um diretório para armazenar os arquivos baixados.

2. **Processamento dos Dados:**
   - Extrai o arquivo ZIP baixado.
   - Detecta e aplica automaticamente a codificação correta do arquivo CSV.

3. **Análise de Séries Temporais:**
   - Decomposição da série temporal em componentes sazonais, de tendência e ruído.
   - Plotagem de gráficos ACF e PACF.
   - Normalização dos dados.

4. **Modelagem Preditiva:**
   - Treinamento de modelos:
     - Regressor de Árvores de Decisão.
     - Regressão Linear.
   - Otimização dos hiperparâmetros usando `GridSearchCV`.
   - Avaliação dos modelos com o erro quadrático médio (MSE).

5. **Geração de Relatório em PDF:**
   - Cria um relatório final com os gráficos gerados durante a análise.

---

## **Como Executar o Projeto**

### **Pré-requisitos**
- Python 3.8 ou superior.
- Navegador Google Chrome instalado.
- [Chromedriver](https://sites.google.com/chromium.org/driver/) compatível com sua versão do Chrome.

### **Instalação**
1. Clone este repositório:
   ```bash
   git clone https://github.com/seu-repositorio/projeto-inmet.git
   cd projeto-inmet
2. Instale as dependências do projeto:    
    ```bash
   pip install -r requirements.txt