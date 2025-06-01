# Telecomx_Alura

# Análise de Evasão de Clientes - TelecomX 📉

# 📊 Relatório de Análise e Previsão da Evasão de Clientes (Churn)

> 📁 Projeto de Data Science | 📡 Setor: Telecomunicações | 🧠 Machine Learning com Random Forest

---

## 🧩 1. Introdução

A **evasão de clientes (churn)** é um dos maiores desafios para empresas de telecomunicações. A perda constante de clientes reduz a **receita**, aumenta os **custos de aquisição** e afeta a **sustentabilidade** do negócio.

🎯 Neste projeto, buscamos:
- Compreender o comportamento de cancelamento de clientes;
- Identificar os fatores de risco;
- Prever quais clientes têm maior chance de cancelar o serviço.

Utilizamos um processo completo de **ETL**, seguido de **análise exploratória** e **modelagem preditiva** com **Random Forest**.

---

## 🔄 2. Extração e Tratamento dos Dados (ETL)

### 📥 Extração
- Origem: arquivo `.json` hospedado em repositório público (Alura);
- Ferramentas: `requests`, `pandas`.

### 🛠️ Transformação
- 🔍 **Expansão de colunas aninhadas** (dicionários);
- ✍️ **Padronização de nomes de colunas** para formato `snake_case`;
- 🧹 **Limpeza de dados**: remoção de duplicatas e registros nulos;
- 🔢 **Conversões**:
  - `churn`: de texto para binário (0 = ativo, 1 = cancelado);
  - `account_charges_total`: convertido para `float`.

### 💾 Carga
- Dados preparados foram carregados em um DataFrame para análise.

---

## 📊 3. Análise Exploratória dos Dados (EDA)

### 🔢 Distribuição do Churn
- Gráfico de barras revelou um **desequilíbrio de classes**, com mais clientes ativos do que cancelados.

### 💸 Variáveis Numéricas
- Clientes que cancelaram apresentaram **padrões diferentes de gasto** (`account_charges_total`);
- Correlações entre variáveis foram mapeadas por **heatmap** para identificar fatores relevantes.

### 🧮 Variáveis Categóricas
- Gráficos por `contract`, `gender`, `internet_service`, entre outros, mostraram **tendências de evasão** em certos grupos.

---

## 🧠 4. Modelagem Preditiva

### 🔧 Pré-processamento
- **One-hot encoding** para variáveis categóricas (com remoção da primeira categoria);
- Exclusão de registros com `churn` ausente;
- **Split dos dados**:  
  - Treino: 70%  
  - Teste: 30%

### 🌳 Modelo
- Algoritmo: **Random Forest Classifier** (`sklearn`);
- Avaliação com métricas de:
  - **Precisão (Precision)**
  - **Revocação (Recall)**
  - **F1-Score**

---

## 📈 5. Resultados

📌 **Acurácia geral do modelo**: `80%`  
⚠️ **Recall para churners**: ~`43%`

> Isso indica que o modelo ainda **não identifica bem todos os clientes que irão cancelar**, o que é comum em conjuntos de dados desbalanceados.

---

## 🧾 6. Conclusão e Recomendações

### ✅ Principais Achados
- Padrões de uso e tipo de contrato estão fortemente ligados ao churn.
- Contratos mensais e maior valor de conta tendem a maior evasão.

### ⚠️ Limitações
- **Desbalanceamento de classes** reduz a sensibilidade do modelo em prever churners.

### 💡 Recomendações
- 📢 Implementar **ações preventivas** com base em perfis de alto risco.
- ⚙️ Testar **técnicas de balanceamento** (SMOTE, undersampling).
- 🧪 Incluir **novas variáveis** (ex: satisfação, histórico de suporte).
- 📊 Criar dashboards com atualizações constantes para análise contínua.

---

## 🧰 Tecnologias Utilizadas

| Ferramenta         | Finalidade                        |
|--------------------|-----------------------------------|
| Python             | Linguagem principal               |
| Pandas, NumPy      | Manipulação de dados              |
| Seaborn, Matplotlib| Visualizações                     |
| Scikit-learn       | Modelagem preditiva               |
| Jupyter Notebook   | Ambiente de desenvolvimento       |
| Git & GitHub       | Controle de versão e compartilhamento |

---

## 🗂️ Estrutura do Projeto

📁 telecom-churn-analysis
├── README.md
├── data/
│ └── TelecomX_Data.json
├── notebooks/
│ └── churn_analysis.ipynb
├── imagens/
│ ├── churn_plot.png
│ ├── correlacao.png
│ └── gasto_total.png
└── requirements.txt


---

## 🤝 Contribuições

Contribuições são muito bem-vindas!  
Sinta-se à vontade para abrir uma *issue* ou enviar um *pull request*.

---

## 👩‍💻 Autoria

**Carla Menon**  
📧 menon.cacau03@gmail.com  
🔗 [LinkedIn](https://https://www.linkedin.com/in/carla-roberta-de-souza-menon/)

---

