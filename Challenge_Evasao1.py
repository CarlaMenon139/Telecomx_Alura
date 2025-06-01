# 1. Importação das bibliotecas
import requests
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# 2. Extração dos dados
url = 'https://raw.githubusercontent.com/alura-cursos/challenge2-data-science/main/TelecomX_Data.json'
response = requests.get(url)
data = response.json()
df = pd.json_normalize(data)

# 3. Expansão de colunas com dicionários
for col in df.columns:
    if df[col].apply(lambda x: isinstance(x, dict) if pd.notnull(x) else False).any():
        expanded_cols = pd.json_normalize(df[col])
        expanded_cols.columns = [f"{col}_{subcol}" for subcol in expanded_cols.columns]
        df = pd.concat([df.drop(columns=[col]), expanded_cols], axis=1)

# 4. Padronização dos nomes das colunas

# Padronizar nomes de colunas (agora incluindo ponto)
df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('-', '_').str.replace('.', '_')

# Tratar dados duplicados e ausentes na variável alvo
df = df.drop_duplicates()
df = df.dropna(subset=['churn'])

print(df.columns.tolist())


# 5. Limpeza de dados
df = df.drop_duplicates()
df = df.dropna(subset=['churn'])
df['churn'] = df['churn'].map({'Yes': 1, 'No': 0})

# 6. Conversão da coluna de gasto total
gasto_col = 'account_charges_total'
df[gasto_col] = pd.to_numeric(df[gasto_col], errors='coerce')

# 7. Análise exploratória
sns.countplot(data=df, x='churn')
plt.title('Distribuição da Evasão de Clientes')
plt.show()

sns.histplot(data=df, x=gasto_col, hue='churn', multiple='stack', bins=30, kde=True)
plt.title('Distribuição do Gasto Total por Situação de Evasão')
plt.show()

plt.figure(figsize=(12,8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title('Correlação entre Variáveis Numéricas')
plt.show()

# 8. Gráficos por variáveis categóricas
categorical_cols = df.select_dtypes(include='object').columns.tolist()
for col in categorical_cols:
    plt.figure(figsize=(8,4))
    sns.countplot(data=df, x=col, hue='churn')
    plt.title(f'Distribuição da variável {col} por churn')
    plt.xticks(rotation=45)
    plt.show()

# 9. Preparação para o modelo
dict_cols = [col for col in df.columns if df[col].apply(lambda x: isinstance(x, dict) if pd.notnull(x) else False).any()]
df = df.drop(columns=dict_cols)

# Separar X e y
X = df.drop('churn', axis=1)
y = df['churn']

# Resetar índices
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# One-hot encoding
X = pd.get_dummies(X)

# Garantir alinhamento novamente
X, y = X.align(y, join='inner', axis=0)

# Dividir treino/teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Diagnóstico após a separação dos dados
print("Total de NaNs em y_train:", y_train.isna().sum())
print("Total de elementos em y_train:", len(y_train))

# Exibir valores ausentes
print(y_train[y_train.isna()])   

# 10  Mapear churn para 0/1 se for categórico
if df['churn'].dtype == 'object':
    df['churn'] = df['churn'].map({'No': 0, 'Yes': 1})

target = 'churn'
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

if target in categorical_cols:
    categorical_cols.remove(target)

# Criar variáveis dummies para categóricas, drop_first=True para evitar multicolinearidade
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Separar features e target
X = df.drop(columns=target)
y = df[target]

# *** CORREÇÃO: remover linhas com NaN no target antes do split ***
mask = y.notna()
X = X[mask]
y = y[mask]

print(df.columns)
print(df.shape)
print(df.head())

# Agora fazer o split normalmente
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

# Treinamento e avaliação
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
