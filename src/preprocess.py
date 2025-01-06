import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carregar o arquivo iris.csv
data = pd.read_csv('data/iris.csv')

# Remover a coluna 'Id', pois não é necessária para o modelo
data = data.drop(columns=['Id'])

# Separar as features e o alvo
X = data.drop(columns=['Species'])  # Features
y = data['Species']  # Target variable

# Transformar o alvo em variáveis numéricas
y = y.map({'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2})

# Dividir os dados em treino e teste (80% treino, 20% teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalizar as features para garantir que todas tenham a mesma escala
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Salvar os dados pré-processados em arquivos CSV
X_train_df = pd.DataFrame(X_train, columns=X.columns)
X_test_df = pd.DataFrame(X_test, columns=X.columns)

X_train_df.to_csv('data/X_train.csv', index=False)
X_test_df.to_csv('data/X_test.csv', index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv', index=False)

print("Pré-processamento concluído e dados salvos.")
