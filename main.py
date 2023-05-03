import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# lendo o dataset
df = pd.read_excel('dataset.xlsx')
### CORRELAÇÕES
def correlations_display():
      # exibindo correlações
      correlations = df.corrwith(df['SARS-Cov-2 exam result'])
      print('columns with greater correlations')
      print(correlations.sort_values(ascending=False).head(10))
      print('\n\n')
      print('columns with smaller correlations')
      print(correlations.sort_values(ascending=True).head(10))
      print('\n\n')
      print(correlations.abs().sort_values(ascending=False).head(10))
      print('\n\n')
      
### RESULTADOS
# calculando acurácia
def display_results(model):
      prds = model.predict(X_test)
      tn, fp, fn, tp = confusion_matrix(y_test, prds).ravel()
      print(f'tn {tn}, fp {fp}, fn {fn}, tp {tp}', '\n\n',
            'Accuracy:', (accuracy_score(y_test, prds)), '\n\n',
            'Classification Report:\n', (classification_report(y_test, prds)))

### PRE-PROCESSAMENTO
# removendo colunas irrelevantes
df = df.drop(columns=['Patient ID', 'Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)'])

# mapeando dados categorizados
for column in df.columns:
      if df[column].dtype == 'object': 
            lbl = LabelEncoder()
            lbl.fit(list(df[column].values))
            df[column] = lbl.transform(list(df[column].values))

# checando a coluna de resultados
print(df['SARS-Cov-2 exam result'].value_counts(normalize=True))

# removendo colunas vazias e preenchendo campos com a media
df = df[df.columns[df.isna().sum()/df.shape[0] < 0.9]]
df = df.fillna(df.median())

### TREINAMENTO
# criando listas X (features) e y (resultado esperado)
# X = df[['Platelets', 'Leukocytes', 'Monocytes', 'Patient age quantile']]
X = df.select_dtypes(include='number')
y = df['SARS-Cov-2 exam result']

# separando 66% do dataset para treinamento e 33% para validação
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state=101)


# criando o modelo e treinando
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)
print('Algoritmo: KNN')
display_results(model)

model = GaussianNB()
model.fit(X_train, y_train)
print('Algoritmo: Naive Bayes')
display_results(model)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)

print('Algoritmo: Arvore de Decisão')
display_results(model)