import numpy as np
from collections import Counter
import matplotlib.pyplot as plt

#distância euclidiana
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

class KNN:
    def __init__(self, k=5): 
        self.k = k
    
    def fit(self, X, y):
        """Recebe os dados de treino"""
        self.X_train = X
        self.y_train = y
    
    def predict(self, X, verbose=True):
        """Prediz a classe do ponto com base nos neighbors"""
        predicted_labels = [self._predict(x, verbose) for x in X]
        return np.array(predicted_labels)
    
    def _predict(self, x, verbose):
        """função auxiliar pra predict"""
        # distância entre ponto X e todos os demais do dataset
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        
        # Pega os K vizinhos, tal como suas classes
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        k_nearest_distances = [distances[i] for i in k_indices]
        k_nearest_points = [self.X_train[i] for i in k_indices]
        
        # Escolhendo a classe mais comum nos vizinhos
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# GERANDO DADOS
np.random.seed(42)

X_class0 = np.random.randn(30, 2) * 1.5 + [2, 3]
y_class0 = np.zeros(30)

X_class1 = np.random.randn(30, 2) * 1.8 + [7, 6]
y_class1 = np.ones(30)

X_class2 = np.random.randn(30, 2) * 1.2 + [5, 2]
y_class2 = np.full(30, 2)

X_train = np.vstack((X_class0, X_class1, X_class2))
y_train = np.hstack((y_class0, y_class1, y_class2))

X_test = np.array([
    [3, 4],  
    [6, 5],  
    [4, 2],  
    [8, 7],  
    [5, 4]   
])

#chamando o modelo
K = int(input('Quantos vizinhos? '))
knn = KNN(k=K) 
knn.fit(X_train, y_train)

# predictions
predictions = knn.predict(X_test)

# Printando o que deu
print("\nResultado:")
for point, pred in zip(X_test, predictions):
    print(f"Ponto {point} → Classe estimada: {pred}")

#VISUALIZAÇÃO
plt.figure(figsize=(12, 8))

plt.scatter(X_class0[:, 0], X_class0[:, 1], c='blue', s=80, edgecolor='k', 
            marker='o', label='Classe 0')
plt.scatter(X_class1[:, 0], X_class1[:, 1], c='green', s=80, edgecolor='k', 
            marker='^', label='Classe 1')
plt.scatter(X_class2[:, 0], X_class2[:, 1], c='orange', s=80, edgecolor='k', 
            marker='s', label='Classe 2')

colors = ['red', 'purple', 'brown', 'pink', 'cyan']
for i, (point, pred) in enumerate(zip(X_test, predictions)):
    plt.scatter(point[0], point[1], c=colors[i], marker='*', s=300, 
               edgecolor='k', linewidth=2, label=f'Teste {i+1} (Estimativa: {pred})')

for test_point in X_test:
    distances = [euclidean_distance(test_point, x_train) for x_train in X_train]
    k_indices = np.argsort(distances)[:knn.k]
    for i in k_indices:
        plt.plot([test_point[0], X_train[i, 0]], [test_point[1], X_train[i, 1]], 
                'black', linestyle=':', alpha=0.8)

plt.title(f'Algoritmo do KNN aplicado a um dataset (K = {K})', fontsize=14)
plt.xlabel('Feature 1', fontsize=12)
plt.ylabel('Feature 2', fontsize=12)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()