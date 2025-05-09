import numpy as np
import pandas as pd

class Collaborative_Filtering:
    def __init__(self, n_factors=10, n_iters=20, reg=0.1):
        self.n_factors = n_factors  # Features do modelo (K)
        self.n_iters = n_iters      # Numero de epochs que ele vai treinar
        self.reg = reg              # Termo de Regularização (λ)
        self.U = None               # Matriz de usuarios (M × K)
        self.F = None               # Matriz de filmes (N × K)
    
    def fit(self, R):
        """
        Treinando o modelo
        R -> matriz de filmes
        """
        n, m = R.shape

        #Inicializando as matrizes com valores aleatorios
        self.U = np.random.rand(n, self.n_factors)
        self.F = np.random.rand(m, self.n_factors)
        
        # Mascara para sabermos quais valores são ou não nulos
        mask = ~np.isnan(R)
        
        for _ in range(self.n_iters):
            # Primeiro paço: Fixa F e treina U
            for i in range(n):
                # Filmes avaliados pelo usuario i
                rated_items = np.where(mask[i])[0]
                if len(rated_items) == 0:
                    continue
                
                # Submatriz de features dos filmes vistos por i
                F_rated = self.F[rated_items]
                
                # Notas aos filmes dadas por i
                R_i = R[i, rated_items]
                
                # Resolve a equaçao
                A = F_rated.T @ F_rated + self.reg * np.eye(self.n_factors)
                b = F_rated.T @ R_i
                self.U[i] = np.linalg.solve(A, b)
            
            # Segundo paço
            for a in range(m):
                # Usuarios que avaliaram o filme a
                rated_users = np.where(mask[:, a])[0]
                if len(rated_users) == 0:
                    continue
                
                # Submatrix of P for users who rated item j
                U_rated = self.U[rated_users]
                
                # Notas dadas ao filme a
                R_a = R[rated_users, a]
                
                # Resolve a equaçao
                A = U_rated.T @ U_rated + self.reg * np.eye(self.n_factors)
                b = U_rated.T @ R_a
                self.F[a] = np.linalg.solve(A, b)
    
    def predict(self, user_idx, item_idx):
        """Avalia um filme para um usuario"""
        return np.dot(self.U[user_idx], self.F[item_idx])
    
    def predict_all(self):
        """Faz a prediçao das notas"""
        return self.U @ self.F.T




#==*=*=*==**=*==**=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=



df = pd.read_csv("Amazon.csv")
#print(df.head())

R = (df.iloc[:,1:].fillna(np.nan)).to_numpy()

# Train ALS model
model = Collaborative_Filtering(n_factors=10, n_iters=100, reg=0.001)
model.fit(R)

#Intervalo analisado
l0 = 0
lf = 15

c0 = 0
cf = 15

#Mostrando matriz original
print(f"\n{R[l0:lf, c0:cf]}\n\n")

# Predict missing ratings
predicted_R = model.predict_all()
print("Prediçao:\n")
print(np.round(np.clip(predicted_R[l0:lf, c0:cf], a_max=5, a_min=0), decimals=1))