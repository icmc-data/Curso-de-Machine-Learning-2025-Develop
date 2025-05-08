import numpy as np
import pandas as pd
from functions import grandient_F

class Collaborative_Filtering:
    def __init__(self, learning_rate, termo_de_reg, epochs, hidden_features):
        self.lr = learning_rate
        self.reg = termo_de_reg
        self.epochs = epochs
        self.k = hidden_features

        self.U = None # Matriz de usuarios 
        self.F = None # Matriz de filmes
    
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    
    def fit(self, R):
        print("Comecando treinamento...")
        # R é a matriz com as notas
        n, m = R.shape

        scale = (1 / np.sqrt(self.k))
        self.U = np.random.uniform(-scale,scale, (n, self.k))
        self.F = np.random.uniform(-scale,scale, (m, self.k))

        #Mascara para nn pegar os valores nulos
        mask = ~np.isnan(R)

        for epoch in range(self.epochs):
            # Atualizando matriz de usuarios
            for i in range(n):
                # Filmes listados por i
                rated_filmes = np.where(mask[i])[0]
                if len(rated_filmes) == 0:
                    continue
                
                # Submatriz de Filmes avaliados por i
                F_a = self.F[rated_filmes] # Matriz (C x K)
                
                # Notas dadas aos filmes dados por i
                R_i = R[i, rated_filmes] # Matriz (1 x C)
                R_i = R_i.reshape(1,-1)

                # Features do usuario i
                U_i = self.U[i].reshape(1,-1) # Matriz (1 x K)

                #Prediction
                P0 = U_i@F_a.T
                P1 = self.sigmoid(P0)


                # Derivada
                #d = self.reg*U_i - (10/2)*((R_i - 5*P1)*(P1**2)*np.exp((-U_i)@F_a.T))@F_a # Calculando
                d = grandient_F(U_i, R_i, F_a, self.reg)
                d = d[0] # Transformando em vetor

                # Atualizando
                self.U[i] =- self.lr*d
            
            # Atualizando matriz de filmes
            for a in range(m):
                # Usuarios listados por a
                rated_users = np.where(mask[:, a])[0]
                if len(rated_users) == 0:
                    continue
                
                # Submatriz de usuarios que avaliaram a
                U_i = self.U[rated_users] # Matriz (C x K)
                
                # Notas dadas ao filme a
                R_a = R[rated_users, a] # Matriz (1 x C)
                R_a = R_a.reshape(-1,1).T

                # Features do filme a
                F_a = self.F[a].reshape(1,-1) # Matriz (1 x K)

                #Prediction
                P0 = F_a@U_i.T
                P1 = self.sigmoid(P0)


                # Derivada
                #d = self.reg*F_a - (10/2)*((R_a - 5*P1)*(P1**2)*np.exp((-F_a)@U_i.T))@U_i # Calculando
                d = grandient_F(F_a, R_a, U_i, self.reg)
                d = d[0] # Transformando em vetor

                # Atualizando
                self.F[a] =- self.lr*d
        
            print(f"Loss da epoch {epoch+1}: {self.compute_loss(R)}")
            #print(f"----\n{self.U}\n----")
        
        print("----------------------------\nTreinamento finalizando\n----------------------------\n")
    
    def predict_all(self):
        matrix = 5*self.sigmoid(self.U @ (self.F).T)
        print(matrix)
    
    def compute_loss(self, R):
        mask = ~np.isnan(R)
        error = (R - self.U @ self.F.T)[mask]
        return np.sum(error ** 2) + self.reg * (np.sum(self.U**2) + np.sum(self.F**2))



df = pd.read_csv("Amazon.csv")
#print(df.head())

R = (df.iloc[:,1:].fillna(np.nan)).to_numpy()

# Train ALS model
model = Collaborative_Filtering(learning_rate=0.01, termo_de_reg=0.001, epochs=100, hidden_features=2)
model.fit(R)

print(R)
print()
model.predict_all()