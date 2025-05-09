import numpy as np
import pandas as pd

class Collaborative_Filtering:
    def __init__(self, learning_rate, termo_de_reg, epochs, hidden_features):
        self.lr = learning_rate
        self.reg = termo_de_reg
        self.epochs = epochs
        self.k = hidden_features

        self.U = None # Matriz de usuarios 
        self.F = None # Matriz de filmes
    
    def fit(self, R):
        print("Comecando treinamento...")
        # R é a matriz com as notas
        n, m = R.shape

        self.U = np.random.rand(n, self.k)
        self.F = np.random.rand(m, self.k)

        #Mascara para nn pegar os valores nulos
        mask = ~np.isnan(R)

        for epoch in range(self.epochs):
            # Atualizando matriz de usuarios
            Loss = 0
            for i in range(n):
                # Filmes listados por i
                rated_filmes = np.where(mask[i])[0]
                if len(rated_filmes) == 0:
                    continue
                
                # Submatriz de Filmes avaliados por i
                F_a = self.F[rated_filmes] # Matriz (C x K)

                
                # Notas dadas aos filmes dados por i
                R_i = R[i, rated_filmes] # Matriz (1 x C)

                # Features do usuario i
                U_i = self.U[i].reshape(1,-1) # Matriz (1 x K)

                #Somando a Loss total
                Loss += (1/2)*(  (np.linalg.norm(R_i - U_i@F_a.T))**2 + self.reg*(np.linalg.norm(U_i))**2  )

                # Derivada
                d = self.reg*U_i - (R_i - U_i@F_a.T)@F_a # Calculando
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

                # Features do filme a
                F_a = self.F[a].reshape(1,-1) # Matriz (1 x K)

                #Somando a Loss total
                Loss += (1/2)*(  (np.linalg.norm(R_a - F_a@U_i.T))**2 + self.reg*(np.linalg.norm(F_a))**2  )

                # Derivada
                d = self.reg*F_a - (R_a - F_a@U_i.T)@U_i # Calculando
                d = d[0] # Transformando em vetor

                # Atualizando
                self.F[a] =- self.lr*d
        
            Loss /= (m+n)
            print(f"Loss da epoch {epoch+1}: {Loss}")
        
        print("----------------------------\nTreinamento finalizando\n----------------------------\n")
    
    def predict_all(self):
        matrix = self.U @ (self.F).T
        print(matrix)



df = pd.read_csv("Amazon.csv")
#print(df.head())

R = (df.iloc[:,1:].fillna(np.nan)).to_numpy()

# Train ALS model
model = Collaborative_Filtering(learning_rate=0.001, termo_de_reg=0.001, epochs=50, hidden_features=50)
model.fit(R)

print(R)
print()
model.predict_all()