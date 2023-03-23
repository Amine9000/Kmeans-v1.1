import numpy as np
import random
import math

# 
def calc_accuracy(y,y_pred):
    """
    calcule la precision (accuracy) de y_pred par rapport a y (y_true)
    cette condition sert a convertir un tableau numpy en liste python
    dans le cas ou nous avons passe y en tant que tableau numpy. 
    """
    if(type(y) != type([])):
        y = y.tolist()
    """
    ici nous additionnons simplement toutes les valeurs
    et arrondissons le resultat a 4 chiffres apres la virgule.
    """
    return round([y[i] - y_pred[i] for i in range(len(y_pred))].count(0)/len(y_pred),4)


class KMeans:
    """
    la class KMeans est une classe qui contient toutes les methodes et attributs
    necessaire pour executer l'algoritme de k-means
    """
    # constructeur
    def __init__(self,init='random',n_init=10,max_iters=100,k=1,tol=1e-4,random_state=None) -> None:
        """
        ici nous choisissons l'slgorithme que nous voulons utiliser 
        pour initialiser les premiers centres.
        soit "random" soit "kfarther"
        """
        self.init = init
        """
        le nombre de fois que nous allons repeter 
        l'entrainement de l'algorithme K-means
        """
        self.n_init = n_init
        # nombre de clusters ou centres
        self.k = k
        """
        nombre de fois nous voulons executer l'algorime 
        n_init fois
            max_iters fois
               k-means
        pour trouver les meilleures centres
        """
        self.max_iters = max_iters
        """
        nous utiliserons cette toleroncs pour arreter les boucles lorsque les erreurs
        cessent de changer ou lorsq'elles commencent a changer de tres petites quantites
        comme 0.0001 
        """
        self.tol = tol
        """
        nous utilisrons cette attribute pour fixer les valeurs aleatoires que nous obtiendrons
        du module 'random'
        """
        self.random_state = random_state

    def _random_select(self,X,k):
        if(self.random_state != None):
            random.seed(self.random_state)
        return [X[random.randint(0,len(X)-1)] for i in range(k)]
    """
    j'ai ecrit cette methodes pour corriger un bogue.
    lorsque je choisis des centres aleatoires je les stocke dans une liste
    et leur indices sont les etiquettes des clusters.
    le probleme est que par exemple si j'ai un cluster de points dans un espace 2D 
    dans les etiquettes reelles ils peuvent etre etiquettes comme cluster 0,
    mais la methode de selection aleatoire choisira peut-etre l'un de ces point
    comme centre et lr stockera dans la liste des centre a l'indice 1 par exemple .
    maintenant , le point choisi aleatoirement a partire du cluster 0 est le centre du cluster 1
    et cela entrainera l'etiquetage des enregistrements du cluster 1 comme cluster 0.
    pour resoidre ce probleme, j'ai modifie la position de chaque centre dans a la bonne
    position en fonction du numero qui lui est attribue dans la liste des etiquettes, par example
    si j'ai choisi un point du cluster 0 comme centre , je dois le stocker 
    a l'indice 0, et ainsi de suite. 
    """
    def _order_centers(self,X,y,centers):
        for j ,center in enumerate(centers):
            i = np.where((X == center).all(axis=1))
            if i != j:
                temp = centers[int(y[i])]
                centers[int(y[i])] = center   
                centers[j] = temp
        return centers
    """
    cette methode calcule la somme des distances minimale entre chaque point
    et son centre de cluster
    """
    def calc_min_dists(self,X,centers):
        sum_dists = 0
        for record in X:
            dists = []
            for center in centers:
                sub = [center[i] - record[i] for i in range(len(record))]
                square = [sub[i]**2 for i in range(len(sub))]
                summ = sum(square)
                dists.append(math.sqrt(summ))
            sum_dists += min(dists)
        return sum_dists
    """
    une methode qui initialise la liste des centers de maniere plus intele
    en choisissant les k points les plus eloignes.
    """
    def _kmeans_pp_select(self,X,k):
        centers = [X[random.randint(0,len(X)-1)]]
        for i in range(int(k-1)):
            probs = []
            for record in X:
                dists = []
                for center in centers:
                    sub = [center[i] - record[i] for i in range(len(record))]
                    square = [sub[i]**2 for i in range(len(sub))]
                    summ = sum(square)
                    dists.append(math.sqrt(summ))
                sum_dists = self.calc_min_dists(X,centers)
                probs.append((min(dists)/sum_dists))
            max_index = probs.index(max(probs))
            centers.append(X[max_index])
        return centers

    """
    cette methode calcule les distanse minimale entre chaque point
    et tout les centres et puis choiser l'index du point comme etiquette
    de ce point 
    """
    def _assign_clusters(slef,X,centers):
        L = []
        for point in X:
            min_index = 0
            min_dist = None
            for index,center in enumerate(centers):
                sub = point  - center
                squared = sub**2
                summ = sum(squared)
                root = math.sqrt(summ)
                if(min_dist == None):
                    min_dist = root
                    min_index = index
                else:
                    if(min_dist > root):
                        min_dist = root
                        min_index = index
            L.append(min_index)
        return L
    """
    c'est une methode qui calcule l'error
    """
    def _sum_squared_error(self,X,L,centers):
        sse = 0
        for r_index,record in enumerate(X):
            for c_index,center in enumerate(centers):
                sub = record - center
                square = sub**2
                Wi = 0
                if(L[r_index] == c_index):
                    Wi = 1
                summ = sum(Wi*square)
                sse += summ
        return sse
    """
    calculer la moyenne d'un cluster et la choisir comme le nouveau centre.
    """
    def _calc_means(self,X,L):
        means = [[ 0 for i in center] for center in self.centers]
        for r_index,record in enumerate(X):
            for c_index,center in enumerate(self.centers):
                Wi = 0
                if (L[r_index] == c_index):
                    Wi = 1
                means[c_index] = means[c_index] + Wi*record/(L.count(c_index)+1)
        return means
            


    def fit(self,X,y):
        centers = []
        if self.init == "kfarther":
            centers = self._kmeans_pp_select(X,self.k)
        else:
            centers = self._random_select(X,self.k)
        centers = self._order_centers(X,y,centers)
        best_centers = centers
        self.centers = best_centers
        L = self._assign_clusters(X,centers)
        min_acc = calc_accuracy(y,L)
        accuracies = []
        SSEs = []
        accuracies.append(min_acc)
        sse = self._sum_squared_error(X,L,centers)
        last_sse = 0
        SSEs.append(sse)
        i = j = 1
        bar_length = 20
        print()
        while i < self.n_init:
            while abs(last_sse - sse) > self.tol  and j < self.max_iters:
                centers = self._calc_means(X,L)
                L = self._assign_clusters(X,centers)
                acc = calc_accuracy(y,L)
                accuracies.append(acc)
                if(min_acc < acc):
                    best_centers = centers
                    min_acc = acc
                last_sse = sse
                sse = self._sum_squared_error(X,L,centers)
                SSEs.append(sse)
                print("\r[ iteration {0}, : error {1} , accuracy : {2}% ]".format(j,sse,acc),end='')
                j += 1
            i += 1
        print()
        self.centers = best_centers
        return (self.centers,accuracies,SSEs,min_acc)

    def predict(self,X):
        L = self._assign_clusters(X,self.centers)
        return L
