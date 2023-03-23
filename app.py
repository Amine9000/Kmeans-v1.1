from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import random
from kmeans import KMeans
from kmeans import calc_accuracy

# gettin data ready

X,y = make_blobs(n_samples=1000,n_features=2,centers=3,cluster_std=1,shuffle=True,random_state=0)

# train set
X_train= X[:900]
y_train= y[:900]

# test set
X_test = X[900:]
y_test = y[900:]


"""
Nous avons choisi la methode  de choix des centre initiaux
et le nombre des clusters et laisse le reste par defaut.
"""

kmeans = KMeans("kfarther",k=3)
centers , accuracies ,SSEs , min_acc = kmeans.fit(X_train,y_train)

"""
cette condition a pour but de choisir les caracteristiques (features)
que nous voulons utiliser pour dessiner le graph.
si il y a plus que deux on choisi les deux premiers  
"""
if(len(centers[0]) > 2):
    center_x,center_y,_ = zip(*centers)
else:
    center_x,center_y = zip(*centers)


L = kmeans.predict(X_test)
acc = calc_accuracy(y_test,L)



# display results

print("train acc :",min_acc)
print("test  acc : ",acc)
print("accuracies : ",accuracies)
print("errors : ",SSEs)

# graph

iters = list([i for i in range(len(accuracies))])
fig,axes = plt.subplots(nrows=2,ncols=2)
axes[0,0].plot(iters,accuracies,'m|--')
axes[0,0].set_title("accuracy")
axes[0,0].set_ylabel("accuracy")
axes[0,1].plot(iters,SSEs,'m|--')
axes[0,1].set_ylabel("error")
axes[0,1].set_title("sum squared error")
axes[1,0].scatter(X_train[:,0],X_train[:,1],c=y_train)
axes[1,0].set_title("predicted train")
axes[1,0].scatter(center_x,center_y,c='r',label="centers")
axes[1,0].legend()
axes[1,1].scatter(X_test[:,0],X_test[:,1],c=L)
axes[1,1].scatter(X_test[:,0],X_test[:,1],c=L)
axes[1,1].set_title("predicted test")
plt.show()