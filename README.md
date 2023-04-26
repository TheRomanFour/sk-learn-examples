# Strojno skripta

## Image manipulation
Download image using curl
```
!curl -o "raspored.png" "https://i.ibb.co/nn9hS1T/Raspored.png"
```

Getting image details

```
import cv2

image = cv2.imread('/content/raspored.png')

height, width, depth = image.shape

print("Njezina visina je", height, " sirina je ", width, " a dubina ", depth)
```

Showing the image 

```
from google.colab.patches import cv2_imshow
cv2_imshow(image)
```

Changing the pixel at location 130 130 to another color:

```
image[130,130]  = [0, 0, 0]
```

Reshaping image width and heigh and changing it to black/white

```
image2 = cv2.resize(image, (int(width*2), int(height/2) ) )
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
print(image2.shape)
cv2_imshow(image2)
```


## Files in colab

Open a file on colab as f and write in it
```
from google.colab import files

with open('/content/drive/MyDrive/Strojno ucenje/primjer.txt', 'w') as f:
  f.write('Pisanje u datoteku na moj drive file')

files.download('/content/drive/MyDrive/Strojno ucenje/primjer.txt')
```


## Import sklearn datasets (iris examples)

```
from sklearn import datasets
db = datasets.load_iris()

db
db.keys()
data = db.data
data[:]
```


Spliting the data to train test form

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(podaci, vrsta_cvijeta, test_size = 0.3, random_state = 42, )
```

Plot the data
```
stupac = baza_sk.feature_names

plt.title("Graf")
plt.plot(podaci[:20], label = stupac )
plt.legend(loc='upper left')
plt.show()
```

ROC Analysis

```from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, y_pred_LogR) #površina ispd krivulje
fpr, tpr, thresholds = roc_curve(y_test, y_pred_LogR)  #crtanje krivulje
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1], 'k--') #dijagonala
plt.axis([0, 1, 0, 1]) #
plt.xlabel('False Positive Rate = 1- Specificity')
plt.ylabel('True Positive Rate = Sensitivity')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
```


Fetch the data of images and transform them into array

```
from sklearn import datasets
baza_faces = datasets.fetch_openml('olivetti_faces')

baza_faces.keys()
baza_faces["data"]

```
```
import numpy as np 
X = np.array(baza_faces.data)
X
```
```
y = np.array(baza_faces.target)
y
```


## Generate your own data sata with random


```
coordinate_dots_between_1_100_15_of_them = np.random.randint(1,100, (15,2))
coordinate_dots_between_1_100_15_of_them
```


## KMeans implementationn

```
from sklearn.cluster import KMeans
kmeans_model = KMeans(n_clusters = 2,  random_state = 2, n_init = 1)
kmeans_model.fit(podaci)
grupa = kmeans_model.predict(podaci)
grupa
```

Finding the centers
```
centroidi = kmeans_model.cluster_centers_
centroidi
```

Ploting 
```
import matplotlib.pyplot as plt
plt.scatter(podaci[:,0], podaci[:,1], c = grupa, cmap = "cool")
plt.scatter(centroidi[:,0] , centroidi[:,1], c = 'red', s = 100, marker = 'x')
```

Predict the given data (dot) value of 300,200
 
 ```
 podatak = kmeans_model.predict([(300,200)])

 ```
 
Sillhouete score 
```
from sklearn import metrics
score = metrics.silhouette_score(podaci, grupa, metric = "euclidean" )
print('ilouetee metrik je %.3f ' % score)
```

Elbow method
```
from yellowbrick.cluster import KElbowVisualizer  
visualizer = KElbowVisualizer(kmeans_model, k=(2, 45), locate_elbow = True) 
visualizer.fit(podaci)
visualizer.show()
```


## Grouping - hierarchial

```
from scipy.spatial.distance import pdist
udaljenost = pdist(podaci)
udaljenost
```
Usage and visulation
```
from scipy.cluster.hierarchy import linkage, dendrogram
stablo = linkage(udaljenost)
dendrogram_crtanje = dendrogram(stablo)
```

Determination of groups based on cluster distance
```
from scipy.cluster.hierarchy import fcluster
#rezemo stablo na visini 4
po_udaljenosti = fcluster(stablo, 4, criterion = 'distance')
po_udaljenosti
plt.scatter(podaci[:,0], podaci[:,1], c = po_udaljenosti)
```

## Linear regresion

```
from sklearn import datasets
baza = datasets.fetch_openml("bodyfat")
baza.keys()
baza.data
```


```
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(X,y)
reg.coef_
reg.intercept_

y_predivdeno = reg.predict(X)

reg.score(X,y) #samo 9 posto podatak je na objasnjeno linearnom regresijom ili je 9 % tocno 100% nemamo ppojma

```

Prediction of person vs real prediction
```
visina_21_osobe = X[20]
visina_21_osobe = visina_21_osobe.reshape(-1,1)
visina_21_osobe
tezina_21_osobe = y[20]
predivanje_21 = round(y_predivdeno[20], 2)
pred = reg.predict(visina_21_osobe)
pred
print("21. OSOBA JE visioka {}, a njena prava tezzina je {}, a predivana tezina je {} ".format(visina_21_osobe, tezina_21_osobe, predivanje_21))
```
Mean square error
```
from sklearn.metrics import mean_squared_error
mean_squared_error(y,y_predivdeno)
```

 Ploting linear regresion
 ```
 import matplotlib.pyplot as plt
plt.scatter(X,y)
plt.xlabel("Visina u cm")
plt.ylabel("Tezina u kg")
plt.xlim(150,210)
plt.ylim(40, 180)
plt.plot(X,y_predivdeno, color = "red")
plt.show()
 ```


## Normalize data 

```
from sklearn.preprocessing import normalize
X_norm = normalize(X)
X_norm
```
