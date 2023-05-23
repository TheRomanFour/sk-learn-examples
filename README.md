# Tensor flow neural network examples


 Ucitavanja tensorflowa i keras-a
```
import tensorflow as tf
from tensorflow import keras
```

Učitavanje baze sa keras datasets - fashion i podjela na train i test set
```
baza=keras.datasets.fashion_mnist
(X_train,y_train), (X_test,y_test)=baza.load_data()

```

```
y_train.shape
    -> array(0-9)
```

Imenovanje svakog od y_train shape classe u čitljivo sto predstavlja
```
class_names=["T-shirt/top", "Trouser", "Pullover","Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```

Vraća čizma npr ako y_train[0] odgovara broju uz cizmu


```class_names[y_train[0]]```

Crtanje slike od X [0] 
```
import matplotlib.pyplot as plt
plt.imshow(X_train[0], cmap='gray')
```
Normalizacija
```
x_train = x_train/255
```
Crtanje 150- 155 slike:

```
for i in range(0,5):
  plt.subplot(151+i)
  plt.imshow(X_valid[i], cmap='gray')
  plt.title(class_names[y_valid[i]])
plt.show()
```

Stvaranje modela

```
model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
model.layers
model.summary()
```

Crtanje modela
```
keras.utils.plot_model(model, show_shapes=True)
```

Kompalacija modela (compile)



```
model.compile(loss="sparse_categorical_crossentropy", 
              optimizer="sgd",
              metrics=["accuracy"])
              
```
Treniramo model

```
history=model.fit(X_train,y_train,batch_size=32, epochs=20, validation_data=(X_valid, y_valid))
```


Crtanje podatka o modelu precinost i slicno 

```
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.ylim(0,1)
plt.show()
```

Testiranje i predikcije modela

```
model.evaluate(X_test,y_test)
predikcije=model.predict(X_test)
```

```
import numpy as np
y_pred=np.argmax(predikcije,axis=-1)
```

Klasifikacijsko izvješće
```
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))
```


## Second part

stvaranje modela CNN neurosnke mreže
```
model=keras.models.Sequential()
model.add(keras.layers.Conv2D(32,(3,3), activation='relu',padding='same',input_shape=(32,32,3)))
model.add(keras.layers.Conv2D(32,(3,3), activation='relu',padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))
model.add(keras.layers.Dropout(0.25))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(512, activation='relu'))


model.add(keras.layers.Dropout(0.5))
model.add(keras.layers.Dense(10, activation='softmax'))


model.summary()
```

Compile 

```
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
```

treinranie

```
history=model.fit(x_train,y_train_one_hot, batch_size=32, epochs=20,validation_split=0.2)
```



Crtanje modela
```
import pandas as pd
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.grid(True)
plt.xlabel('Epochs')
plt.show
˙``


Testiranje modela 

```
model.evaluate(x_test,y_test_one_hot)
```

Testiranje na vlasitoj slici

```
!curl -o macka.jpg https://www.kucni-ljubimci.com/wp-content/uploads/2017/04/Sretna-maca.jpg



slika = plt.imread("/content/macka.jpg")


slika.shape



from skimage.transform import resize  
slika_resized = resize(slika, (32,32))




Predivadanje za nasu sliku
```
import numpy as np
vjerojatnosti = model.predict(np.array( [slika_resized,] )) 



broj_u_klase=['avion','automobil','ptica','mačka','jelen','pas','zaba','konj','brod', 'kamion']


index = np.argsort(vjerojatnosti[0,:])




for i in range (9,5,-1):  #prvih nekoliko vjerojatnosti
  print(broj_u_klase[index[i]], ":", vjerojatnosti[0,index[i]])
```




## Encoder


```
import tensorflow
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
```


```
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
```

Encoder
```
encoder=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(30, activation="relu")   #bottleneck
])
```

Decoder
```
decoder=keras.models.Sequential([
    keras.layers.Dense(100, activation="relu",input_shape=[30]),
    keras.layers.Dense(28*28, activation="sigmoid"),
    keras.layers.Reshape([28,28])
])

```

Compile encoder 

```

stacked_autoencoder=keras.models.Sequential([encoder,decoder])

stacked_autoencoder.compile(loss="binary_crossentropy", optimizer='adam', metrics=['accuracy'])

history=stacked_autoencoder.fit(x_train,x_train, epochs=10,validation_data=[x_test,x_test])



```


Crtanje pravih vrijednosti i predivdanih

```
plt.figure(figsize=(20,5))

for i in range(8):
  plt.subplot(2,8,i+1)
  plt.imshow(x_test[i], cmap="binary")

  plt.subplot(2,8,8+1+i)
  pred=stacked_autoencoder.predict(x_test[i].reshape(1,28,28))
  plt.imshow(pred.reshape(28,28),cmap="binary")
  ```
  
  
 Decoder predict
  
```

plt.figure(figsize=(10,5))

plt.subplot(1,3,1)
plt.imshow(x_test[0], cmap="binary")

plt.subplot(1,3,2)
latent_vector=encoder.predict(x_test[0].reshape(1,28,28))
plt.imshow(latent_vector, cmap="binary")

plt.subplot(1,3,3)
pred=decoder.predict(latent_vector)
plt.imshow(pred.reshape(28,28),cmap="binary")

```


Sum ili noise

```
plt.figure(figsize=(10,5))

plt.subplot(1,2,1)
plt.imshow(x_test[0], cmap="binary")

plt.subplot(1,2,2)
noise=np.random.random((28,28))/4
plt.imshow(x_test[0]+noise, cmap="binary")
```


Dodavanje suma na x_train

```
x_train_noise=x_train+((np.random.random(x_train.shape))/4)
x_test_noise=x_test+((np.random.random(x_test.shape))/4)
```

onda isti postupak treniranja-- history compile i to


Autoencoder se rijesava suma kad prediva sada

```
plt.figure(figsize=(20,5))

for i in range(8):
  plt.subplot(2,8,i+1)
  plt.imshow(x_test_noise[i], cmap="binary")

  plt.subplot(2,8,8+1+i)
  pred=stacked_autoencoder.predict(x_test_noise[i].reshape(1,28,28))
  plt.imshow(pred.reshape(28,28),cmap="binary") 
  
```
