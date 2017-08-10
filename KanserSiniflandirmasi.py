import numpy as np
import csv
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Dosyamızı açıp okuyoruz
file = open(r"breast-cancer-wisconsin.csv","r")
dataDict = csv.DictReader(file,delimiter=",")

# Dosyamızdaki sütunların isimleri
print(dataDict.fieldnames)

# Id sütununu ve keyleri cıkarıp 2D bir array oluşturuyoruz
data = []
for rowDict in dataDict:
    del rowDict["Id"]
    data.append( [ int(rowDict[v]) if rowDict[v].isdigit()else 0 for v in dataDict.fieldnames[1:] ] )


print("Örnek Veri: ",data[0])
print("Örnek Veri: ",data[10])
print("Örnek Veri: ",data[20])
print("Örnek Veri: ",data[30])

# Verimizi Numpy arraya çevirdik
data  = np.asarray(data)
print(data.shape)

# yapay zeka için Veri ve Sonuc larını ayırıyoruz
X = data[:,0:9]
Y = data[:,9:10]
print("Veri:",X[0],"Sonuc:",Y[0])

#------------------------------------------------------------------
print("Model Oluşturuluyor")
from keras.models import Sequential
from keras import layers

model = Sequential()
model.add(layers.Dense(32,activation=layers.activations.relu,input_shape=(9,)))
model.add(layers.Dense(32,activation=layers.activations.relu))
model.add(layers.Dense(32,activation=layers.activations.softmax))

model.compile(optimizer="adam",loss="sparse_categorical_crossentropy",metrics=["accuracy"])
#------------------------------------------------------------------

# Sonuc 2 ise İYİ-HUYLU, 4 ise KÖTÜ-HUYLU tümör demek
def ReturnResult(value):
    if(value == 2):
        return "İYİ-HUYLU"
    elif(value == 4):
        return "KÖTÜ-HUYLU"
    else:
        return "BELİRSİZ"

# Tekrar tekrar Eğitip tahminlerde bulunuyoruz
for q in range(5):
    print("Model Eğitiliyor:",q)
    model.fit(X,Y,batch_size=512,epochs=10,validation_split=0.1,verbose=2,shuffle=True)

    print("...Tahminler Yapılıyor...")
    for i in range(5):
        index = np.random.randint(600,len(X)-1)
        pred = model.predict_classes(X[index].reshape(1,9),verbose=0)

        if pred[0] == Y[index][0]:
            print('\033[92m'+"☑ Cevap:", ReturnResult(Y[index][0]),Y[index][0], "Tahmin:", ReturnResult(pred[0]),pred[0],'\033[0m')
        else:
            print('\033[91m'+"☒ Cevap:", ReturnResult(Y[index][0]),Y[index][0], "Tahmin:", ReturnResult(pred[0]),pred[0],'\033[0m')

