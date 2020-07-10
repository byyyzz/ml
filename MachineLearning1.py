import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB



df= pd.read_csv("ISIC_2019_Training_Metadata.csv")
df.head()


#istenmeyen lesion_id kolonunun silinmesi
df=df.drop("lesion_id" , axis=1)
df.head()


print (df.info())


np.sum(df.isna())

#veri setindeki null değerlerin silinmesi
df= df.dropna(axis=0)
print (df.isnull().sum())
print (df.shape)



df.head(80)


#veri setlerini birleştirme
dfg= pd.read_csv("ISIC_2019_Training_GroundTruth.csv")
typedf=pd.DataFrame([t for t in np.where(dfg==1,dfg.columns,"").flatten().tolist()
                    if len (t)>0], columns = (["TYPE"]))
dfyeni=pd.concat([df,typedf], axis=1)
dfyeni.head(20)



dfyeni.to_csv(r'/users/Acer/mlverisetison.csv',index=False)





#lezyon türleri grafiği
fig = plt.figure(figsize=(5,5))
dfyeni['TYPE'].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Lezyon Türleri")
print("")



dfyeni.groupby("TYPE")['image'].nunique().plot(kind='bar',color="orange")
plt.xticks(rotation=0)
plt.xlabel("Lezyon Türleri")
plt.ylabel("Sayılar")
plt.show()


#lezyonun bulunduğu bölge grafik
fig = plt.figure(figsize=(5,5))
dfyeni["anatom_site_general"].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Lezyonun Bulunduğu Bölge")
print("")


dfyeni.groupby("anatom_site_general")['image'].nunique().plot(kind='bar', color="orange")
plt.xticks(rotation=0)
plt.xlabel("Lezyonun Bulunduğu Bölge")
plt.ylabel("Sayılar")
plt.show()



#cinsiyet dağılımı grafik
fig = plt.figure(figsize=(5,5))
dfyeni["gender"].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Cinsiyet Dağılımı")
print("")

dfyeni.groupby("gender")['image'].nunique().plot(kind='bar',color="orange")
plt.xticks(rotation=0)
plt.xlabel("Cinsiyet")
plt.ylabel("Sayılar")
plt.show()

#yaş grafik
fig = plt.figure(figsize=(5,5))
dfyeni["age_approx"].value_counts().plot(kind = 'pie', autopct='%.1f%%')
plt.ylabel(" ", fontsize = 15)
plt.title("Yaş Dağılımı")
print("")


dfyeni.groupby("gender")['image'].nunique().plot(kind='bar',color="orange")
plt.xticks(rotation=0)
plt.xlabel("Cinsiyet")
plt.ylabel("Sayılar")
plt.show()

#mel olanların cinsiyet dağılımları grafik
dfyeni[dfyeni["TYPE"]=="MEL"].groupby("gender")['image'].nunique().plot(kind='bar')
plt.xticks(rotation=0)
plt.xlabel("Cinsiyet")
plt.ylabel("Sayılar")
plt.tight_layout()
plt.show()




#mel olanların lezyon bölgesi grafik
dfyeni[dfyeni["TYPE"]=="MEL"].groupby("anatom_site_general")['image'].nunique().plot(kind='bar')
plt.xticks(rotation=0)
plt.xlabel("Lezyonun Bulunduğu Bölge")
plt.ylabel("Sayılar")
plt.tight_layout()
plt.show()
​

dfyeni=dfyeni.dropna(axis=0)
print (df.isnull().sum())
print (df.shape)
​
#veriyi nümerik hale getirme
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
dfyeni["TYPE"]=le.fit_transform(dfyeni["TYPE"])


le = preprocessing.LabelEncoder()
dtype_object=dfyeni.select_dtypes(include=['object'])
print (dtype_object.head())
for x in dtype_object.columns:
    dfyeni[x]=le.fit_transform(dfyeni[x])
​
print (dfyeni.head())

​
#verinin test,eğitim diye ayrılması ve ölçeklendirilmesi
X = dfyeni.iloc[:,:4].values
y = dfyeni["TYPE"].values
​
​
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 20)
​
​
​
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)



score=[]
algorithms=[]
​
​
#KNN algoritması
from sklearn.neighbors import KNeighborsClassifier
​
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,y_train)
knn.predict(X_test)
score.append(knn.score(X_test,y_test)*100)
algorithms.append("KNN")
print("KNN accuracy =",knn.score(X_test,y_test)*100)
​

from sklearn.metrics import confusion_matrix
y_pred=knn.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)
​

f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title(" KNN Confusion Matrix")
plt.show()
​
from sklearn.metrics import classification_report
​
target_names=["NV","MEL","BCC","AK","BKL","DF","VASC","SCC"]
print(classification_report(y_true, y_pred, target_names=target_names))


#naive bayes algoritması
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(X_train,y_train)
​
score.append(nb.score(X_test,y_test)*100)
algorithms.append("Navie-Bayes")
print("Navie Bayes accuracy =",nb.score(X_test,y_test)*100)
​
​
from sklearn.metrics import confusion_matrix
y_pred=nb.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)
​
​
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Navie Bayes Confusion Matrix")
plt.show()
target_names=["NV","MEL","BCC","AK","BKL","DF","VASC","SCC"]
print(classification_report(y_true, y_pred, target_names=target_names))


#Decision Tree algoritması
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train,y_train)
print("Decision Tree accuracy:",dt.score(X_test,y_test)*100)
score.append(dt.score(X_test,y_test)*100)
algorithms.append("Decision Tree")
​
​
from sklearn.metrics import confusion_matrix
y_pred=dt.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)
​
​
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Decision Tree Confusion Matrix")
plt.show()
target_names=["NV","MEL","BCC","AK","BKL","DF","VASC","SCC"]
print(classification_report(y_true, y_pred, target_names=target_names))



#Logistic Regression algoritması
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver='lbfgs')
lr.fit(X_train,y_train)
score.append(lr.score(X_test,y_test)*100)
algorithms.append("Logistic Regression")
print("Logistic Regression accuracy {}".format(lr.score(X_test,y_test)))
​
​
from sklearn.metrics import confusion_matrix
y_pred=lr.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)
​
​
f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Logistic Regression Confusion Matrix")
plt.show()
target_names=["NV","MEL","BCC","AK","BKL","DF","VASC","SCC"]
print(classification_report(y_true, y_pred, target_names=target_names))


#SVM algoritması
from sklearn.svm import SVC
svm=SVC(random_state=10)
svm.fit(X_train,y_train)
score.append(svm.score(X_test,y_test)*100)
algorithms.append("Support Vector Machine")
print("svm test accuracy =",svm.score(X_test,y_test)*100)
​

from sklearn.metrics import confusion_matrix
y_pred=svm.predict(X_test)
y_true=y_test
cm=confusion_matrix(y_true,y_pred)
​

f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Support Vector Machine Confusion Matrix")
plt.show()
target_names=["NV","MEL","BCC","AK","BKL","DF","VASC","SCC"]
print(classification_report(y_true, y_pred, target_names=target_names))



#ANN algoritması
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
X = dfyeni.iloc[:,:4].values
print (X.shape[0])
y = dfyeni['TYPE'].values.reshape(X.shape[0], 1)
​

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)
​
sc = StandardScaler()
sc.fit(X_train)
X_train = sc.transform(X_train)
sc.fit(X_test)
X_test = sc.transform(X_test)
​
sknet = MLPClassifier(hidden_layer_sizes=(8), learning_rate_init=0.02, max_iter=100)
sknet.fit(X_train, y_train)
​
score.append(sknet.score(X_test,y_test)*100)
algorithms.append("Artificial Neural Networks")
print("Ann test accuracy =",svm.score(X_test,y_test)*100)
​
y_pred = sknet.predict(X_test)
y_true=y_test
​cm=confusion_matrix(y_true,y_pred)


f,ax=plt.subplots(figsize=(5,5))
sns.heatmap(cm,annot=True,linewidths=0.5,linecolor="red",fmt=".0f",ax=ax)
plt.xlabel("y_pred")
plt.ylabel("y_true")
plt.title("Artificial Neural Networks Confusion Matrix")
plt.show()
target_names=["NV","MEL","BCC","AK","BKL","DF","VASC","SCC"]
print(classification_report(y_true, y_pred, target_names=target_names))
​


#algoritmaların karşılaştırılması
print (algorithms)
print (score)
​
x_pos = [i for i, _ in enumerate(algorithms)]
​
plt.bar(x_pos, score, color='orange')
plt.xlabel("Algoritmalar")
plt.ylabel("Basari Yuzdeleri")
plt.title("Basari Siralamalar")
​
plt.xticks(x_pos, algorithms,rotation=90)
​
plt.show()
