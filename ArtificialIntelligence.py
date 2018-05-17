
# coding: utf-8

# 20170315課堂練習

# In[13]:


#abSumTest

a=5
b=3
a+b


# In[14]:


#mark
#單行註解

'''
多行
註解
'''


# In[15]:


#evalInputNumPrint

num1=eval(input("請輸入第一個數字:"))
num2=eval(input("請輸入第二個數字:"))
num3=eval(input("請輸入第三個數字:"))

total=num1+num2+num3
print(num1,"+",num2,"+",num3,"=",total)


# In[16]:


#abFirstSecondChang

a=1
b=10+a
print("first a =",a)
print("first b =",b)
a=b*a
print("second a =",a)
print("second b =",b)


# In[17]:


#平行設定

x,y,z=3,4,5
print("x =",x)
print("y =",y)
print("z =",z)


# In[18]:


#交換兩變數值

x,y=3,4
print("Before x =",x)
print("Before y =",y)
print("-----Change-----")
x,y=y,x
print("After x =",x)
print("After y =",y)


# In[19]:


#複合設定運算

x,y,z=1,2,3
x+=1    #x=x+1
y*=2    #y=y*2
z**=3    #z=z**3,**等於次方功能
print("x =",x,", y =",y,", z =",z,)


# In[20]:


#海龍公式計算面積

import math
a,b,c=3,4,5
s=(a+b+c)/2
area=math.sqrt(s*(s-a)*(s-b)*(s-c))    #使用math.sqrt函式開根號
area2=(s*(s-a)*(s-b)*(s-c))**0.5    #用**(指數)乘以0.5等同於math.sqrt，皆為開根號用法
print("面積(math.sqrt)為 = ",area)
print("面積(指數方式乘以0.5)為 = ",area2)
print("----------------------------")
import math
a,b,c=12,33,25
s=(a+b+c)/2
area=math.sqrt(s*(s-a)*(s-b)*(s-c))
print("面積為 = ",area)


# 20170322課堂練習

# In[21]:


#Python的動態類型繫合 變數使用前不需要宣告資料類型，使用時只要根據變數存放的資料決定其資料類型。

x=254
print("x=",x," type=",type(x))
x=254.0
print("x=",x," type=",type(x))
x=True
print("x=",x," type=",type(x))
x="write"
print("x=",x," type=",type(x))


# In[22]:


#2,8,16進位

print("二進位=",0o137)
print("八進位=",0b111)
print("十六進位=",0xff)


# In[23]:


#導入math函式庫

import math
print("面積=",(4*(math.pi*4.5*4.5*4.5)/3))


# In[24]:


#"不指定型態"導致浮點數產生誤差,與浮點數的存法有關係

x=3.141592627
print("x=",x)
print("x-3.14=",x-3.14)
print("2.1-2.0=",2.1-2.0)


# In[25]:


#導入matplotlib函式庫,其中pyplot是畫圖的函式,plot是畫折線圖的

import matplotlib.pyplot as pt    
#as=針對此函式進行命名,以後呼叫就可用此名稱進行呼叫
x=[1,2,3,4,5]
y=[7,2,3,5,9]
a=[0,1,2,3,4]
b=[0,1,2,3,4]
q=[0,2,4,1,3]
w=[5,3,1,4,2]
pt.plot(x,y)
pt.plot(a,b,"--",color="red",label="March")    
pt.plot(q,w,"^",color="green",label="Jun")   
#label是標籤,顯示該數列的標籤名字,但中文不可使用,亂碼
#"--"更改線為虛線;"^"更改為顯示點,點為三角形;"s"更改為顯示點,點為正方形
pt.legend()
#顯示label框框
pt.show() 
#顯示其結果


# In[26]:


#導入matplotlib函式庫,其中pyplot是畫圖的函式,bar是畫直條圖的

import matplotlib.pyplot as pt    
#as=針對此函式進行命名,以後呼叫就可用此名稱進行呼叫
x=[1,2,3,4,5]
y=[7,2,3,5,9]
pt.bar(x,y)
#導入變數座標數值:以"長條圖"的方式
pt.show()
#顯示其結果


# In[27]:


#導入matplotlib函式庫,,其中pyplot是畫圖的函式,scatter是畫點點圖的

import matplotlib.pyplot as pt    
#as=針對此函式進行命名,以後呼叫就可用此名稱進行呼叫
x=[1,2,3,4,5]
y=[7,2,3,5,9]
pt.scatter(x,y)
#導入變數座標數值:以"點"的方式
pt.show()
#顯示其結果


# In[28]:


#導入numpy函式庫引入了多維陣列以及可以直接有效率地操作多維陣列的函式與運算子,
#因此在NumPy上只要能被表示為針對陣列或矩陣運算的演算法,其執行效率幾乎都可以與編譯過的等效C語言程式碼一樣快。 
#使用random函式隨機畫點。

import numpy as np
import matplotlib.pyplot as pt    
#as=針對此函式進行命名,以後呼叫就可用此名稱進行呼叫
x=np.random.random(50)
y=np.random.random(50)
pt.scatter(x,y)
#導入變數座標數值:以"點"的方式
pt.show()
#顯示其結果


# In[29]:


#使用random函式隨機畫100000個點,當中發現問題:100000點結論會使整張佈滿點,等同於無法辨識之後的圖,無意義,不屬於大數據的巨量資料圖

import numpy as np
import matplotlib.pyplot as pt    
#as=針對此函式進行命名,以後呼叫就可用此名稱進行呼叫
x=np.random.random(100000)
y=np.random.random(100000)
pt.scatter(x,y)
#導入變數座標數值:以"點"的方式
pt.show()
#顯示其結果


# In[30]:


#Sin & Cos Wave 數值振幅圖

import numpy as np
import matplotlib.pyplot as pt    
x=np.arange(0,360)
y=np.sin(x*np.pi/180)
z=np.cos(x*np.pi/180)
pt.xlim(0,360)
#x軸的範圍:角度
pt.ylim(-1.2,1.2)
#y軸的範圍:數值
pt.title("Sin $ Cos Wave")
#設定標頭名稱
pt.xlabel("Degree")
#設定x軸標籤名稱
pt.ylabel("Value")
#設定y軸標籤名稱
pt.plot(x,y,label="Sin")
#Sin畫線
pt.plot(x,z,label="Cos")
#Cos畫線
#導入變數座標數值:以"線"的方式
pt.legend()
#顯示label框框
pt.show()
#顯示其結果


# 20170329課堂練習

# In[33]:


#scikit-learn 機器學習模組,測試資料集datasets子套件
#iris鳶尾花資料集

from sklearn import datasets
iris = datasets.load_iris()
iris


# In[34]:


#scikit-learn 機器學習模組,測試資料集datasets子套件
#iris鳶尾花資料集

from sklearn import datasets
iris = datasets.load_iris()
print(iris["DESCR"]) #資料集描述資料
print(iris["feature_names"]) #資料集的欄位名稱
print(iris["data"]) #資料集的資料
print(iris["target"]) #資料集的分類結果


# 20170414課堂練習

# In[35]:


#sklearniris函式庫 KMeans演算法分3類 silhouette_評估函數好壞(越接近1越好)

from sklearn import datasets
boston=datasets.load_boston()
print(boston.DESCR)
print(boston.target) #等同於print(boston["target"])
print(boston.data) #等同於print(boston["data"])
#CRIM(犯罪率) ZN(房屋大於25000ft比率)
#INDUS(住宅比率) CHAS(有無臨河) NOX(空屋比率) RM(房間數)
#AGE(自有住宅比例) DIS(離市中心距離) RAD(離高速公路距離)
#TAX(房屋稅率) PTRATIO(小學老師比例) B(黑人比率)
#LSTAT(低收入戶比率) MEDV(受雇者收入)

from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict
lr=linear_model.LinearRegression() #做線性回歸分析
predict=cross_val_predict(lr, boston.data, boston.target, cv=10) #做十份練習,一分作為訓練集
print(predict)

import matplotlib.pyplot as plt #對於房價做預測分布,也可用svm去預測,準確率最高
plt.figure()
plt.scatter(boston.target, predict)
y=boston.target
plt.plot([y.min(),y.max()],[y.min(), y.max()], 'k--', lw=4) #畫出預測參考線
plt.plot()
plt.show()


# In[36]:


#利用for迴圈比較當n_clusters分k類,評估函數的數值高低(用長條圖表示)

from sklearn import datasets,cluster,metrics
import matplotlib.pyplot as plt
iris=datasets.load_iris()
silhouette_avgs=[]    #silhouette_avgs因要存很多個,所以用陣列
ks=range(2,10)
for k in ks:
    iris_kmeans=cluster.KMeans(n_clusters=k).fit(iris["data"])
    silhouette_avg=metrics.silhouette_score(iris["data"],iris_kmeans.labels_)
    silhouette_avgs.append(silhouette_avg)
plt.bar(ks,silhouette_avgs)
plt.show()


# In[37]:


#datasets.loaddigits剖析圖片(一張圖88，其中每格中還分為44，所以實際上一張圖是32*32) 
#陣列 [[ 0. 0. 5. ..., 0. 0. 0.] [ 0. 0. 0. ..., 10. 0. 0.] [ 0. 0. 0. ..., 16. 9. 0.] ..., 
#     [ 0. 0. 1. ..., 6. 0. 0.] [ 0. 0. 2. ..., 12. 0. 0.] [ 0. 0. 10. ..., 12. 1. 0.]] 
#以下表示為以上的縱列 [0 1 2 ..., 8 9 8] 即第一列為0、第二列為1、第三列為2...第N-2列為8、第N-1列為9、第N列為8

from sklearn import datasets
digits=datasets.load_digits()
print(digits["DESCR"])    #DESCR是指描述
print(digits["data"])    #DESCR是指資料
print(digits["target"])


# In[38]:


#利用SVM方式存文字為圖片

from sklearn import datasets
digits=datasets.load_digits()
#print(digits["DESCR"])    #DESCR是指描述
#print(digits["data"])    #DESCR是指資料
#print(digits["target"])
plt.figure(1,figsize=(3,3))    #figsize為圖片尺寸
plt.imshow(digits.images[0],cmap=plt.cm.gray_r,interpolation='nearest')    
#imshow=image show,images[0]表示第一張圖(也可寫23456...),cmap為選擇顏色(gray為灰色),後面加_r是因為要reverse反轉,不然一般是黑底白字
plt.show()


# 20170510課堂練習

# In[39]:


#sklearn datasets_boston

from sklearn import datasets
boston=datasets.load_boston()
print(boston.DESCR)
print(boston.target) #等同於print(boston["target"])
print(boston.data) #等同於print(boston["data"])
#CRIM(犯罪率) ZN(房屋大於25000ft比率)
#INDUS(住宅比率) CHAS(有無臨河) NOX(空屋比率) RM(房間數)
#AGE(自有住宅比例) DIS(離市中心距離) RAD(離高速公路距離)
#TAX(房屋稅率) PTRATIO(小學老師比例) B(黑人比率)
#LSTAT(低收入戶比率) MEDV(受雇者收入)

from sklearn import linear_model
from sklearn.cross_validation import cross_val_predict
lr=linear_model.LinearRegression() #做線性回歸分析
predict=cross_val_predict(lr, boston.data, boston.target, cv=10) #做十份練習,一分作為訓練集
print(predict)

import matplotlib.pyplot as plt #對於房價做預測分布,也可用svm去預測,準確率最高
plt.figure()
plt.scatter(boston.target, predict)
y=boston.target
plt.plot([y.min(),y.max()],[y.min(), y.max()], 'k--', lw=4) #畫出預測參考線
plt.plot()
plt.show()


# 20170517課堂練習

# In[40]:


#DecisionTree_iris

from sklearn.datasets import load_iris
from sklearn import tree
from sklearn.cross_validation import train_test_split
from sklearn import metrics
iris = load_iris()
iris_X = iris.data
iris_y = iris.target
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(iris.data, iris.target)
test_y_predicted = clf.predict(test_X)
print(test_y_predicted)
print(test_y)
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)

from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn import metrics

# 讀入鳶尾花資料
iris = load_iris()
iris_X = iris.data
iris_y = iris.target

# 切分訓練與測試資料
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)

# 建立分類器
clf = neighbors.KNeighborsClassifier()
iris_clf = clf.fit(train_X, train_y)

# 預測
test_y_predicted = iris_clf.predict(test_X)
print(test_y_predicted)

# 標準答案
print(test_y)
accuracy = metrics.accuracy_score(test_y, test_y_predicted)
print(accuracy)

from sklearn.datasets import load_iris
from sklearn import neighbors
from sklearn.cross_validation import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt

# 讀入鳶尾花資料
iris = load_iris()
iris_X = iris.data
iris_y = iris.target

# 切分訓練與測試資料
train_X, test_X, train_y, test_y = train_test_split(iris_X, iris_y, test_size = 0.3)

# 選擇 k
range = np.arange(1, round(0.2 * train_X.shape[0]) + 1)
accuracies = []

for i in range:
    clf = neighbors.KNeighborsClassifier(n_neighbors = i)
    iris_clf = clf.fit(train_X, train_y)
    test_y_predicted = iris_clf.predict(test_X)
    accuracy = metrics.accuracy_score(test_y, test_y_predicted)
    accuracies.append(accuracy)

# 視覺化
plt.scatter(range, accuracies)
plt.show()
appr_k = accuracies.index(max(accuracies)) + 1
print(appr_k)


# In[1]:


#face_completion

from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
data=datasets.fetch_olivetti_faces()
#print(data.DESCR)
#print(data.target)
#print(data.data)
#plt.imshow(data.images[0],cmap='gray',interpolation='nearest')
#plt.show()
#把影像變成一列
targets=data.target
data=data.images.reshape(len(data.images),-1)
#訓練資料30張臉(300張圖片)，測試資料10張臉(100張圖片)
train=data[targets<30]
test=data[targets>=30]
# 從100張測試影像中,亂數選5張出來，變數test的大小變成(5,4096)
n_faces = 5
from sklearn.utils import check_random_state
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]
#把每張訓練影像和測試影像都切割成上下兩部分: X人臉上半部分
#， Y人臉下半部分。
n_pixels = data.shape[1]
X_train = train[:, :(n_pixels + 1) // 2]
y_train = train[:, n_pixels // 2:]
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]
#決定預測的演算法
from sklearn.linear_model import LinearRegression
ESTIMATORS = {
    "Linear regression": LinearRegression(),
}
y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train) #模型訓練
    y_test_predict[name] = estimator.predict(X_test) 
    #模型預測
# Plot the completed faces
image_shape = (64, 64)
n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)
for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))
    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1, title="true faces")
    sub.axis("off")
    sub.imshow(true_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest")
    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))
        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)
        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j, title=est)
        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape), cmap=plt.cm.gray, interpolation="nearest")
plt.show()


from sklearn import datasets
from sklearn.utils import check_random_state
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
data = datasets.fetch_olivetti_faces()
#print(data.DESCR)
#print(data.target)
#print(data.data)
targets = data.target
data = data.images.reshape((len(data.images), -1)) #把影像變成一列
train = data[targets < 30]
test = data[targets >= 30]
# 測試影像從100張亂數選5張出來，變數test的大小變成(5,4096)
n_faces = 5
rng = check_random_state(4)
face_ids = rng.randint(test.shape[0], size=(n_faces, ))
test = test[face_ids, :]

#把每張訓練影像和測試影像都切割成上下兩部分: X人臉上半部分， Y人臉下半部分。
n_pixels = data.shape[1]
X_train = train[:, :(n_pixels + 1) // 2]
# Lower half of the faces
y_train = train[:, n_pixels // 2:]
X_test = test[:, :(n_pixels + 1) // 2]
y_test = test[:, n_pixels // 2:]

ESTIMATORS = {
    "Linear regression": LinearRegression(),
}

y_test_predict = dict()
for name, estimator in ESTIMATORS.items():
    estimator.fit(X_train, y_train)
    y_test_predict[name] = estimator.predict(X_test)

# Plot the completed faces
image_shape = (64, 64)

n_cols = 1 + len(ESTIMATORS)
plt.figure(figsize=(2. * n_cols, 2.26 * n_faces))
plt.suptitle("Face completion with multi-output estimators", size=16)

for i in range(n_faces):
    true_face = np.hstack((X_test[i], y_test[i]))

    if i:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1)
    else:
        sub = plt.subplot(n_faces, n_cols, i * n_cols + 1,
                          title="true faces")

    sub.axis("off")
    sub.imshow(true_face.reshape(image_shape),
               cmap=plt.cm.gray,
               interpolation="nearest")

    for j, est in enumerate(sorted(ESTIMATORS)):
        completed_face = np.hstack((X_test[i], y_test_predict[est][i]))

        if i:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j)

        else:
            sub = plt.subplot(n_faces, n_cols, i * n_cols + 2 + j,
                              title=est)

        sub.axis("off")
        sub.imshow(completed_face.reshape(image_shape),
                   cmap=plt.cm.gray,
                   interpolation="nearest")

plt.show()

