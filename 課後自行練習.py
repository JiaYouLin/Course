
# coding: utf-8

# 自行練習

# In[2]:


#MustHtmlRequests

import requests
c=requests.get("http://www.must.edu.tw/")
print(c.text)


# In[3]:


#ForRangeTest

for i in range(1,11):
    print(i)
    if i == 5:
        break
print("迴圈結束")


# In[5]:


#LeapYear_Blockly

tmp = None
year = None

def text_prompt(msg):
    try:
        return raw_input(msg)
    except NameError:
        return input(msg)


year = float(text_prompt('請輸入西元年'))
tmp = year % 100
if False:
    tmp = year % 400
    if tmp == 0:
        print('是閏年')
    else:
        print('不是閏年')
else:
    tmp = year % 4
    if tmp == 0:
        print('是閏年')
    else:
        print('不是閏年')


# In[6]:


#MathLibraryFunction

import math

num1=math.ceil(8.8)    #回傳比參數大1之值
num2=math.fabs(-5)    #abs絕對值,f浮點數
num3=math.factorial(5)    #factorial階乘,5!=5*4*3*2*1
print("ceil(8.8)=",num1," , fabs(-5)=",num2," , factorial(5)",num3)
print("---------------------------------------------------------")

num4=math.floor(4.3)    #回傳比參數小1之值
num5=math.floor(-5.5)
print("floor(4.3) =",num4," , floor(-5.5)=",num5)
print("---------------------------------------------------------")

num6=math.gcd(25,155)    #回傳兩值最大公因數
print("gcd(25,155) =",num6)
print("---------------------------------------------------------")

num7=math.exp(1)    #回傳自然對數（底數e）的次方數乘積
num8=math.exp(5) 
print("exp(1) =",num7)
print("exp(5) =",num8)
print("---------------------------------------------------------")

#math.log(x[,base])傳回正數值參數x的自然對數值，預設底數為e，若要設定底數可加上選擇性參數[,base]
num9=math.log(2)    #回傳自然對數（底數e）的次方數乘積
num10=math.log(2,2)    #回傳自然對數（底數e）的次方數乘積
print("log(2) =",num9)
print("log(2,2) =",num10)
print("---------------------------------------------------------")

num11=math.sqrt(4)    #sqrt(4)=4開根號=(4)**0.5=4的指數為0.5次方
num12=(4)**0.5
print("sqrt(4) =",num11)
print("(4)**0.5 =",num12)
print("---------------------------------------------------------")

num13=math.radians(45)    #回傳參數(x)由角度轉換成弳度的結果，公式：弳度=角度*pi/180
print("radians(45) =",num13)
print("---------------------------------------------------------")

num14=math.degrees(0.7853981633974483)    #回傳參數(x)由弳度轉換成角度的結果，公式：角度=弳度*180/pi
print("degrees(0.7853981633974483) =",num14)
print("---------------------------------------------------------")

#注意：參數x必須為弳度而非角度，也就表示若求sin及cos就必須先根據公式：弳度=角度*pi/180，將角度轉換成弳度
sin1=math.sin(30*math.pi/180)    #求sin等同於math.sin(math.radians(30))
print("sin(30*math.pi/180)=",sin1)
cos1=math.cos(30*math.pi/180)    #求cos等同於math.cos(math.radians(30))
print("cos(30*math.pi/180)=",cos1)
print("---------------------------------------------------------")

finite1=math.isfinite(1000000)    #isfinite(x)傳回數值參數(x)是否為有限
print("isfinite(1000000)=",finite1)
print("---------------------------------------------------------")

inf1=math.isinf(1000000)    #isinf(x)傳回數值參數(x)是否為無限
inf2=math.isinf(-math.inf)
print("isinf(1000000)=",inf1)
print("isinf(-math.inf)=",inf2)
print("---------------------------------------------------------")

nan1=math.isnan(10)    #isnan(x)傳回數值參數(x)是否為NaN(not a number)
nan2=math.isnan(math.nan)
print("isnan(10)=",nan1)
print("isnan(math.nan)=",nan2)
print("---------------------------------------------------------")


# In[7]:


# 亂數函式

import random

num1=random.randint(1,10)    #回傳介於兩數之間（大於等於前者、小於等於後者）的數，每次呼叫之回傳值不一定相同
print("randint(1,10)=",num1)
num2=random.randint(1,10)
print("randint(1,10)=",num2)
num3=random.randint(1,10)
print("randint(1,10)=",num3)
print("---------------------------------------------------------")

num4=random.random()    #傳回一個大於等於0.0、小於等於1.0的隨機浮點數，每次呼叫之回傳值不一定相同
print("random()=",num4)
num5=random.random()
print("random()=",num5)
print("---------------------------------------------------------")

L=[1,2,3,4,5]    #變數L是一個包含五個元素的串列
random.shuffle(L)    #shuffle(L)是將L中的元素隨機重排
print("shuffle(L)=",L)    #印出L
print("---------------------------------------------------------")


# In[8]:


# 小型猜密碼遊戲

import random    #匯入random模組

num=random.randint(1,3)    #隨機產生一個範圍介於1~3的整數並指派給變數num
answer=eval(input("請猜數字1~3："))
print("您的數字是：",answer)
print(num,"==",answer,"is",num==answer)    #印出兩者比較結果，True表示猜中，False表示猜錯


# In[9]:


#下數兩種印星星方式差別在於中間的逗號與加號,用逗號會在IDEL中前端的空白,加號連接就不會

n=eval(input("請輸入高度"))
for i in range(1,n+1):
    print(" "*(n-i),"*"*(2*i-1))

n=eval(input("請輸入高度"))
for i in range(1,n+1):
    print(" "*(n-i)+"*"*(2*i-1))


# In[10]:


#List_NewSubtotalInListFinal

grades=[[95,100,100],[86,90,75],[98,98,96],[78,90,80],[70,68,72]]

for i in range(5):
    subTotal=0
    for j in range(3):
        subTotal+=grades[i][j]
    grades[i].append(subTotal)
    
for i in range(5):
    print("學生",i+1,"的總分為",grades[i][3])


# In[11]:


#List_Matrix

mataix=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
print(mataix)


# In[13]:


#List_InputMatrix

matrix=[]
rows=eval(input("請輸入矩陣的列數:"))
cols=eval(input("請輸入矩陣的行數:"))


for i in range(rows):
    matrix.append([])
    for j in range(cols):
        element=eval(input("請輸入矩陣的元素(由上往下逐一輸入):"))
        matrix[i].append(element)
        
print(matrix)


# In[14]:


#List_UseDefPrintMatrix

matrix=[[1,2,3],[4,5,6],[7,8,9],[10,11,12]]
#定義printMatrix()方法用來印出矩陣
def printMatrix(matrix):
    for i in range(len(matrix)):
        print("Test print matrix=",matrix[i])   #逐列(i)取出matrix len
        for j in range(len(matrix[i])):
            print(matrix[i][j],'\t',end='')     #在列(i)中,逐行(j)取出matrix len
            #end=''是避免print後直接換行，可達到不換行作用
        print('\n')    #排版用,新增空的一行拉開兼間距
#呼叫printMatrix()方法印出矩陣
printMatrix(matrix)


# In[16]:


#ReversalString

Engstr=input("請輸入一個英文單字:")
Engchar=list(Engstr)
Engchar.reverse()
print(''.join(Engchar))


# In[17]:


#InputNScore

n=eval(input("你要輸入幾筆成績?"))
total=0
score=list()
for i in range(int(n)):
    scoreTemp=eval(input("請輸入第"+str(i)+"筆成績:"))
    score.append(scoreTemp)
    total=total+score[i]
print("總分為:",total)


# In[29]:


#TurtleNineStaircases

import turtle

turtle.showturtle()
turtle.penup()
turtle.setpos(-280,-230)
turtle.pendown()
print("階梯起始位置:",turtle.position())
turtle.forward(50)
for i in range(9):
    turtle.left(90)
    turtle.forward(50)
    turtle.right(90)
    turtle.forward(50)
x=-210
y=-210
turtle.penup()
turtle.setpos(x,y)
turtle.pendown()
for j in range(1,10): 
    turtle.penup()
    turtle.write(j,font=("Arial",12,"normal"))
    print("階梯第",j,"位置:",turtle.position())
    x+=50
    y+=50
    turtle.setpos(x,y)    
    turtle.pendown()


# In[31]:


#Factorial

def F(n):
    if n==0:
        return 1
    elif n>0:
        if n==1:
            print("1")
        else:
            print(str(n)+"*",end="")
        return n*F(n-1)
    else:
        return -1

n=eval(input("請輸入階乘數:"))
#print(n,"!=",F(n))
print(F(n))


# In[32]:


#ImputValue

# Python Program
number=int(input("請輸入一數值:"))
if number>=60:
    print("pass")
else:
    print("fail")


# In[33]:


#reverse

def reverse(data):
    return list(data[i] for i in range(len(data)-1, -1, -1))
        

data = input("請輸入一個英文單字：")
for char in reverse(data):
    print(char)


# In[34]:


#DegreeCToDegreeF

degreeC=eval(input("請輸入攝氏溫度:"))
degreeF=degreeC*1.8+32
print("攝氏",degreeC,"度可以轉換成華氏",degreeF,"度")         
if degreeC>=28:
    print("悶熱")
elif degreeC>=20:
    print("舒適")
else:
    print("寒冷")


# In[35]:


#99MultiplicationTable

result1,result2='',''
for i in range(1,10):
    result1=''
    for j in range(1,10):
        result1=result1+str(i)+'*'+str(j)+'='+str(i*j)+'\t'
    result2=result2+result1+'\n'

print(result2)


# In[37]:


#ScoreInput

score = eval(input("請輸入成績(0~100):"))
if score >= 80:
    print("A")
elif score >= 70:
    print("B")
elif score >= 60:
    print("C")
else:
    print("F")


# In[38]:


#ScoreInput

score = eval(input("請輸入成績(0~100):"))
if score >= 80:
    print("A")
else:
    if score >= 70:
        print("B")
    else:
        if score >= 60:
            print("C")
        else:
            print("F")


# In[39]:


#Leap

import calendar
for y in range(2015,2019):
    if (calendar.isleap(y)):
        print(str(y)+"是閏年")


# In[40]:


#StringToCharOnSpace

str="Hello,World"

for i in str:
    print(i,"-",end="")

