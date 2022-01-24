#coding:utf-8
import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler #引入归一化的包


# 加载txt和csv文件
def loadtxtAndcsv_data(filename,split,dataType):
    return np.loadtxt(filename,delimiter=split,dtype=dataType)

#加载npy文件
def loadnpy_data(fileName):
    return np.load(fileName)



def linearRegression():
    print("加载数据...\n")
    data = loadtxtAndcsv_data("data.txt",",",np.float64)#读取数据
    X = np.array(data[:,0:-1],dtype=np.float64)  #x对应0到倒数第二列
    y = np.array(data[:,-1],dtype=np.float64)   #y对应最后一列

    #归一化操作
    sclaer = StandardScaler()
    sclaer.fit(X)
    x_train = sclaer.transform(X)
    x_test = np.array([1650,3])
    x_test = sclaer.transform(x_test.reshape(1,-1))
    print("x_test:",x_test)


    #线性模型拟合
    model = linear_model.LinearRegression()
    model.fit(x_train,y)

    #预测结果
    result = model.predict(x_test)
    print(model.coef_)
    print(model.intercept_)
    print(result)

if __name__ == "__main__":
    data = linearRegression()
    # print(data)