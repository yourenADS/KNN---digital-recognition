import numpy as np
import operator
from PIL import Image
from os import listdir

# 全局变量，数据大小
sizeof_data_base = 0

# 增加新样本的个数


def get_new_data_num():
    file = open('new_data_num.txt')
    num=file.readline()
    new_data_num = (int)(num)
    file.close()
    return new_data_num

def upddate_new_data_num():
    new_data_num=get_new_data_num()
    file = open('new_data_num.txt','w')
    new_data_num = new_data_num+1
    file.write((str)(new_data_num))
    file.close()
    return 


def ima_line(filename):
    # 创建1*1024的0向量
    line_num = np.zeros((1,1024))
    file = open(filename)
    for i in range(32):
        temp =  file.readline()
        for j in range(32):
            # 由于处理的数据为char类型所以强转为Int类型
            line_num[0,i*32 + j] = int(temp[j])
    #返回的也是一个矩阵，但是是一个1 * 1024 的矩阵
    file.close()
    return line_num



# 创建training的label
def creat_data_label():
    trainingdata = listdir('trainingDigits')
    global sizeof_data_base 
    sizeof_data_base = len(trainingdata)
    return_label = []
    for i in range(sizeof_data_base):
        # 添加label
        label = int(trainingdata[i].split('_')[0])
        return_label.append(label)
    return return_label



# 根据 trainingDigits 创建一个size * 1024 的数据库
def creat_data_base():
    # 返回数据集之中每个数据的名称
    trainingdata = listdir('trainingDigits')
    # 返回数据集的个数
    global sizeof_data_base 
    sizeof_data_base = len(trainingdata)
    return_data_base = np.zeros((sizeof_data_base,1024))
    for i in range(sizeof_data_base):
        filename = trainingdata[i]
        return_data_base[i,:]= ima_line('trainingDigits/%s' %(filename))
    return return_data_base 


# 将一个测试集与数据集做比较
def chose(test_data , training_data):
    #将1 * 1024 的矩阵转为 size * 1024 的矩阵
    global sizeof_data_base 
    sizeof_data_base= len(training_data)
    change_test_data = np.tile(test_data,(sizeof_data_base, 1))
    ans = change_test_data - training_data
    ans = ans**2
    row_sum=ans.sum(axis=1)
    row_sum = row_sum**0.5
    row_sum = row_sum.argsort()
    return row_sum
    

# 返回一个测试值的预测label
def judge_test_label(test_data ,data_base,k):
    label = creat_data_label()
    k_number = chose(test_data,data_base)
    sorted_ans = [0,0,0,0,0,0,0,0,0,0,0,0]
    for i in range(k):
        label_i = label[k_number[i]]
        sorted_ans[label_i] += 1
    max = 0
    ans = 0
    for i in range(1,11):
        if sorted_ans[i]>max:
            ans=i
    return ans

def copy(test_data , label):
    new_data_num = get_new_data_num()
    file = open('trainingDigits/%d_new_%d_txt'%(label,new_data_num),'w')
    for i in range(32):
        for j in range(32):
            if(test_data[0][i*32+j] == 0):
                file.write('0')
            else:
                file.write('1')
        file.write('\n')
    upddate_new_data_num()
    file.close()
    return 


# 实行测试
def test():
    k = 3
    test_file = listdir('testDigits')
    length = len(test_file)
    data_base=creat_data_base()
    error = 0
    for i in range(length):
        file = 'testDigits/' + test_file[i]
        test_data = ima_line(file)
        test_label = (int)(test_file[i].split('_')[0])
        project_label = judge_test_label(test_data,data_base,k)
        print("预测值:%d  实际值:%d" %(test_label,project_label))
        if test_label != project_label:
            error = error+1
            copy(test_data,test_label)
            print("error")
        else:
            print("successful")
    print('错误率:%lf' %(error/sizeof_data_base))


def rewrite_photo(filename):
    num = int(filename[7])
    img = Image.open(filename)
    #重新设置像素格式
    img = img.resize((32,32))
    #设置黑白图片
    img = img.convert('L')
    temp_image = open('photos/%dtest.txt' %(num),'w')
    for i in range(32):
        for j in range(32):
            color = img.getpixel((j,i))
            if color > 160:
                temp_image.write('0')
            else:
                temp_image.write('1')
        temp_image.write('\n')
    temp_image.close()

def test_photo():
    error = 0
    data_base=creat_data_base()
    for i in range(10):
        file = rewrite_photo('photos/%d.jpg' %(i))
        file = ima_line('photos/%dtest.txt' %(i))
        label = i
        project_label = judge_test_label(file,data_base,5)
        if(label == project_label):
            print("真实值:%d == 预测值:%d,successful"%(label , project_label))
        else:
            print("真实值:%d != 预测值:%d,error"%(label , project_label))
            copy(file,label)
            error=error+1
    print('错误个数:%d'%(error))
    return 
