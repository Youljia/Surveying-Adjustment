'''
    1.平差:
        根据多余观测，在一定的函数模型以及数学准则的条件下，求得观测值的最或然值并对其进行精度评定
    2.平差方法:
        条件平差（adjustment with condition）
        附有参数的条件平差
        参数平差（parameter adjustment）
        附有限制条件的参数平差
'''


# 导入矩阵运算模块
import numpy as np
from numpy import linalg
import math

def calculate_Q(i, j, a, q, r):
    translate_ij = {'l': 0, 'w': 1, 'k': 2, 'v': 3, 'L': 4}
    try:
        i1 = translate_ij[i]
        j1 = translate_ij[j]
    except Exception as error:
        raise ValueError("库中不存在该基本向量的协因数阵")
    n = np.dot(np.dot(a, q), a.T)
    q33 = np.dot(np.dot(np.dot(np.dot(q, a.T), linalg.inv(n)), a), q)
    all_Q = [
        [q, np.dot(q, a.T), -np.dot(np.dot(q, a.T), linalg.inv(n)), -q33, (q-q33)],
        [np.dot(a, q), n, -np.identity(r), -np.dot(a, q), 0],
        [-np.dot(np.dot(linalg.inv(n), a), q), -np.identity(r), linalg.inv(n), np.dot(np.dot(linalg.inv(n), a), q), 0],
        [-q33, -np.dot(q, a.T), np.dot(np.dot(q, a.T), linalg.inv(n)), q33, 0],
        [q-q33, 0, 0, 0, q-q33]
    ]
    return all_Q[i1][j1]



def conditional_adjust(n, t, a, w, q, l):
    # 计算改正数v
    n1 = np.dot((np.dot(a, q)), a.T)
    k = np.dot(-linalg.inv(n1), w)
    v = np.dot(np.dot(q, a.T), k)

    # 计算最或然值
    l1 = v + l

    # 精度评定
    e2 = np.dot(np.dot(v.T, linalg.inv(q)), v) / (n - t)
    ql = q - np.dot(np.dot(np.dot(np.dot(q, a.T), linalg.inv(n1)), a), q)


    # 返回成果
    return (l1, math.sqrt(e2), ql)

def additional_conditional_adjust(n, t, a, b, w, q, l):
    # 计算改正数v
    n1 = np.dot(np.dot(a, q), a.T)
    m = np.dot(np.dot(b.T, linalg.inv(n1)), b)
    ux = np.dot(np.dot(b.T, linalg.inv(n1)), w)
    x_ = -np.dot(linalg.inv(m), ux)
    k = -np.dot(linalg.inv(n1), (np.dot(b, x_) + w))
    v = np.dot(np.dot(q, a.T), k)
    l1 = v + l

    # 精度评定
    e2 = np.dot(np.dot(v.T, linalg.inv(q)), v) / (n - t)
    qkk = linalg.inv(n1) - np.dot(np.dot(np.dot(np.dot(linalg.inv(n1), b), linalg.inv(m)), b.T), linalg.inv(n1))
    ql_ = q - np.dot(np.dot(np.dot(np.dot(q, a.T), qkk), a), q)

    return (l1, math.sqrt(e2), ql_)

def parameter_adjust(n, t, b, l, p, l0):
    # 计算最或然值
    n1 = np.dot(np.dot(b.T, p), b)
    w = np.dot(np.dot(b.T, p), l)
    x_ = np.dot(linalg.inv(n1), w)
    v = np.dot(b, x_) - l
    l_ = l0 + v

    # 精度评定
    e2 = np.dot(np.dot(v.T, p), v) / (n - t)
    qxx = linalg.inv(n1)

    return (l_, math.sqrt(e2), qxx)

def convert_to_seconds(i, j, k):
    return i*3600 + j*60 + k

def convert_to_angle(i):
    a = int(i / 3600)
    b = int((i % 3600) / 60)
    c = int(i - a * 3600 - b * 60)
    return ("{}.{}.{}.".format(a, b, c))

if __name__ == "__main__":
    """
    # 条件平差测试用例
    a = np.matrix([[1, 1, 0, -1, 0], [0, 1, -1, 0, 1]])
    n = 5
    t = 3
    h1 = 2.42
    h2 = 16.14
    h3 = 50.56
    h4 = 18.62
    h5 = 34.35
    value = [[h1], [h2], [h3], [h4], [h5]]
    l = np.array(value)
    w = np.matrix([[h1+h2-h4], [h2-h3+h5]])
    q = np.diag([2, 1, 1, 1, 1])
    results = conditional_adjust(n, t, a, w, q, l)
    print(results)
    """
    '''
    # 附有参数的条件平差测试用例
    n = 4
    t = 3
    a = np.matrix([[1, 1, -1, 0], [0, 1, 0, -1]])
    b = np.matrix([[0], [1]])
    value = [[-3], [convert_to_seconds(20, 30, 30)]]
    w = np.array(value)
    q = np.eye(4)
    l = np.matrix([[convert_to_seconds(30, 20, 20)], [convert_to_seconds(20, 10, 10)], [convert_to_seconds(50, 30, 33)], [convert_to_seconds(40, 40, 40)]])
    result = additional_conditional_adjust(n, t, a, b, w, q, l)
    l_ = []
    for i in (0, 1, 2, 3):
        l_.append(convert_to_angle(result[0][i]))
    print(l_)
    '''

    # 参数平差测试用例
    n = 3
    t = 2
    b = np.matrix([[-1, 0], [0, 1], [-1, 1]])
    l = np.matrix([[0], [0], [-0.07]])
    p = np.diag([2, 1, 4])
    l0 = np.matrix([[10.4], [24.52], [34.85]])
    print(parameter_adjust(n, t, b, l, p, l0))



