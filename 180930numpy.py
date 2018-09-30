# numpy库简单使用


from numpy import *

# 4*4的数组
a = random.rand(4, 4)
# print(a)

# 上述数组转换成矩阵
randMat = mat(a)

# 逆矩阵
invrandMat = randMat.I

# 矩阵*他的逆矩阵  结果是对角线元素为1，其余都为0
# 计算机内计算有误差
print(randMat*invrandMat)

# 误差值(矩阵*他的逆矩阵 - 4阶单位矩阵）
print(randMat*invrandMat - eye(4))

