# kNN 算法
# 样本集中每组数据（二维数组）都有分类 A，B， 计算坐标系上任意点的分组情况（A/B)
# 机器学习实战PDF 17页


from numpy import *
# 运算符模块
import operator


def createDataSet():
    """样本集 维度4， 一一对应他们的label"""
    group = array([[1.0, 1.1], [1.0, 1.0],
                   [0, 0], [0, 0.1]
                   ])

    labels = ['A', 'A', 'B', 'B']
    return group, labels


# 实例化
group, labels = createDataSet()


def classify0(inX, dataSet, labels, k):
    """
        欧拉距离公式
        4个元素依次是：目标坐标inX， 样本集dataSet，样本集中每个元素的类别labels， kNN中的k--与前k个进行对照
    """
    # dataSet的维度
    dataSetSize = dataSet.shape[0]

    # tile 把inX二维数组化，dataSetSize表示生成数组后的行数(即上述的维度)，1表示列的倍数
    # 做一个减法
    # 依次计算 目标坐标inX 与 dataSet中每一个坐标的的差值
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet

    # 上述差值的平方
    sqdiffMat = diffMat**2

    # 上述各个平方的和
    # axis=1 表示矩阵中行之间求和 =0表示列求和
    sqDistences = sqdiffMat.sum(axis=1)

    # 对上述平方和进行开方 得出距离
    distences = sqDistences**0.5

    # 对距离进行 非降序排序 值是索引值
    sortedDistIndicies = distences.argsort()

    # 空字典
    classCount = {}

    #
    for i in range(k):
        # 前k个距离最近的元素的分类
        voteLabel = labels[sortedDistIndicies[i]]
        # voteLabel对应的值为 voteLabel对应的值+1 ，如果voteLabel对应的值不存在，则为0
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1

    # 对字典进行排序 升序
    # 转换成二维数组
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    # 返回字典
    return sortedClassCount[0][0]


# 实例
thisPoint = classify0([0, 0], group, labels, 3)
print(thisPoint)



