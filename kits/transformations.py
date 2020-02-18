import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

'''
in: 一个ndarray的2列(m×1)
out: 一个m×1 的 1 列
description: 2列每个元素对应相加
'''


def f_sum(column_1, column_2):
    return column_1 + column_2


'''
in: 一个ndarray的2列(m×1)
out: 一个m×1 的 1 列
description: 2列每个元素对应相减
'''


def f_subtract(column_1, column_2):
    return column_1 - column_2


'''
in: 一个ndarray的2列(m×1)
out: 一个m×1 的 1 列
description: 2列每个元素对应相减
'''


def f_multiply(column_1, column_2):
    return column_1 * column_2


'''
in:  一个ndarray的2列(m×1)
out: 一个m×1 的 1 列
description: 2列每个元素对应相减
'''


def f_divide(column_1, column_2):
    if np.all(column_2 != 0):
        return column_1 / column_2
    return []


class Binaries:
    name = ['sum', 'subtract', 'multiply', 'divide']
    func = [f_sum, f_subtract, f_multiply, f_divide]

    def __init__(self):
        pass


def f_log(column):
    if np.all(column > 0):
        return np.log2(column)
    return []


'''
in:  一个ndarray 的1列(m×1)
out: 一个m×1 的 1 列
description: 2列每个元素绝对值对应求平方根
'''


def f_square_root(column):
    return np.sqrt(np.abs(column))


'''
in:  一个ndarray 的1列(m×1)
out: 一个m×1 的 1 列
description: 2列每个元素对应求平方根,负值对绝对值求平方根加符号,例如square_root(-9)=-3
'''


def f_square(column):
    return np.sqrt(np.abs(column)) * np.sign(column)


'''
in:  一个ndarray 的1列(m×1)
out: 一个m×1 的 1 列
description: 对应元素替换成该元素在这一列出现的频次,例:[7,7,2,3,3,4] -> [2,2,1,2,2,1]
'''


def f_frequency(column):
    freq = pd.value_counts(column)
    return np.array(list(map(lambda x: freq[x], column)))


'''
in:  一个ndarray 的1列(m×1)
out: 一个m×1 的 1 列
description:每个值对应四舍五入
'''


def f_round(column):
    return np.round(column).astype('int')


'''
in:  一个ndarray 的1列(m×1)
out: 一个m×1 的 1 列
description:每个值对应双曲正切
'''


def f_tanh(column):
    return np.tanh(column)


'''
in:  一个ndarray 的1列(m×1)
out: 一个m×1 的 1 列
description:每个值对应sigmoid,自己查一下sigmoid函数
'''


def f_sigmoid(column):
    return (1 / (1 + np.exp(-column)))


'''
in:  一个ndarray 的1列(m×1),
out: 一个m×1 的 1 列
description:对该列值的分布进行,自己查一下保序回归
'''


def f_isotonic_regression(column):
    inds = range(column.shape[0])
    return IsotonicRegression().fit_transform(inds, column)


'''
in:  一个ndarray 的1列(m×1),
out: 一个m×1 的 1 列
description:对该列值的分布进行z分数,查一下z分数
'''


def f_zscore(column):
    mv, stv = np.mean(column), np.std(column)
    if stv != 0:
        return (column - mv) / stv
    return []


'''
in:  一个ndarray 的1列(m×1),
out: 一个m×1 的 1 列
description:对该列值的分布进行-1到1正则化,查一下normalization
'''


def f_normalize(column):
    maxv, minv = np.max(column), np.min(column)
    if maxv == minv:
        return []
    return -1 + 2 / (maxv - minv) * (column - minv)


class Unaries:
    name = ['log', 'square_root', 'square', 'frequency', 'round', 'tanh', 'sigmoid', 'isotonic_regression', 'zscore',
            'normalize']
    func = [f_log, f_square_root, f_square, f_frequency, f_round, f_tanh, f_sigmoid, f_isotonic_regression, f_zscore,
            f_normalize]

    def __init__(self):
        pass
