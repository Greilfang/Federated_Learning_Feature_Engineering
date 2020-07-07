import numpy as np
import pandas as pd
import torch
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
    condition = None
    if torch.is_tensor(column_2):
        condition = torch.all(torch.ne(column_2, 0))
    else:
        condition = np.all(column_2 != 0)
    if condition:
        return column_1 / column_2
    return None


class Binaries:
    name = ['sum', 'subtract', 'multiply', 'divide']
    func = [f_sum, f_subtract, f_multiply, f_divide]

    # name = ['multiply', 'divide']
    # func = [f_multiply, f_divide]

    def __init__(self):
        pass


def f_log(column):
    if torch.is_tensor(column):
        return torch.log2(column) if torch.all(torch.gt(column, 0)) else None
    else:
        return np.log2(column) if np.all(column > 0) else None


'''
in:  一个ndarray 的1列(m×1)
out: 一个m×1 的 1 列
description: 2列每个元素绝对值对应求平方根
'''


def f_square_root(column):
    return torch.sqrt(torch.abs(column)) if torch.is_tensor(column) else np.sqrt(np.abs(column))


'''
in:  一个ndarray 的1列(m×1)
out: 一个m×1 的 1 列
description: 2列每个元素对应求平方根,负值对绝对值求平方根加符号,例如square_root(-9)=-3
'''


def f_square(column):
    return torch.sqrt(torch.abs(column)) * torch.sign(column) if torch.is_tensor(column) else np.sqrt(
        np.abs(column)) * np.sign(column)


'''
in:  一个ndarray 的1列(m×1)
out: 一个m×1 的 1 列
description: 对应元素替换成该元素在这一列出现的频次,例:[7,7,2,3,3,4] -> [2,2,1,2,2,1]
'''


def f_frequency(column):
    anchor = None
    if torch.is_tensor(column):
        anchor = column.location
        column = column.get()

    freq = pd.value_counts(np.array(column))
    freq_result = list(map(lambda x: freq[x], np.array(column)))
    return torch.tensor(freq_result).float().send(anchor) if torch.is_tensor(column) else np.array(freq_result)


'''
in:  一个ndarray 的1列(m×1)
out: 一个m×1 的 1 列
description:每个值对应四舍五入
'''


def f_round(column):
    return torch.round(column) if torch.is_tensor(column) else np.round(column).astype('int')


'''
in:  一个ndarray 的1列(m×1)
out: 一个m×1 的 1 列
description:每个值对应双曲正切
'''


def f_tanh(column):
    return torch.tanh(column) if torch.is_tensor(column) else np.tanh(column)


'''
in:  一个ndarray 的1列(m×1)
out: 一个m×1 的 1 列er
description:每个值对应sigmoid,自己查一下sigmoid函数
'''


def f_sigmoid(column):
    return torch.sigmoid(column) if torch.is_tensor(column) else (1 / (1 + np.exp(-column)))


'''
in:  一个ndarray 的1列(m×1),
out: 一个m×1 的 1 列
description:对该列值的分布进行,自己查一下保序回归
'''


def f_isotonic_regression(column):
    inds = range(column.shape[0])
    anchor = None
    if torch.is_tensor(column):
        anchor = column.location
        column = column.get()

    if torch.is_tensor(column):
        return torch.tensor(IsotonicRegression().fit_transform(inds, column)).float().send(anchor)
    else:
        return IsotonicRegression().fit_transform(inds, column)


'''
in:  一个ndarray 的1列(m×1),
out: 一个m×1 的 1 列
description:对该列值的分布进行z分数,查一下z分数
'''


def f_zscore(column):
    if torch.is_tensor(column):
        mv, stv = torch.mean(column), torch.std(column)
        condition = torch.all(torch.ne(stv, 0))
    else:
        mv, stv = np.mean(column), np.std(column)
        condition = np.all(stv != 0)
    if condition:
        return (column - mv) / stv
    return None


'''
in:  一个ndarray 的1列(m×1),
out: 一个m×1 的 1 列
description:对该列值的分布进行-1到1正则化,查一下normalization
'''


def f_normalize(column):
    if torch.is_tensor(column):
        maxv, minv = torch.max(column), torch.min(column)
        condition = torch.equal(maxv, minv)
    else:
        maxv, minv = np.max(column), np.min(column)
        condition = maxv == minv
    if condition:
        return None
    return -1 + 2 / (maxv - minv) * (column - minv)


class Unaries:
    name = ['log', 'square_root', 'square', 'frequency', 'round', 'tanh', 'sigmoid', 'isotonic_regression', 'zscore',
            'normalize']
    func = [f_log, f_square_root, f_square, f_frequency, f_round, f_tanh, f_sigmoid, f_isotonic_regression, f_zscore,
            f_normalize]

    def __init__(self):
        pass
