# -*- coding: utf-8 -*- 
# @Date : 2023/2/9
# @Author : YEY
# @File : my_decorators.py

import functools
import matplotlib.pyplot as plt
from warnings import filterwarnings


def dividing_line(title=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if title:
                print('=' * 50)
                print('{}:'.format(title))
                return func(*args, **kwargs)
            else:
                print('=' * 50)
                return func(*args, **kwargs)

        return wrapper

    return decorator


def plt_support_cn(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
        return func(*args, **kwargs)

    return wrapper


def ignore_np_bool8_warning(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        filterwarnings(action='ignore', category=DeprecationWarning,
                       message='`np.bool8` is a deprecated alias for `np.bool_`.')
        return func(*args, **kwargs)

    return wrapper
