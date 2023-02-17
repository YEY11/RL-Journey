# -*- coding: utf-8 -*- 
# @Date : 2023/2/17
# @Author : YEY
# @File : my_path_utils.py

from pathlib import Path
import inspect


def get_out_file_path(filename):
    caller_path = Path(inspect.stack()[1].filename).parent
    out_path = Path(str(caller_path).replace('src', 'out'))
    Path(out_path).mkdir(parents=True, exist_ok=True)
    out_file_path = Path.joinpath(out_path, filename)
    return out_file_path
