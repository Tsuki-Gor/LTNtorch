from ltn.core import Variable, Predicate, Constant, Function, Connective, diag, undiag, Quantifier, \
    LTNObject, process_ltn_objects, LambdaModel # "\"的作用是换行
import torch
import ltn.fuzzy_ops

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # 强行使用CPU，如果使用原来默认的，会使用CUDA

# 该文件的作用是告诉setuptools如何打包这个库，以及如何安装
# 初始化模块，使ltn文件夹可以被视为一个Python包