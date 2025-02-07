# 从 ltn.core 模块中导入逻辑推理网络（LTN）所需的类和函数
from ltn.core import Variable, Predicate, Constant, Function, Connective, diag, undiag, Quantifier, \
    LTNObject, process_ltn_objects, LambdaModel # "\"的作用是换行

# 解释每个导入的类和函数的用途：
# Variable: 表示逻辑变量的类
# Predicate: 表示谓词的类，用于定义逻辑关系
# Constant: 表示常量的类
# Function: 表示函数的类，用于定义逻辑函数
# Connective: 表示逻辑连接词的类（如 AND, OR, NOT）
# diag: 用于将张量对角化的函数
# undiag: 用于将对角化张量还原的函数
# Quantifier: 表示逻辑量词的类（如 For All, Exists）
# LTNObject: 表示逻辑推理网络中的对象的类
# process_ltn_objects: 用于处理 LTN 对象的函数
# LambdaModel: 表示 Lambda 模型的类，用于定义可学习的逻辑函数和谓词
import torch  # 导入 PyTorch 库，用于张量计算
import ltn.fuzzy_ops  # 导入 ltn.fuzzy_ops 模块，用于模糊逻辑操作
# 设置设备为 GPU（如果可用），否则为 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu") # 强行使用CPU，如果使用原来默认的，会使用CUDA

# 该文件的作用是告诉setuptools如何打包这个库，以及如何安装
# 初始化模块，使ltn文件夹可以被视为一个Python包
