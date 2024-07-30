## Function概览

### 类 `Function` 的详细解释

#### 类定义和文档字符串

```python
class Function(nn.Module):
    r"""
    Class representing LTN functions.

    An LTN function is :ref:`grounded <notegrounding>` as a mathematical function (either pre-defined or learnable)
    that maps from some n-ary domain of individuals to a tensor (individual) in the Real field.
    ...
    """
```

- **类定义**：`Function` 继承自 `torch.nn.Module`，表示逻辑张量网络（LTN）中的函数。
- **文档字符串**：解释了该类的功能，参数，属性和一些注意事项。

#### 构造函数 `__init__`

```python
def __init__(self, model=None, func=None):
    ...
    super(Function, self).__init__()
    ...
    if model is not None:
        if not isinstance(model, nn.Module):
            raise TypeError("Function() : argument 'model' (position 1) must be a torch.nn.Module, not " + str(type(model)))
        self.model = model
    else:
        if not isinstance(func, types.LambdaType):
            raise TypeError("Function() : argument 'func' (position 2) must be a function, not " + str(type(model)))
        self.model = LambdaModel(func)
```

- **参数**：
  - `model`：一个 `torch.nn.Module` 对象，用于定义可训练的模型。
  - `func`：一个 Python 函数，用于定义不可训练的函数。
- **初始化**：
  - `super(Function, self).__init__()` 调用父类 `torch.nn.Module` 的构造函数。
  - 检查 `model` 和 `func` 是否同时提供或都未提供，并根据情况抛出异常。
  - 检查 `model` 是否是 `nn.Module` 类型，否则抛出 `TypeError`。
  - 检查 `func` 是否是 lambda 函数，否则抛出 `TypeError`。
  - 根据提供的参数，初始化 `self.model`。

#### 方法 `__repr__`

```python
def __repr__(self):
    return "Function(model=" + str(self.model) + ")"
```

- 返回表示该对象的字符串，主要用于调试和打印。

#### 方法 `forward`

```python
def forward(self, *inputs, **kwargs):
    ...
    if not all(isinstance(x, LTNObject) for x in inputs):
        raise TypeError("Expected parameter 'inputs' to be a tuple of LTNObject, but got " + str([type(i) for i in inputs]))

    proc_objs, output_vars, output_shape = process_ltn_objects(inputs)

    output = self.model(*[o.value for o in proc_objs], **kwargs)
    output = torch.reshape(output, tuple(output_shape + list(output.shape[1::])))
    output = output.float()

    return LTNObject(output, output_vars)
```

- **参数**：`inputs` 是一组 `LTNObject`，以及其他关键字参数 `kwargs`。
- **逻辑**：
  - 检查 `inputs` 中的每个元素是否都是 `LTNObject` 类型，否则抛出 `TypeError`。
  - 调用 `process_ltn_objects` 处理输入对象，得到处理后的对象、输出变量和输出形状。
  - 使用 `self.model` 计算函数输出，并调整输出形状。
  - 返回一个新的 `LTNObject`，其 `value` 属性包含计算结果，`free_vars` 属性包含自由变量标签。

### 关键点总结

- **继承**：`Function` 类继承自 `torch.nn.Module`。
- **初始化**：根据提供的 `model` 或 `func` 参数初始化 LTN 函数。
- **前向传播**：实现 `forward` 方法来计算函数输出，进行输入验证和处理。

这段代码实现了一个灵活的 LTN 函数类，可以使用可训练的 PyTorch 模型或简单的 Python 函数来定义函数的行为。

## 文档字符串中英对照翻译

### Class representing LTN functions.

表示LTN函数的类。

### An LTN function is :ref:`grounded <notegrounding>` as a mathematical function (either pre-defined or learnable) that maps from some n-ary domain of individuals to a tensor (individual) in the Real field.

一个LTN函数被定义为一个数学函数（可以是预定义的或可学习的），将某些n元域的个体映射到实数域中的一个张量（个体）。

### In LTNtorch, the inputs of a function are automatically broadcasted before the computation of the function, if necessary. Moreover, the output is organized in a tensor where the first :math:`k` dimensions are related with the :math:`k` variables given in input, while the last dimensions are related with the features of the individual in output. See :ref:`LTN broadcasting <broadcasting>` for more information.

在LTNtorch中，如果需要，函数的输入会在计算前自动广播。此外，输出被组织成一个张量，其中前k个维度与输入的k个变量相关，而最后的维度与输出个体的特征相关。详情见LTN广播。

### Parameters

参数

#### model : :class:`torch.nn.Module`, default=None

PyTorch模型，成为LTN函数的基础。

#### func : :obj:`function`, default=None

函数，成为LTN函数的基础。

### Attributes

属性

#### model : :class:`torch.nn.Module` or :obj:`function`

LTN函数的基础。

### Raises

抛出异常

#### :class:`TypeError`

当输入参数类型不正确时抛出。

#### :class:`ValueError`

当输入参数值不正确时抛出。

### Notes

备注

#### - the output of an LTN function is always an :ref:`LTN object <noteltnobject>` (:class:`ltn.core.LTNObject`);

LTN函数的输出总是一个LTN对象。

#### - LTNtorch allows to define a function using a trainable model **or** a python function, not both;

LTNtorch允许使用可训练的模型或Python函数来定义函数，但不能同时使用。

#### - defining an LTN function using a python function is suggested only for simple and non-learnable mathematical operations;

建议仅使用Python函数来定义简单且不可学习的数学操作。

#### - examples of LTN functions could be distance functions, regressors, etc;

LTN函数的例子可以是距离函数、回归器等。

#### - differently from LTN predicates, the output of an LTN function has no constraints;

与LTN谓词不同，LTN函数的输出没有约束。

#### - evaluating a function with one variable of :math:`n` individuals yields :math:`n` output values, where the :math:`i_{th}` output value corresponds to the function calculated with the :math:`i_{th}` individual;

评估一个具有n个个体的变量的函数会产生n个输出值，其中第i个输出值对应于用第i个个体计算的函数。

#### - evaluating a function with :math:`k` variables :math:`(x_1, \dots, x_k)` with respectively :math:`n_1, \dots, n_k` individuals each, yields a result with :math:`n_1 * \dots * n_k` values. The result is organized in a tensor where the first :math:`k` dimensions can be indexed to retrieve the outcome(s) that correspond to each variable;

评估一个具有k个变量的函数，每个变量分别具有n1, ..., nk个个体，会产生n1 * ... * nk个值的结果。结果被组织成一个张量，其中前k个维度可以索引以检索与每个变量对应的结果。

#### - the attribute `free_vars` of the `LTNobject` output by the function tells which dimension corresponds to which variable in the `value` of the `LTNObject`. See :ref:`LTN broadcasting <broadcasting>` for more information;

函数输出的LTN对象的`free_vars`属性指示LTN对象的`value`中的哪个维度对应哪个变量。

#### - to disable the :ref:`LTN broadcasting <broadcasting>`, see :func:`ltn.core.diag()`.

要禁用LTN广播，请参阅`ltn.core.diag()`函数。

### Examples

例子

#### Unary function defined using a :class:`torch.nn.Sequential`.

使用torch.nn.Sequential定义的一元函数。

```python
import ltn
import torch
function_model = torch.nn.Sequential(
                        torch.nn.Linear(4, 3),
                        torch.nn.ELU(),
                        torch.nn.Linear(3, 2)
                  )
f = ltn.Function(model=function_model)
print(f)
```

输出：

```
Function(model=Sequential(
  (0): Linear(in_features=4, out_features=3, bias=True)
  (1): ELU(alpha=1.0)
  (2): Linear(in_features=3, out_features=2, bias=True)
))
```

#### Unary function defined using a function. Note that `torch.sum` is performed on `dim=1`. This is because in LTNtorch the first dimension (`dim=0`) is related to the batch dimension, while other dimensions are related to the features of the individuals. Notice that the output of the print is `Function(model=LambdaModel())`. This indicates that the LTN function has been defined using a function, through the `func` parameter of the constructor.

使用函数定义的一元函数。注意，`torch.sum`在`dim=1`上执行。这是因为在LTNtorch中，第一个维度（`dim=0`）与批处理维度相关，而其他维度与个体的特征相关。注意打印输出`Function(model=LambdaModel())`。这表明LTN函数是通过构造函数的`func`参数使用函数定义的。

```python
f_f = ltn.Function(func=lambda x: torch.repeat_interleave(
                                             torch.sum(x, dim=1, keepdim=True), 2, dim=1)
                                        )
print(f_f)
```

输出：

```
Function(model=LambdaModel())
```

#### Binary function defined using a :class:`torch.nn.Module`. Note the call to `torch.cat` to merge the two inputs of the binary function.

使用torch.nn.Module定义的二元函数。注意调用`torch.cat`以合并二元函数的两个输入。

```python
class FunctionModel(torch.nn.Module):
    def __init__(self):
        super(FunctionModel, self).__init__()
        self.elu = torch.nn.ELU()
        self.dense1 = torch.nn.Linear(4, 5)
        self.dense2 = torch.nn.Linear(5, 2)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.elu(self.dense1(x))
        out = self.dense2(x)
        return out
```

```python
function_model = FunctionModel()
b_f = ltn.Function(model=function_model)
print(b_f)
```

输出：

```
Function(model=FunctionModel(
  (dense1): Linear(in_features=4, out_features=5, bias=True)
))
```

#### Binary function defined using a function. Note the call to `torch.cat` to merge the two inputs of the binary function.

使用函数定义的二元函数。注意调用`torch.cat`以合并二元函数的两个输入。

```python
b_f_f = ltn.Function(func=lambda x, y:
                                torch.repeat_interleave(
                                    torch.sum(torch.cat([x, y], dim=1), dim=1, keepdim=True), 2,
                                    dim=1))
print(b_f_f)
```

输出：

```
Function(model=LambdaModel())
```

#### Evaluation of a unary function on a constant. Note that:

对常量的一元函数求值。注意：

- the function returns a :class:`ltn.core.LTNObject` instance;
  函数返回一个LTNObject实例；
- since a constant has been given, the `LTNObject` in output does not have free variables;
  由于给出了一个常量，输出的LTNObject没有自由变量；
- the shape of the `LTNObject` in output is `(2)` since the function has been evaluated on a constant, namely on one single individual, and returns individuals in :math:`\mathbb{R}^2`;
  输出的LTNObject的形状是`(2)`，因为函数在一个常量上求值，即在一个单一的个体上求值，并返回实数域中的个体；
- the attribute `value` of the `LTNObject` in output contains the result of the evaluation of the function.
  输出的LTNObject的`value`属性包含函数求值的结果。

```python
c = ltn.Constant(torch.tensor([0.5, 0.01, 0.34, 0.001]))
out = f_f(c)
print(type(out))
print(out)
print(out.value)
print(out.free_vars)
print(out.shape())
```

输出：

```
<class 'ltn.core.LTNObject'>
LTNObject(value=tensor([0.8510, 0.8510]), free_vars=[])
tensor([0.8510, 0.8510])
[]
torch.Size([2])
```

#### Evaluation of a unary function on a variable. Note that:

对变量的一元函数求值。注意：

- since a variable has been given, the `LTNObject` in output has one free variable;
  由于给出了一个变量，输出的LTNObject有一个自由变量；
- the shape of the `LTNObject` in output is `(2, 2)` since the function has been evaluated on a variable with two individuals and returns individuals in :math:`\mathbb{R}^2`.
  输出的LTNObject的形状是`(2, 2)`，因为函数在具有两个个体的变量上求值，并返回实数域中的个体。

```python
v = ltn.Variable('v', torch.tensor([[0.4, 0.3],
                                    [0.32, 0.043]]))
out = f_f(v)
print(out)
print(out.value)
print(out.free_vars)
print(out.shape())
```

输出：

```
LTNObject(value=tensor([[0.7000, 0.7000],
        [0.3630, 0.3630]]), free_vars=['v'])
tensor([[0.7000, 0.7000],
        [0.3630, 0.3630]])
['v']
torch.Size([2, 2])
```

#### Evaluation of a binary function on a variable and a constant. Note that:

对变量和常量的二元函数求值。注意：

- like in the previous example, the `LTNObject` in output has just one free variable, since only one variable has been given to the predicate;
  与前一个例子一样，输出的LTNObject只有一个自由变量，因为只给谓词提供了一个变量；
- the shape of the `LTNObject` in output is `(2, 2)` since the function has been evaluated on a variable with two individuals and returns individuals in :math:`\mathbb{R}^2`. The constant does not add dimensions to the output.
  输出的LTNObject的形状是`(2, 2)`，因为函数在具有两个个体的变量上求值，并返回实数域中的个体。常量不增加输出的维度。

```python
v = ltn.Variable('v', torch.tensor([[0.4, 0.3],
                                    [0.32, 0.043]]))
c = ltn.Constant(torch.tensor([0.4, 0.04, 0.23, 0.43]))
out = b_f_f(v, c)
print(out)
print(out.value)
print(out.free_vars)
print(out.shape())
```

输出：

```
LTNObject(value=tensor([[1.8000, 1.8000],
        [1.4630, 1.4630]]), free_vars=['v'])
tensor([[1.8000, 1.8000],
        [1.4630, 1.4630]])
['v']
torch.Size([2, 2])
```

#### Evaluation of a binary function on two variables. Note that:

对两个变量的二元函数求值。注意：

- since two variables have been given, the `LTNObject` in output has two free variables;
  由于给出了两个变量，输出的LTNObject有两个自由变量；
- the shape of the `LTNObject` in output is `(2, 3, 2)` since the function has been evaluated on a variable with two individuals, a variable with three individuals, and returns individuals in :math:$\mathbb{R}^2$;
- 输出的LTNObject的形状是`(2, 3, 2)`，因为函数在具有两个个体的变量和具有三个个体的变量上求值，并返回实数域中的个体；
- the first dimension is dedicated to variable `x`, which is also the first one appearing in `free_vars`, the second dimension is dedicated to variable `y`, which is the second one appearing in `free_vars`, while the last dimensions is dedicated to the features of the individuals in output;
  第一个维度专用于变量`x`，它也是`free_vars`中第一个出现的变量，第二个维度专用于变量`y`，它是`free_vars`中第二个出现的变量，而最后一个维度专用于输出个体的特征；
- it is possible to access the `value` attribute for getting the results of the function. For example, at position `(1, 2)` there is the evaluation of the function on the second individual of `x` and third individuals of `y`.
  可以访问`value`属性以获取函数的结果。例如，在位置`(1, 2)`上，有`x`的第二个个体和`y`的第三个个体的函数评估。

```python
x = ltn.Variable('x', torch.tensor([[0.4, 0.3],
                                    [0.32, 0.043]]))
y = ltn.Variable('y', torch.tensor([[0.4, 0.04, 0.23],
                                    [0.2, 0.04, 0.32],
                                    [0.06, 0.08, 0.3]]))
out = b_f_f(x, y)
print(out)
print(out.value)
print(out.free_vars)
print(out.shape())
print(out.value[1, 2])
```

输出：

```
LTNObject(value=tensor([[[1.3700, 1.3700],
         [1.2600, 1.2600],
         [1.1400, 1.1400]],
        [[1.0330, 1.0330],
         [0.9230, 0.9230],
         [0.8030, 0.8030]]]), free_vars=['x', 'y'])
tensor([[[1.3700, 1.3700],
         [1.2600, 1.2600],
         [1.1400, 1.1400]],
        [[1.0330, 1.0330],
         [0.9230, 0.9230],
         [0.8030, 0.8030]]])
['x', 'y']
torch.Size([2, 3, 2])
tensor([0.8030, 0.8030])
```

## 使用函数定义的一元函数

### 代码解释

```python
f_f = ltn.Function(func=lambda x: torch.repeat_interleave(
    torch.sum(x, dim=1, keepdim=True), 2, dim=1)
)
print(f_f)
```

### 逐步解释

#### 创建 LTN 函数

```python
f_f = ltn.Function(func=lambda x: torch.repeat_interleave(
    torch.sum(x, dim=1, keepdim=True), 2, dim=1)
)
```

- `ltn.Function`：创建一个 LTN 函数对象。
- `func=lambda x:`：使用 lambda 表达式定义函数 `func`。
  - `x`：函数的输入，是一个张量。

#### 函数体解释

```python
torch.sum(x, dim=1, keepdim=True)
```

- `torch.sum`：计算张量 `x` 的和。
- `dim=1`：在第 1 维度上求和。
- `keepdim=True`：保留求和后的维度。

```python
torch.repeat_interleave(torch.sum(x, dim=1, keepdim=True), 2, dim=1)
```

- `torch.repeat_interleave`：重复张量的元素。
- `2`：重复两次。
- `dim=1`：在第 1 维度上重复。

#### 打印 LTN 函数对象

```python
print(f_f)
```

- `print(f_f)`：输出 LTN 函数对象的表示形式。

### 输出解释

```
Function(model=LambdaModel())
```

- 表示 `f_f` 是通过 lambda 函数创建的 `LTN` 函数。

### 例子详解

1. **定义输入张量**：

```python
import torch
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
```

2. **计算过程**：

```python
sum_x = torch.sum(x, dim=1, keepdim=True)
# sum_x = tensor([[6.0], [15.0]])

repeated_x = torch.repeat_interleave(sum_x, 2, dim=1)
# repeated_x = tensor([[6.0, 6.0], [15.0, 15.0]])
```

3. **使用 LTN 函数**：

```python
out = f_f(x)
print(out.value)
# out.value = tensor([[6.0, 6.0], [15.0, 15.0]])
```

这个例子展示了如何定义一个简单的 LTN 函数并使用它处理张量。

## 为什么输出是两个0.8510


在这个例子中，`print(out.value)` 的结果是两个数字，这是因为定义的 LTN 函数是对输入张量进行求和并重复结果的操作。让我们详细解释一下这个过程。

### 详细解释

#### 输入张量

```python
c = ltn.Constant(torch.tensor([0.5, 0.01, 0.34, 0.001]))
```

`c` 是一个常量，值为 `[0.5, 0.01, 0.34, 0.001]`。

#### 定义的 LTN 函数

```python
f_f = ltn.Function(func=lambda x: torch.repeat_interleave(
    torch.sum(x, dim=1, keepdim=True), 2, dim=1)
)
```

这段代码定义了一个 LTN 函数 `f_f`，它包含以下步骤：

1. **求和**：

   ```python
   torch.sum(x, dim=1, keepdim=True)
   ```

   - `x` 是输入张量，`dim=1` 表示在第 1 维度（列）上求和。
   - `keepdim=True` 保留求和后的维度。
2. **重复求和结果**：

   ```python
   torch.repeat_interleave(torch.sum(x, dim=1, keepdim=True), 2, dim=1)
   ```

   - `torch.repeat_interleave` 在第 1 维度上将结果重复两次。

### 计算过程

输入常量 `c` 是一维张量 `torch.tensor([0.5, 0.01, 0.34, 0.001])`。

```python
c_tensor = c.value.view(1, -1)  # 转换为二维张量以匹配函数的预期输入
sum_c = torch.sum(c_tensor, dim=1, keepdim=True)
# sum_c = tensor([[0.5 + 0.01 + 0.34 + 0.001]]) = tensor([[0.851]])
repeated_c = torch.repeat_interleave(sum_c, 2, dim=1)
# repeated_c = tensor([[0.851, 0.851]])
```

### 使用 LTN 函数计算结果

```python
out = f_f(c)
print(out.value)
# out.value = tensor([0.851, 0.851])
```

#### 总结

这个函数对输入张量在第 1 维度上求和，然后重复求和结果。这就是为什么对常量 `c` 求值的结果是两个相同的数字。函数输出是一个张量 `[0.851, 0.851]`。

## Sphinx的一些语法


在文档字符串中，`:obj:` 和 `:class:` 是用于 Sphinx 文档生成器的角色（roles），它们帮助生成链接和格式化文档。以下是它们的具体含义和用法：

### `:obj:`

- **含义**：`:obj:` 表示一个对象，通常用于引用变量、函数、类实例等。
- **用法**：用于指定一个通用对象而不特别指明它的类型。
- **示例**：
  ```python
  inputs : :obj:`tuple` of :class:`ltn.core.LTNObject`
  ```

  这里 `:obj:` 指定 `inputs` 是一个 `tuple` 对象。

### `:class:`

- **含义**：`:class:` 表示一个类，通常用于引用类定义。
- **用法**：用于明确引用某个类。
- **示例**：
  ```python
  inputs : :obj:`tuple` of :class:`ltn.core.LTNObject`
  ```

  这里 `:class:` 指定 `tuple` 中的元素是 `ltn.core.LTNObject` 类的实例。

### 结合示例详细解释

```python
inputs : :obj:`tuple` of :class:`ltn.core.LTNObject`
    Tuple of :ref:`LTN objects <noteltnobject>` for which the function has to be computed.
```

- **解释**：

  - `inputs` 参数是一个元组（`tuple`）。
  - 元组中的元素是 `ltn.core.LTNObject` 类的实例。
  - `:ref:`LTN objects <noteltnobject>`是一个参考链接，指向文档中解释`LTN objects` 的部分。
- **文档字符串生成器**：

  - Sphinx 会根据这些标记生成适当的文档链接，使得用户可以点击链接查看详细信息。

### 总结

- `:obj:` 用于引用通用对象。
- `:class:` 用于引用类定义。
- 这些标记帮助生成器创建清晰、易于导航的文档。
