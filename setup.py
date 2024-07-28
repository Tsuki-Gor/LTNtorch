from setuptools import setup, find_packages # 引入打包工具，这里用的是setuptools
# setuptools是一个集成了easy_install的功能，可以简化安装第三方库的过程，也可以打包分发自己的库
# find_packages是一个setuptools的函数，会自动发现有__init__.py的文件夹，可以递归查找，找到所有包
# setup是一个配置函数，用来描述库的名字，版本，作者，简介等信息
# 详细的参数可以参考官方文档：https://setuptools.pypa.io/en/latest/setuptools.html
# 该文件的作用是告诉setuptools如何打包这个库，以及如何安装

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='LTNtorch',
    version='1.0.1',
    packages=find_packages(include=['ltn']),
    install_requires=[
        "numpy",
        "torch"
    ],
    python_requires='>=3.7',
    url='https://github.com/bmxitalia/LTNtorch',
    download_url='https://github.com/bmxitalia/LTNtorch',
    license='MIT',
    author='Tommaso Carraro',
    author_email='tommaso.carraro@studenti.unipd.it',
    description='LTNtorch: PyTorch implementation of Logic Tensor Networks',
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords=['pytorch', 'machine-learning', 'framework', 'neural-symbolic-computing', 'fuzzy-logic'],
    classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Programming Language :: Python :: 3.7',
            'Programming Language :: Python :: 3.8',
            'Programming Language :: Python :: 3.9',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'License :: OSI Approved :: MIT License',
        ]
)
