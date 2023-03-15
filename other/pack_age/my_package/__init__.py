# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/13 22:01
# from . import my_package_one
# from . import my_package_three

print('这是父亲包 的 init')
# 在 __all__中可以声明定义允许用户可以调用的方法，以在 __all__ 中限定用户调用的范围
__all__ = ['my_package_one','my_package_three']