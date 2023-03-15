# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/13 21:43
def d_hello():
    print("D_HELLO")

"""6、调用上一级文件夹的子文件夹下文件中的函数
比如我想在d.py中调用上一级文件夹 A 的子文件夹 B 下的 b.py 
要调取 B 目录下的文件。 需要在 B 的上一级目录(A) 这一层才可以"""

import os
import sys

print(os.path.abspath('..'))
"""D:\my_practice_on_pycharm\My_Code_Library\other\A"""

sys.path.append(os.path.abspath('..'))
from B import b
b.b_hello()
