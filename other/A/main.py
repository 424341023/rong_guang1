# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/13 20:46
# main.py
def main_hello():
    print("MAIN_HELLO")

"""1、调用同文件中的函数
这个比较简单，比如我想在main.py中调用main_hello()函数，那么我直接main_hello()即可"""
a = main_hello()

"""2、调用同目录下不同文件中的函数
比如我想在main.py中调用a.py中的a_hello()函数，可做如下操作"""
import a
a.a_hello()

"""3、调用同级文件夹文件中的函数
比如我想在main.py中调用文件夹B下b.py中的b_hello()函数，可做如下操作："""
from B import b
b.b_hello()

