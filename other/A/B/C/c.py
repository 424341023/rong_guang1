# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/13 20:47
import os
import sys


def c_hello():
    print("C_HELLO")

"""4、调用上一级文件夹文件中的函数
比如我想在c.py中调用a.py中的函数，这时候如果用上面的方法就会报错No module named xx。解决方法就是将目录A加到系统路径下"""

# 打印文件绝对路径（absolute path）
print (os.path.abspath(__file__))
"""D:\my_practice_on_pycharm\My_Code_Library\other\A\B\C\c.py"""
print(os.path.abspath(''))
"""D:\my_practice_on_pycharm\My_Code_Library\other\A\B\C"""
print(os.path.abspath('..'))
"""D:\my_practice_on_pycharm\My_Code_Library\other\A\B"""
print(os.path.abspath('../..'))
"""D:\my_practice_on_pycharm\My_Code_Library\other\A"""
print(os.path.abspath('../../..'))
"""D:\my_practice_on_pycharm\My_Code_Library\other"""

# 打印文件的目录路径（文件的上一层目录），这个时候是在 C 这一层。
print (os.path.dirname( os.path.abspath(__file__) ))
"""D:\my_practice_on_pycharm\My_Code_Library\other\A\B\C"""

# 打印文件的目录路径（文件的上两层目录）, 这个时候是在 B 这一层。就是os.path.dirname这个再用了一次
print (os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
"""D:\my_practice_on_pycharm\My_Code_Library\other\A\B"""

print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
print(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__)))))
"""D:\my_practice_on_pycharm\My_Code_Library\other\A"""

print(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
print(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname( os.path.abspath(__file__))))))
"""D:\my_practice_on_pycharm\My_Code_Library\other"""

"""4、调用上一级文件夹文件中的函数
比如我想在c.py中调用b.py中的函数，解决方法就是将 B 的上一级目录 A 加到系统路径下
要调取 B 目录下的文件。 需要在 B 的上一级目录(A) 这一层才可以"""

BASE_DIR1=os.path.abspath('../..')
sys.path.append(BASE_DIR1)
from B import b
b.b_hello()

"""5、调用上两级文件夹文件中的函数
比如我想在c.py中调用a.py中的函数，解决方法就是将 A 的上一级目录 other 加到系统路径下
要调取 A 目录下的文件。 需要在 A的上一级目录(other) 这一层才可以"""

BASE_DIR2=os.path.abspath('../../..')
sys.path.append(BASE_DIR2)
from A import main
main.main_hello()
