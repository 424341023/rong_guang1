# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/13 22:05

import os
import sys

sys.path.append(os.path.abspath('..'))
print(os.path.abspath('..'))
"""想调用my_package，需要进入my_package的上一级目录(pack_age)"""
import my_package.my_package_one.my_moudel
import my_package.my_package_two.my_moudel
import my_package.my_package_three.my_moudel

f1 = my_package.my_package_one.my_moudel.test1
f2 = my_package.my_package_two.my_moudel.test2
f3 = my_package.my_package_three.my_moudel.test3

# 运行 my_print.py文件，结果为：

# **********
# ********************
# ******************************
