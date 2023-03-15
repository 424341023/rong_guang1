# 程序功能:
# 编 写 者: 宗 才
# 编写时间: 2023/3/13 22:22
# 三个包均导入
import my_package.my_package_one
import my_package.my_package_two
import my_package.my_package_three

#  运行test.py文件，结果为：
# 这是父亲包 的 init
# 这是子包 one 的 init
# 这是子包 two 的 init
# 这是子包 three 的 init

"""总结：导入父模块中的子模块的时候，优先执行父模块中的 init ，再执行指定模块中的 init"""
"""也就是说，当我们去 import 一个 Package 的时候，它会隐性的去执行 __init__.py ， 
   而在 __init__.py 中定义的对象，会被绑定到当前的命名空间里面来。"""
