# 程序功能:
# Python 父类的 私有属性 和 私有方法
# 编 写 者: 宗 才
# 编写时间: 2023/3/8 9:05

''''''

'''
(1)__方法名__(self)  python提供的内置类方法

        如__str__ 和 __repr__ 两者的目的都是为了显式的显示对象的一些必要信息，方便查看和调试。
         __str__被print默认调用，__repr__被控制台输出时默认调用。
         即，使用__str__控制用户展示，使用__repr__控制调试展示。
         
   self.__属性名__   python提供的内置类属性
        如__name__  当前定义的[类]的名字
         __dict__ [类或对象]的属性（包含一个字典，由类的属性(包括公有属性和私有属性)组成）

(2) 方法名(self)  类公有方法
   self.属性名    类公有属性

() _方法名(self)  类私有方法
   self._属性名   类私有属性    约定俗成的一种提示,强行访问仍旧可以正常访问
(3) __方法名(self)  类私有方法
   self.__属性名    类私有属性  强行访问是无法访问的,可通过间接方式进行访问( _类名__变量名/ _类名__方法名())
   
    1、子类对象 不能够在自己的方法内部，直接 访问 父类的 私有属性 或 私有方法；
    2、子类对象 可以通过 父类 的公有方法 间接 访问到 私有属性 或 私有方法；
       私有属性、方法 是对象的隐私，不对外公开，外界 以及 子类 都不能直接访问；
       私有属性、方法 通常用于做一些内部的事情；
'''

class A(object):

    def __init__(self):
        self.num1 = 100

        # 在属性名前增加两个下划线定义的就是私有属性
        self.__num2 = 200

    # 在方法名前增加两个下划线定义的就是私有方法
    def __test(self):
        print("私有方法 %d %d " % (self.num1,self.__num2))


class B(A):

    def demo(self):
        # 在子类的对象方法中，不能访问父类的私有属性
        # AttributeError: 'B' object has no attribute '_B__num2'
        # print("访问父类的私有属性 %d" % self.__num2)

        # 不能直接访问父类的私有方法
        # self.__test()
        # 可以间接地访问父类的私有方法
        self._A__test()
        pass

# 创建一个子类对象
b = B()
print(b)
# AttributeError: 'B' object has no attribute '_B__test'
b.demo()

a = A()
# 外部无法直接访问类的私有属性
# AttributeError: 'A' object has no attribute '__num2'
# print(a.__num2)
# 外部无法直接访问类的私有方法
# AttributeError: 'A' object has no attribute '__test'
# a.__test()

# 外部可通过间接方式进行访问
print(a._A__num2)
a._A__test()

print(a.__str__())
