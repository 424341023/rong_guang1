# 程序功能:
# Python:父类，子类继承关系详解。（如何在子类中增加新的属性？）
# python中子类对父类方法的继承和重写
# 编 写 者: 宗 才
# 编写时间: 2023/3/8 8:30


class People(object):
# 第6行 最左边的小写o表示 父类People的子类是Teacher

    def __init__(self, name, city):
# 第9行 最左边的小写o表示 父类People中的__init__()初始化方法在子类Teacher中被重写(override)
        self.name = name
        self.city = city

    def moveto(self, newcity):
# 第14行 最左边的小写o表示 父类People中的moveto()方法在子类Teacher中被重写
        self.city = newcity

    def __lt__(self, other):
# 第18行 最左边的小写o表示 父类People中的__lt__()方法在子类Teacher中被重写
        return self.city < other.city

    def __str__(self):
# 第22行 最左边的第二个小写o表示 父类object中的__str__()方法在子类People中被重写
#       最左边的第一个小写o表示 父类Peoole中的__str__()方法在子类Teacher中被重写
        return '(%s, %s)' % (self.name, self.city)

    def hobby(self, hobby):
        print("My hobby is: {}".format(hobby))

    __repr__ = __str__


class Teacher(People):

    # 对父类方法进行扩展
    def __init__(self, name, city, school):
        super(Teacher, self).__init__(name, city)  # 保留父类初始化方法
        self.school = school                       # 增加新的属性

    # 覆盖父类的方法
    def moveto(self, newschool):
        self.school = newschool
        # 在方法内部，可以通过self. 访问对象的属性
        print(self.school)
        # 在方法内部，也可以通过self. 访问其它的对象方法
        # 这里访问的是继承自父类People中的 hobby()方法
        print(self.hobby("swimming"))

    # 覆盖父类的方法
    def __lt__(self, other):
        return self.school < other.school

    # 覆盖父类的方法
    def __str__(self):
        return '(%s, %s, %s)' % (self.name, self.city, self.school)

    __repr__ = __str__

# 实例化对象，传入的参数须包括__init__()初始化方法中的所有参数
s1 = Teacher('Jane', 'Beijing', 'SCU')
# 子类Teacher继承了父类的hobby()方法，调用该方法时，需传入一个hobby参数
print(s1.hobby('running'))