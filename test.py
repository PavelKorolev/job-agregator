# Основные элементы Python (до middle уровня)

# 1. Переменные и базовые типы данных
# Переменные хранят данные разных типов
x = 10  # int - целое число
y = 3.14  # float - число с плавающей точкой
name = "Alice"  # str - строка
is_active = True  # bool - булево значение (True/False)
colors = ['red', 'green']  # list - список
person = {'name': 'Bob'}  # dict - словарь
unique_nums = {1, 2, 2, 3}  # set - множество
point = (10, 20)  # tuple - кортеж
data = b'hello'  # bytes - байты

# 2. Основные операторы
# Арифметические операторы
sum_value = 5 + 3  # 8
diff = 5 - 3  # 2
product = 5 * 3  # 15
quotient = 5 / 3  # 1.666...
floor_div = 5 // 3  # 1 (целочисленное деление)
modulus = 5 % 3  # 2 (остаток от деления)
power = 5 ** 3  # 125 (возведение в степень)

# Операторы сравнения
5 == 5  # True (равно)
5 != 3  # True (не равно)
5 > 3  # True (больше)
5 < 3  # False (меньше)
5 >= 5  # True (больше или равно)

# Логические операторы
True and False  # False (логическое И)
True or False  # True (логическое ИЛИ)
not True  # False (логическое НЕ)

# Бинарные операции
a = 0b1010  # 10 в двоичной
b = 0b1100  # 12 в двоичной
bin(a & b)  # '0b1000' (AND)
bin(a | b)  # '0b1110' (OR)
bin(a ^ b)  # '0b0110' (XOR)
bin(~a)  # '-0b1011' (NOT)
bin(a << 2)  # '0b101000' (сдвиг влево)
bin(a >> 1)  # '0b0101' (сдвиг вправо)

# 3. Условные выражения
# if/elif/else - условные конструкции
age = 18
if age < 18:
    print("Ребенок")
elif 18 <= age < 65:
    print("Взрослый")
else:
    print("Пенсионер")

# Тернарный оператор
status = "Взрослый" if age >= 18 else "Ребенок"


# Match-case (Python 3.10+)
def handle_command(command):
    match command.split():
        case ["go", direction]:
            print(f"Going {direction}")
        case ["take", *items]:
            print(f"Taking {', '.join(items)}")


# 4. Циклы
# for - итерация по последовательности
for i in range(3):  # 0, 1, 2
    print(i)

# while - цикл с условием
count = 0
while count < 3:
    print(count)
    count += 1

# break и continue
for num in range(5):
    if num == 3:
        break  # выход из цикла
    if num == 1:
        continue  # переход к следующей итерации
    print(num)


# 5. Функции
# def - объявление функции
def greet(name: str) -> str:  # аннотации типов
    """Документация функции"""
    return f"Hello, {name}!"


print(greet.__doc__)  # Документация функции

# Вызов функции
message = greet("Alice")


# Параметры по умолчанию
def power(x: float, n: int = 2) -> float:
    return x ** n


# Именованные аргументы
result = power(n=3, x=2)

# Walrus (морж-оператор) operator (Python 3.8+)
if (length := len("hello")) > 3:
    print(f"Length is {length}")

# Кеширование
from functools import lru_cache, cache


@lru_cache(maxsize=32)
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


@cache
def fib(n):
    return n if n < 2 else fib(n - 1) + fib(n - 2)


from functools import partial


def multiply(x, y):
    return x * y


# Фиксируем первый аргумент как 2
double = partial(multiply, 2)

print(double(4))  # Выведет 8 (2 * 4)
print(double(5))  # Выведет 10 (2 * 5)

# 6. Работа со строками
# Базовые операции
s = "Python"
len(s)  # 6 - длина строки
s[0]  # 'P' - доступ по индексу
s + " rocks!"  # Конкатенация
"Py" in s  # True - проверка вхождения

# Методы строк
"hello".upper()  # "HELLO"
"HELLO".lower()  # "hello"
"python".capitalize()  # "Python"
"  text  ".strip()  # "text" - удаление пробелов
"a,b,c".split(",")  # ['a', 'b', 'c'] - разделение

# f-строки (форматирование)
name = "Alice"
age = 25
msg = f"{name} is {age} years old"

# 7. Списки (list)
# Базовые операции
nums = [1, 2, 3]
nums.append(4)  # [1, 2, 3, 4]
nums.insert(1, 5)  # [1, 5, 2, 3, 4]
nums.remove(2)  # [1, 5, 3, 4]
nums.pop()  # возвращает 4, nums = [1, 5, 3]

# Срезы (slicing)
letters = ['a', 'b', 'c', 'd']
letters[1:3]  # ['b', 'c'] (с 1 по 3 не включительно)
letters[:2]  # ['a', 'b'] (с начала до 2)
letters[1:]  # ['b', 'c', 'd'] (с 1 до конца)
letters[::2]  # ['a', 'c'] (каждый второй)

# List comprehension
squares = [x ** 2 for x in range(5)]  # [0, 1, 4, 9, 16]


# Оптимизация памяти с помощью __slots__
class OptimizedList:
    __slots__ = ['items']  # Запрещает создание __dict__

    def __init__(self, items):
        self.items = items


# 8. Кортежи (tuple)
# Неизменяемые последовательности
point = (10, 20)
x, y = point  # распаковка

# 9. Словари (dict)
# Хранение данных в виде ключ-значение
person = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}

# Работа со словарями
person["name"]  # "Alice" - доступ по ключу
person.get("age", 0)  # 25 (0 - значение по умолчанию)
person["email"] = "a@example.com"  # добавление
"age" in person  # True - проверка ключа

# Итерация по словарю
for key, value in person.items():
    print(f"{key}: {value}")

from collections import defaultdict

# Словарь со значением по умолчанию
d = defaultdict(list)
d["key"].append(1)

# 10. Множества (set)
# Уникальные неупорядоченные элементы
unique_nums = {1, 2, 2, 3}  # {1, 2, 3}

# Операции с множествами
a = {1, 2, 3}
b = {3, 4, 5}
a.union(b)  # {1, 2, 3, 4, 5} (объединение)
a.intersection(b)  # {3} (пересечение)
a.difference(b)  # {1, 2} (разность)

# frozenset - неизменяемое множество
fs = frozenset([1, 2, 3])
# fs.add(4)  # AttributeError - нельзя изменить

# Хешируемость и функция hash()
# Хешируемые объекты могут быть ключами словаря
hash("hello")  # возвращает хеш-значение
hash((1, 2))  # кортеж хешируем


# hash([1, 2]) # TypeError: unhashable type: 'list'

# Для объектов хеш определяется методом __hash__
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __hash__(self):
        return hash((self.x, self.y))


p = Point(1, 2)
hash(p)  # работает благодаря __hash__

# 11. Файловый ввод/вывод
# Чтение файла
with open("file.txt", "r") as file:
    content = file.read()

# Запись в файл
with open("output.txt", "w") as file:
    file.write("Hello, World!")

# Работа с временными файлами
from tempfile import TemporaryFile

with TemporaryFile('w+') as tmp:
    tmp.write('test')
    tmp.seek(0)
    print(tmp.read())

# 12. Исключения (try/except)
# Обработка ошибок
try:
    result = 10 / 0
except ZeroDivisionError:
    print("Нельзя делить на ноль!")
except Exception as e:
    print(f"Произошла ошибка: {e}")
else:
    print("Ошибок не было")
finally:
    print("Это выполнится всегда")

# 13. Работа с датами и временем (datetime)
from datetime import datetime, timedelta

now = datetime.now()  # текущая дата и время
today = datetime.today()  # текущая дата
future = now + timedelta(days=7)  # дата через 7 дней

# Форматирование даты
formatted = now.strftime("%Y-%m-%d %H:%M:%S")

# 14. Модули и импорты
# Импорт всего модуля
import math

math.sqrt(16)  # 4.0

# Импорт конкретных функций
from math import cos  # noqa

# Импорт с псевдонимом
from math import cos as Cos  # noqa


# 15. Классы и ООП
class Person:
    # Статическая переменная (общая для всех экземпляров)
    species = "Homo sapiens"

    def __init__(self, name, age):
        self.name = name
        self.age = age

    # Метод экземпляра
    def greet(self):
        return f"Hi, I'm {self.name}"

    # Метод класса (работает с классом, а не экземпляром)
    @classmethod
    def from_birth_year(cls, name, birth_year):
        age = datetime.now().year - birth_year
        return cls(name, age)

    # Статический метод (не имеет доступа ни к классу, ни к экземпляру)
    @staticmethod
    def is_adult(age):
        return age >= 18

    # Декоратор property для геттеров/сеттеров
    @property
    def age(self):
        return self._age

    @age.setter
    def age(self, value):
        if value < 0:
            raise ValueError("Age cannot be negative")
        self._age = value


# Абстрактный базовый класс (ABC)
from abc import ABC, abstractmethod


class Animal(ABC):
    @abstractmethod
    def make_sound(self):
        pass


class Dog(Animal):
    def make_sound(self):
        return "Woof!"


from dataclasses import dataclass


@dataclass
class Point:
    x: float
    y: float


from enum import Enum, auto, unique


class Color(Enum):
    RED = 1
    GREEN = 2
    BLUE = 3
    BLACK = 1 # ValueError: Ошибка из-за дублирования значения


print(Color.RED)        # Color.RED
print(Color.RED.value)  # 1
print(Color(1))         # Color.RED (получение по значению)
print(Color['RED'])     # Color.RED (получение по имени)


class Status(Enum):
    PENDING = auto()  # 1
    RUNNING = auto()  # 2
    DONE = auto()     # 3

# 16. Генераторы
# Функция-генератор
def countdown(n):
    while n > 0:
        yield n
        n -= 1


# Использование генератора
for num in countdown(5):
    print(num)  # 5, 4, 3, 2, 1

# 17. Декораторы
from functools import wraps


def logger(func):
    @wraps(func)  # сохраняет метаданные оригинальной функции
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)

    return wrapper


@logger
def add(a, b):
    """Складывает два числа"""
    return a + b


# Без @wraps add.__name__ был бы "wrapper", а docstring терялся
print(add.__name__)  # "add"
print(add.__doc__)  # "Складывает два числа"


# Декоратор с параметрами
def repeat(num_times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(num_times):
                result = func(*args, **kwargs)
            return result

        return wrapper

    return decorator


@repeat(num_times=3)
def greet(name):
    print(f"Hello {name}")


# Декоратор класса
def add_greeting(decorated_class):
    original_init = decorated_class.__init__

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)  # Вызываем оригинальный __init__
        self.greeting = "Hello!"  # Добавляем новый атрибут

    decorated_class.__init__ = new_init  # Заменяем __init__
    return decorated_class


# Применяем декоратор к классу
@add_greeting
class Person:
    def __init__(self, name):
        self.name = name


# Создаем объект
p = Person("Alice")
print(p.name)  # Выведет: Alice
print(p.greeting)  # Выведет: Hello!

# 18. Lambda-функции
# Анонимные функции
square = lambda x: x ** 2
square(5)  # 25

# Использование с map/filter
nums = [1, 2, 3]
squared = list(map(lambda x: x ** 2, nums))  # [1, 4, 9]
evens = list(filter(lambda x: x % 2 == 0, nums))  # [2]

# 19. Работа с файловой системой (os, pathlib)
import os
from pathlib import Path

# os модуль
os.listdir()  # список файлов в директории
os.path.exists("file.txt")  # проверка существования

# pathlib (современный подход)
path = Path("file.txt")
path.exists()  # проверка существования
path.read_text()  # чтение файла

# 20. Основные встроенные функции
abs(-5)  # 5 (модуль числа)
len("hello")  # 5 (длина последовательности)
max([1, 2, 3])  # 3 (максимальное значение)
min([1, 2, 3])  # 1 (минимальное значение)
sum([1, 2, 3])  # 6 (сумма)
round(3.14159, 2)  # 3.14 (округление)
sorted([3, 1, 2])  # [1, 2, 3] (сортировка)
zip([1, 2], ['a', 'b'])  # итератор (1,'a'), (2,'b')

# 21. Работа с коллекциями (itertools, collections)
from itertools import (chain, repeat, zip_longest,
                       islice, takewhile, dropwhile,
                       product, permutations, combinations_with_replacement, groupby)

# chain - объединяет итераторы
list(chain([1, 2], [3, 4]))  # [1, 2, 3, 4]

# cycle - бесконечный цикл по элементам
# for i in cycle([1, 2]): print(i)  # 1, 2, 1, 2, ...

# repeat - повторение элемента
list(repeat(5, 3))  # [5, 5, 5]

# zip_longest - zip с заполнением недостающих
list(zip_longest([1, 2], [3, 4, 5], fillvalue=0))  # [(1,3), (2,4), (0,5)]

# islice - срез для итераторов
list(islice(range(10), 2, 8, 2))  # [2, 4, 6]

# takewhile/dropwhile - фильтрация
list(takewhile(lambda x: x < 5, [1, 4, 6, 4]))  # [1, 4]
list(dropwhile(lambda x: x < 5, [1, 4, 6, 4]))  # [6, 4]

# product - декартово произведение
list(product([1, 2], ['a', 'b']))  # [(1,'a'), (1,'b'), (2,'a'), (2,'b')]

# combinations_with_replacement - комбинации с повторениями
list(combinations_with_replacement([1, 2], 2))  # [(1,1), (1,2), (2,2)]

# groupby - группировка по ключу
data = sorted([('a', 1), ('b', 2), ('a', 3)], key=lambda x: x[0])
for key, group in groupby(data, key=lambda x: x[0]):
    print(key, list(group))  # a [('a',1), ('a',3)]  b [('b',2)]

from collections import Counter

# Counter - подсчет элементов
count = Counter("hello")  # {'h':1, 'e':1, 'l':2, 'o':1}

# permutations - все перестановки
list(permutations([1, 2, 3], 2))  # [(1,2), (1,3), (2,1), ...]

from collections import namedtuple, deque

Point = namedtuple('Point', ['x', 'y'])
p = Point(1, 2)

queue = deque()
queue.append(1)
queue.popleft()

# 22. Контекстные менеджеры (with)
# Вариант 1: как функция с @contextmanager
# Собственный контекстный менеджер
from contextlib import contextmanager


@contextmanager
def temp_file():
    print("Создание временного файла")
    yield "temp.txt"
    print("Удаление временного файла")


with temp_file() as f:
    print(f"Работа с файлом {f}")


# Вариант 2: как класс с __enter__ и __exit__
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None

    def __enter__(self):
        self.file = open(self.filename, self.mode)
        return self.file

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file:
            self.file.close()
        if exc_type:
            print(f"Exception occurred: {exc_val}")
        return False  # False - пробрасывает исключение, True - подавляет


# Использование
with FileManager("test.txt", "w") as f:
    f.write("Hello, context manager!")


# 23. Аннотации типов
# Указание типов для переменных и функций
def add(a: int, b: int) -> int:
    return a + b


# 24. Основы работы с JSON
import json

# Преобразование в JSON
data = {"name": "Alice", "age": 25}
json_str = json.dumps(data)  # строка JSON

# Чтение из JSON
loaded_data = json.loads(json_str)  # обратно в dict

# 25. Основы тестирования с pytest
"""
Установка pytest:
pip install pytest

Запуск тестов:
pytest filename.py -v  # -v для подробного вывода
"""


# Простейший тест
def test_addition():
    assert 1 + 1 == 2


# Тест с ожидаемым исключением
import pytest


def test_divide_by_zero():
    with pytest.raises(ZeroDivisionError):
        1 / 0


# Фикстуры (setup/teardown)
@pytest.fixture
def sample_data():
    # Подготовка данных перед тестом
    data = [1, 2, 3, 4, 5]
    yield data  # это то, что будет передано в тест
    # Очистка после теста (необязательно)
    print("Тест завершен")


def test_sum(sample_data):
    assert sum(sample_data) == 15


# Параметризованные тесты
@pytest.mark.parametrize("input,expected", [
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square(input, expected):
    assert input ** 2 == expected


# Маркировка тестов
@pytest.mark.slow
def test_large_computation():
    # Этот тест будет пропущен при обычном запуске
    # Запуск только медленных тестов: pytest -m slow
    assert 2 ** 100 > 1_000_000


# Пропуск теста
@pytest.mark.skip(reason="Еще не реализовано")
def test_future_feature():
    assert False


# Мокирование (требуется pytest-mock)
def test_with_mock(mocker):
    mock_requests = mocker.patch('requests.get')
    mock_requests.return_value.status_code = 200
    from mymodule import check_url
    assert check_url("http://test.com") == 200


# Тестирование классов
class TestMathOperations:
    def test_add(self):
        assert 1 + 1 == 2

    def test_multiply(self):
        assert 2 * 3 == 6


# Проверка предупреждений
def test_warning():
    with pytest.warns(UserWarning):
        import warnings
        warnings.warn("Это предупреждение", UserWarning)


# Плагины pytest:
# - pytest-cov: проверка покрытия кода тестами
# - pytest-xdist: параллельное выполнение тестов
# - pytest-django: поддержка Django
# - pytest-asyncio: тестирование asyncio кода

# 27. Регулярные выражения
import re

emails = re.findall(r'\b[\w.-]+@[\w.-]+\.\w+\b', "test@example.com")

# 28. Логирование
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.info("Информационное сообщение")  # INFO:__main__:Информационное сообщение

# 28. Работа с окружением
from dotenv import load_dotenv
import os

load_dotenv()
db_url = os.getenv("DB_URL")

# 29. Асинхронность (asyncio)
import asyncio


async def fetch_data():
    await asyncio.sleep(1)
    return "Данные"


async def main():
    data = await fetch_data()
    print(data)


asyncio.run(main())
