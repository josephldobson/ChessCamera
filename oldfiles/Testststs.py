import math


def main():
    # 1
    a = complex(6, -2)
    b = complex(2, -3)
    # part 1
    print(math.degrees(math.atan2(b.imag, b.real)))
    print(math.degrees(math.atan2(a.imag, a.real)))
    # part 2
    unit_vector(a, b)
    # part 3 diff eq
    find_solution()


def unit_vector(a, b):
    print(a / math.sqrt(a.real**2 + a.imag**2) - b / math.sqrt(b.real**2 + b.imag**2))

def find_solution():
    x = 0
    y = 2
    while y > 1:
        x += 0.1
        y = 3 * x ** 2 + math.sin(3 * x) / 3
    print(x)

if name == 'main':
    main()