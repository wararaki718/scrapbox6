def func_a() -> int:
    return 1

def func_b() -> int:
    return 2

def func_c() -> int:
    a = func_a()
    b = func_b()
    return a + b
