def a() -> str:
    return 1000000 * "a"

def a1() -> str:
    return 1000000 * "a"

def b() -> str:
    return a()

def c() -> str:
    return b()

def d() -> str:
    return b()

def f() -> str:
    return g()

def e() -> str:
    return g()

def g() -> str:
    return c()

def main() -> None:
    a = a1()
    x = d()
    y = e()
    z = f()
    return (x, y, z, a)

if __name__ == "__main__":
    main()
