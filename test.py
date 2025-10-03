def test():
    a = [0,1,2,3,4,5,6]
    print('hello')
    for i in a:
        yield i

if __name__ == "__main__":
    while True:
        for i in test():
            print(i)
        break
    from collections import deque
    from itertools import islice

    q = deque()
    q.append(1)
    q.append(2)
    print(q)
    print(islice(q, 2))
    print(q)

    it = iter(q)
    print(next(it))
    it = iter(q)
    print(next(it))
    it = iter(q)
    print(next(it))
