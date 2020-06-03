from cftool.dist import Parallel


def add_one(x):
    import time
    time.sleep(1)
    return x + 1

if __name__ == '__main__':
    print(Parallel(4)(add_one, list(range(10)))._rs)
