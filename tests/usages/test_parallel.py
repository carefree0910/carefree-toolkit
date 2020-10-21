from cftool.dist import Parallel


def test():
    def add_one(x):
        import time

        time.sleep(1)
        return x + 1

    assert Parallel(4)(add_one, list(range(10))).ordered_results == list(range(1, 11))


if __name__ == "__main__":
    test()
