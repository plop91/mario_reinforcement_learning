from ctypes import cdll
import subprocess
import os

if not os.path.exists('./libneat.so'):
    if os.path.exists('./Makefile'):
        subprocess.run(['make'])
    else:
        raise Exception('Makefile not found')
lib = cdll.LoadLibrary('./libneat.so')
print(lib)


class Foo(object):

    def __init__(self):
        self.node = lib.NewNode(0,10)
        print(self.node)
        self.genome = lib.NewGenome(3,2,3)
        print(self.genome)

    def test(self):
        print(lib.NodeGetId(self.node))
        print(lib.NodeGetValue(self.node))
        # print(self.genome)


f = Foo()
f.test()
