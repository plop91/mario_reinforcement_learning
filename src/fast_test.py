import ctypes
import subprocess
import os

if not os.path.exists('./libneat.so'):
    if os.path.exists('./Makefile'):
        subprocess.run(['make'])
    else:
        raise Exception('Makefile not found')
lib = ctypes.cdll.LoadLibrary('./libneat.so')
print(lib)


lib.NewGenome.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.NewGenome.restype = ctypes.c_void_p

lib.GenomeLoadRunTestCase.argtypes = [ctypes.c_void_p]
lib.GenomeLoadRunTestCase.restype = ctypes.c_void_p


g = lib.NewGenome(3, 3, 2)
lib.GenomeLoadRunTestCase(g)

if os.path.exists('./libneat.so'):
    os.remove('./libneat.so')
if os.path.exists('./neat.o'):
    os.remove('./neat.o')