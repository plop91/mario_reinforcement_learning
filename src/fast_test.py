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


lib.NewGenome.argtypes = [ctypes.c_char_p]
lib.NewGenome.restype = ctypes.c_void_p

lib.SetName.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
lib.SetName.restype = None

lib.GetName.argtypes = [ctypes.c_void_p]
lib.GetName.restype = ctypes.c_char_p


g = lib.NewGenome("test string".encode('utf-8'))
print(lib.GetName(g).decode('utf-8'))
lib.SetName(g, "new string".encode('utf-8'))
print(lib.GetName(g).decode('utf-8'))

if os.path.exists('./libneat.so'):
    os.remove('./libneat.so')
if os.path.exists('./neat.o'):
    os.remove('./neat.o')