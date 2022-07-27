import numpy as np
from reader import Reader

def AND(a, b):
	return a*b

def NOT(a):
	return 1-a

def OR(a, b):
	return np.where(a+b>1, 1, a+b) 

def XOR(a, b):
	return np.where(a!=b, 1, 0)

def NAND(a, b):
	return np.where(a+b>1, 0, 1)

def NOR(a, b):
	return np.where(a+b==0, 1, 0)

def XNOR(a, b):
	return np.where(a!=b, 0, 1)

if __name__ == '__main__':
	data, ground_truth, features = Reader("./demo/data/train.csv").get_data()
	a, b = ground_truth, data[-1]
	a, b = a[:5], b[:5]
	print(a)
	print(b)
	print("-"*10)
	print("AND")
	print(AND(a, b))

	print("-"*10)
	print("NOT")
	print(NOT(a))

	print("-"*10)
	print("OR")
	print(OR(a, b))

	print("-"*10)
	print("XOR")
	print(XOR(a, b))

	print("-"*10)
	print("NAND")
	print(NAND(a, b))

	print("-"*10)
	print("NOR")
	print(NOR(a, b))

	print("-"*10)
	print("XNOR")
	print(XNOR(a, b))