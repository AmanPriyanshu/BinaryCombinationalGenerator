from GFG import GeneticFormulaGenerator
from reader import Reader
from binary_operators import AND, OR, NOT, NOR, XOR, NAND, XNOR
import numpy as np

def main():
	data, ground_truth, features = Reader("./demo/data/train.csv").get_data()
	# data = data[-2:]
	# features = features[-2:]
	gfg = GeneticFormulaGenerator(operators=[AND, OR, NOT, NOR, XOR, NAND, XNOR], operands=[clf for clf in data], target=ground_truth, max_genome_sequence=10, lamda=0)
	sorted_pop = gfg.search(pop_size=100, gen_iters=2)
	best_pop = sorted_pop[-1]
	root = gfg.make_tree_representation(best_pop, operand_reps=features)
	gfg.visulaize_tree(root)

if __name__ == '__main__':
	main()