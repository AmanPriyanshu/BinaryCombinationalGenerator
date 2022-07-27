import pandas as pd
import numpy as np

class Reader:
	def __init__(self, path):
		print("Ensure first column is ground truth!")
		data = pd.read_csv(path)
		self.features = list(data.columns)[1:]
		data = data.values
		self.labels = [i for i in np.unique(data.T[0])]
		self.ground_truth = self.get_sparsed(data.T[0])
		self.data = np.stack([self.get_sparsed(col) for col in data.T[1:]])

	def get_sparsed(self, column):
		tmp_data = np.zeros((column.shape[0], len(self.labels)))
		for i, v in enumerate(column):
			tmp_data[i][self.labels.index(v)] = 1
		return tmp_data

	def get_data(self):
		return self.data, self.ground_truth, self.features

if __name__ == '__main__':
	data, ground_truth, features = Reader("./demo/data/train.csv").get_data()
	print(data.shape, ground_truth.shape, len(features), features)
	print("ground", np.sum(ground_truth, 1))
	print("data", np.sum(data, (0, 2)))