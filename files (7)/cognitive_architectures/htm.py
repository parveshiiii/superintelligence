import numpy as np

class HTM:
    def __init__(self, input_size, columns, cells_per_column):
        self.input_size = input_size
        self.columns = columns
        self.cells_per_column = cells_per_column
        self.synapses = np.random.rand(columns, cells_per_column, input_size)

    def compute_overlap(self, input_vector):
        overlap = np.dot(self.synapses, input_vector)
        return overlap

    def activate_columns(self, input_vector):
        overlap = self.compute_overlap(input_vector)
        active_columns = np.argsort(overlap)[-self.columns:]
        return active_columns