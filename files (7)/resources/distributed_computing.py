from multiprocessing import Pool

class DistributedComputing:
    def __init__(self, num_workers):
        self.num_workers = num_workers

    def distribute_task(self, task, data):
        with Pool(self.num_workers) as pool:
            results = pool.map(task, data)
        return results