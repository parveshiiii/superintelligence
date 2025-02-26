import torch

class MemoryManagement:
    def __init__(self, memory_size: int):
        self.memory = torch.zeros(memory_size)
        self.memory_size = memory_size

    def allocate_memory(self, size: int) -> torch.Tensor:
        if size > self.memory_size:
            raise MemoryError("Requested memory size exceeds available memory")
        allocated_memory = self.memory[:size]
        self.memory = self.memory[size:]
        self.memory_size -= size
        return allocated_memory

    def compress_memory(self, data: torch.Tensor) -> torch.Tensor:
        compressed_data = torch.compress(data, data > 0)
        return compressed_data