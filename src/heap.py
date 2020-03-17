"""
A heap implementation for no reason at all.
Yes, I do know that heapq exists.
Sometimes a man just needs to go wild and write his own heap.
"""


from typing import Tuple, Any


class Heap:
    """Maxheap with keys and objects"""

    def __init__(self, limit: int = None):
        self.heap_keys = []
        self.heap_objs = []
        self.heapsize = 0
        self.limit = limit or float('inf')

    def insert(self, el: Tuple[float, Any]):
        self.heapsize += 1
        key, obj = el
        self.heap_keys.append(key)
        self.heap_objs.append(obj)
        idx = self.heapsize - 1
        while idx > 0 and key > self.heap_keys[(idx - 1) // 2]:
            self.heap_keys[idx], self.heap_keys[(idx - 1) // 2] = (
                self.heap_keys[(idx - 1) // 2],
                self.heap_keys[idx],
            )
            self.heap_objs[idx], self.heap_objs[(idx - 1) // 2] = (
                self.heap_objs[(idx - 1) // 2],
                self.heap_objs[idx],
            )
            idx = (idx - 1) // 2
        if self.heapsize > self.limit:
            self.heap_keys.pop()
            self.heap_objs.pop()

    def heapify(self, i: int):
        if i >= self.heapsize:
            raise BaseException(
                f"Heapify {i} out of range for heap of size {self.heapsize}"
            )
        idx = i
        key = self.heap_keys[idx]
        if self.heapsize > 2 * i + 1 and self.heap_keys[2 * i + 1] > key:
            idx = 2 * i + 1
            key = self.heap_keys[idx]
        if self.heapsize > 2 * i + 2 and self.heap_keys[2 * i + 2] > key:
            idx = 2 * i + 2
            key = self.heap_keys[idx]
        if idx == i:
            return
        self.heap_keys[i], self.heap_keys[idx] = self.heap_keys[idx], self.heap_keys[i]
        self.heap_objs[i], self.heap_objs[idx] = self.heap_objs[idx], self.heap_objs[i]
        self.heapify(idx)

    def pop(self) -> Tuple[Any, Any]:
        if self.heapsize == 0:
            raise BaseException("Pop from emptyheap!")
        self.heapsize -= 1
        retkey, retobj = self.heap_keys[0], self.heap_objs[0]
        self.heap_keys[0] = self.heap_keys[-1]
        self.heap_objs[0] = self.heap_objs[-1]
        self.heap_keys.pop()
        self.heap_objs.pop()
        self.heapify(0)
        return retkey, retobj

    def peek(self) -> Tuple[Any, Any]:
        return self.heap_keys[0], self.heap_objs[0]
