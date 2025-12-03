import heapq
from typing import TypeVar, Generic, List, Tuple, Optional

T = TypeVar('T')

class PriorityQueue(Generic[T]):
    """
    A priority queue implementation using binary heap (heapq).
    This provides O(log n) for push and pop operations.
    """
    
    def __init__(self):
        self.elements: List[Tuple[float, int, T]] = []
        self.counter = 0  # To break ties when priorities are equal
        self.entry_finder: dict[T, Tuple[float, int, T]] = {}
    
    def push(self, item: T, priority: float) -> None:
        """Add an item to the queue with given priority."""
        if item in self.entry_finder:
            self.remove(item)
        entry = (priority, self.counter, item)
        heapq.heappush(self.elements, entry)
        self.entry_finder[item] = entry
        self.counter += 1
    
    def pop(self) -> Optional[T]:
        """Remove and return the lowest priority item."""
        while self.elements:
            priority, counter, item = heapq.heappop(self.elements)
            if item is not None and item in self.entry_finder:
                del self.entry_finder[item]
                return item
        return None
    
    def remove(self, item: T) -> None:
        """Mark an existing item as removed."""
        entry = self.entry_finder.pop(item)
        entry = (entry[0], entry[1], None)  # Mark as removed
    
    def peek(self) -> Optional[T]:
        """Return the lowest priority item without removing it."""
        while self.elements and self.elements[0][2] is None:
            heapq.heappop(self.elements)
        return self.elements[0][2] if self.elements else None
    
    def is_empty(self) -> bool:
        """Check if the queue is empty."""
        return len(self.entry_finder) == 0
    
    def __len__(self) -> int:
        """Return the number of items in the queue."""
        return len(self.entry_finder)
    
    def contains(self, item: T) -> bool:
        """Check if item is in the queue."""
        return item in self.entry_finder
    
    def get_priority(self, item: T) -> Optional[float]:
        """Get the priority of an item if it exists in the queue."""
        if item in self.entry_finder:
            return self.entry_finder[item][0]
        return None