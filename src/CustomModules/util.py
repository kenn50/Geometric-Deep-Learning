import numpy as np

def print_values(a):
    print("The value of a is:", a)


def multiply_by_two(a):
    return a * 2

class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    def to_numpy(self):
        return np.array([self.x, self.y])
    
    @staticmethod
    def from_numpy(arr):
        return Vector(arr[0], arr[1])
    