
"""
test the module build with setup.py
"""


import torch
import add_extension

a = torch.tensor([1, 2, 3], dtype=torch.float32)
b = torch.tensor([4, 5, 6], dtype=torch.float32)
c = add_extension.add(a, b)
print(c)