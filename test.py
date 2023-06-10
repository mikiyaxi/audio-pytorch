
import torch 


'''
The `getattr` function in Python is used to access the attribute of an object, takes two or three arguments:

1. The object from which you are trying to get the attribute.
2. A string that names the attribute you're trying to get.
3. Optionally, a default value that will be returned if the named attribute doesn't exist.

getattr(torch, 'has_mps', False): 
is trying to get the attribute `has_mps` from the `torch` module. 
If `torch` does not have an attribute named `has_mps`, 
then `getattr` will return `False`. 

However, if `torch.has_mps` exists, 
print(getattr(torch, 'has_mps', False)) = True
'''


has_mps = getattr(torch, 'has_mps', False)

if getattr(torch, "has_mps", False):
    device = 'mps'

elif torch.cuda.is_available():
    device = 'gpu'

else:
    device = 'cpu'

print(getattr(torch, "has_mps", False))
print(device)
