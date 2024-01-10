import torch
import torch.nn.functional as F

class MyModule(torch.nn.Module):
    def forward(self, x):
        x = F.relu(x)
        x = F.conv2d(x, weight=torch.randn(16, 16, 3, 3))
        x = F.max_pool2d(x, 2)
        return x

# Create the module
module = MyModule()

# Run the module
input = torch.randn(1, 16, 28, 28)
output = module(input)

print(output)