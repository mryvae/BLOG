import torch
import torch.nn.functional as F

class MyModule(torch.nn.Module):
    def forward(self, x):
        x = F.relu(x)
        x = F.conv2d(x, weight=torch.randn(16, 16, 3, 3))
        x = F.max_pool2d(x, 2)
        return x

def modify_graph(module, input, output):
    print("Graph before modification:")
    print(module)
    print(input)
    print(output)

    # Modify the graph
    new_output = output * 2

    print("Graph after modification:")
    print(module)
    print(input)
    print(new_output)

# Create the module
module = MyModule()

# Register the hook
module.register_forward_hook(modify_graph)

# Run the module
input = torch.randn(1, 16, 28, 28)
output = module(input)
