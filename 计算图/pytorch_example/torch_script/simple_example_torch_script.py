import torch
import torch.nn as nn
import pickle

class simpleModel(nn.Module):
    def __init__(self):
        super(simpleModel,self).__init__()
        self.linear1 = nn.Linear(in_features=4,out_features=1)
        self.linear2 = nn.Linear(in_features=4,out_features=1)
        self.linear3 = nn.Linear(in_features=4,out_features=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x,y,z):
        return self.relu(self.linear1(x)+self.linear2(y)+self.linear3(z))
    
example_input_x = torch.randn(1,4)
example_input_y = torch.randn(1,4)
example_input_z = torch.randn(1,4)

sample_inputs = [example_input_x,example_input_y,example_input_z]

model = simpleModel()
model = torch.jit.script(model)

print(model.graph)

for n in model.graph.nodes():
    last_node = n

print(last_node)

print(model(example_input_x,example_input_y,example_input_z))

# save model
# path = "./simple_torch_script_model"
# torch.jit.save(model,path+".pt")

# model_loaded = torch.jit.load(path+".pt")

# load saved model
# print(model_loaded(example_input_x,example_input_y,example_input_z))

# with open(path+"_sample_input.txt","w") as file:
#     for tensor in sample_inputs:
#         shape_str = ",".join(map(str,tensor.shape))
#         file.write(shape_str+"\n")
        
