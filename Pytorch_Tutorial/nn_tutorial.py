import torch
import torch.nn as nn

def test_Conv():
    """
    Conv2d包含卷积核参数和偏置参数
    """
    conv1 = nn.Conv2d(1, 3, 3)
    print(type(conv1))
    params = list(conv1.parameters())
    print(len(params))

    conv1.zero_grad()
    for p in params:
        print(p.shape)
        print(p.grad, p.grad_fn)
    
    input = torch.randn(1, 1, 24, 24)
    output = conv1(input)

    print("=======")
    print(output.shape)
    output.backward(torch.randn(1, 3, 22, 22))
    print(output.grad)

    for p in params:
        print(p.shape)
        print(p.grad, p.grad_fn)
    
    print("==========")
    conv1.zero_grad()
    for p in params:
        print(p.shape)
        print(p.grad, p.grad_fn)

if __name__ == "__main__":
    test_Conv()