# langur-python documentation
### Table of contents
1. Neurons
   1. Activation
   2. Weight initialization
2. Networks
---
### 1. Neurons
#### 1.i. Activation functions
All functions take in an array of values as input (it is omitted in the function definitions below).
| Name        | Function           |
| ------------- |:-------------:|
| Self-defined Step Function| DefinedStep(threshold=0, act_alue=1, inact_alue=0) |
|Exponential Linear Unit | ELU(alfa=1) |
| Identity| Identity() |
| Gaussian | Gaussian() |
| Leaky Rectified Linear Unit | LeakyReLU() |
| Parametric Rectified Linear Unit | PReLU(alfa=1) |
| Rectified Linear Unit | ReLU() |
| Scaled Exponential Linear Unit | SELU(alfa=1.67326, beta=1.0507) |
| Sigmoid | Sigmoid() |
| Sigmoid Linear Unit | SiLU() |
| Softplus | Softplus() |
| Softsign | Softsign() |
| Binary step | Step() |
| Hyperbolic Tangent | Tanh() |

#### 1.j. Weight initialization functions
All functions take in an integer of input_size (it is omitted in the function definitions below).
| Name        | Function           | Description |
| ------------- |:-------------:|:-------------:|
| He | He() | Random standard normal distribution using the He method (multiplied by the sqaure root of (2/number of input layers))  |
| Random | Random() | Random standard normal distribution |
| Random with Multiplication by Constant| AlphaRandom(alpha) | Random standard normal distribution multiplied by a constant (alpha) |
| Xavier | Xavier() | Random standard normal distribution using the Xavier method (multiplied by the sqaure root of (1/number of input layers))  |
| Zeros | Zesros() | Initialization with zeros |