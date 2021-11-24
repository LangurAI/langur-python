# langur-python documentation
### Table of contents
1. Neurons
   1. Activation functions
2. Networks
---
### 1. Neurons
#### 1.1. Activation functions
All functions take in an array of values as input (it is omitted in the function definitions below).
| Name        | Function           |
| ------------- |:-------------:|
| Self-defined Step Function| defined_step(threshold=0, act_alue=1, inact_alue=0) |
|Exponential Linear Unit | ELU(alfa=1) |
| Identity| identity() |
| Gaussian | gaussian() |
| Leaky Rectified Linear Unit | LeakyReLU() |
| Parametric Rectified Linear Unit | PReLU(alfa=1) |
| Rectified Linear Unit | ReLU() |
| Scaled Exponential Linear Unit | SELU(alfa=1.67326, beta=1.0507) |
| Sigmoid | sigmoid() |
| Sigmoid Linear Unit | SiLU() |
| Softplus | softplus() |
| Softsign | softsign() |
| Binary step | step() |
| Hyperbolic Tangent | tanh() |