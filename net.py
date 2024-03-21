import numpy as np

class Layer:
  def __init__(self, ipt, out):
    self.ipt = ipt
    self.out = out
    self.weights = (2 * np.random.rand(ipt, out) - 1) / np.sqrt(ipt)
    # ipt to out: [i][o]
    self.biases = np.zeros(out)

  def activation(self, x):
    return 1/(1 + np.e**(-x))
  def activation_derivative(self, x):
    y = self.activation(x)
    return y*(1-y)
  def process(self, inputs):
    return self.activation(np.dot(inputs, self.weights) + self.biases)
    


class Model:
  def __init__(self, structure):
    self.layers = [Layer(structure[i], structure[i+1]) for i in range(len(structure) - 1)]

  def process(self, inputs, cost=False):
    ipts = inputs.data
    for layer in self.layers:
      ipts = layer.process(ipts)
    if cost:
      return self.Cost(ipts, inputs.expected)
    return ipts

  def Cost(self, otp, expected):
    return np.sum((otp - expected) ** 2)/len(otp)

  def train(self, inputs, learning_rate):
    # dC/dw =  dz/dw * da/dz * dc/da
    # dC/daL-1 = sum(dz/daL-1 * da/dz * dc/daL)
    d, expected = inputs.data, inputs.expected
    activations = [d]

    for layer in self.layers:
      d = layer.process(d)
      activations.append(d)

    # Last layer
    weight_grad = np.zeros(self.layers[-1].weights.shape)
    bias_grad = np.zeros(self.layers[-1].biases.shape)
    activation_derivatives = np.zeros(self.layers[-1].ipt)
    weight_grads = []
    bias_grads = []
    for k in range(self.layers[-1].ipt):
      ad = 0
      for j in range(self.layers[-1].out):
        dcdw = 2 * (activations[-1][j] - expected[j]) * self.layers[-1].activation_derivative(activations[-1][j]) * activations[-2][k]
        weight_grad[k][j] = dcdw
        if k == 0:
          dcdb = 2 * (activations[-1][j] - expected[j]) * self.layers[-1].activation_derivative(activations[-1][j])
          bias_grad[j] = dcdb
        ad += 2 * (activations[-1][j] - expected[j]) * self.layers[-1].activation_derivative(activations[-1][j]) * self.layers[-1].weights[k][j]
      activation_derivatives[k] = ad
    weight_grads.append(weight_grad)
    bias_grads.append(bias_grad)
    for i in range(len(self.layers) - 1):
      weight_grad = np.zeros(self.layers[-i-2].weights.shape)
      bias_grad = np.zeros(self.layers[-i-2].biases.shape)
      new_activation_derivatives = np.zeros(self.layers[-i-2].ipt)
      for k in range(self.layers[-i-2].ipt):
        ad=0
        for j in range(self.layers[-i-2].out):
          dcdw = activation_derivatives[j] * self.layers[-i-2].activation_derivative(activations[-i-2][j]) * activations[-i-3][k]
          weight_grad[k][j] = dcdw
          if k == 0:
            dcdb = activation_derivatives[j] * self.layers[-i-2].activation_derivative(activations[-i-2][j])
            bias_grad[j] = dcdb
          ad += self.layers[-i-2].activation_derivative(activations[-i-2][j]) * self.layers[-i-2].weights[k][j] * activation_derivatives[j]
        new_activation_derivatives[k] = ad
      
      activation_derivatives = new_activation_derivatives[:]
      weight_grads.append(weight_grad)
      bias_grads.append(bias_grad)
    for i, (wg, bg) in enumerate(zip(weight_grads, bias_grads)):
      self.layers[-i-1].weights -= learning_rate * wg
      self.layers[-i-1].biases -= learning_rate * bg

      
class Inputs:
  def __init__(self, data, expected):
    self.data = np.array(data)
    self.expected = np.array(expected)


i = Inputs([1, -0.5, 3], [0.5, 0])
m = Model([3, 2, 2])
print(m.process(i))
print(m.process(i, cost=True))
for _ in range(100):
  m.train(i, 0.1)
print(m.process(i))
print(m.process(i, cost=True))
