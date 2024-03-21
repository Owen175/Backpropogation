import numpy as np
import time 

class Layer:
  def __init__(self, ipt, out):
    self.ipt = ipt
    self.out = out
    self.weights = (2 * np.random.rand(ipt, out) - 1) / np.sqrt(ipt)
    # ipt to out: [i][o]
    self.biases = np.zeros(out)

  def activation(self, x):
    return 1/(1 + np.exp(-x))
  def activation_derivative(self, x):
    y = self.activation(x)
    return self.activation(x)*(1-self.activation(x))
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
    
  def train(self, inputs, learning_rate=0.1, epochs=1, batch_size=1):
    for _ in range(epochs):
      for i in range(0, len(inputs), batch_size): # 0 is necessary
        batch = inputs[i:i+batch_size]
        
        [self.__internal_train(b, learning_rate) for b in batch]
    
  def __internal_train(self, inputs, learning_rate):
    # dC/dw =  dz/dw * da/dz * dc/da
    # dC/daL-1 = sum(dz/daL-1 * da/dz * dc/daL)
    activations, expected = [inputs.data], inputs.expected

    for layer in self.layers:
      activations.append(layer.process(activations[-1]))

    # Last layer
    weight_grad = np.zeros(self.layers[-1].weights.shape)
    bias_grad = np.zeros(self.layers[-1].biases.shape)
    activation_derivatives = [np.zeros(self.layers[-1].ipt)]

    for k in range(self.layers[-1].ipt):
      dcdz = 2 * (activations[-1] - expected) * self.layers[-1].activation_derivative(activations[-1])

      if k == 0:
          bias_grad = dcdz * 1
      weight_grad[k] = dcdz * activations[-2][k]

      activation_derivatives[0][k] = np.sum(dcdz * self.layers[-1].weights[k])
      
    self.layers[-1].biases -= learning_rate * bias_grad
    self.layers[-1].weights -= learning_rate * weight_grad
    
    
    for i in range(len(self.layers) - 1):
      weight_grad = np.zeros(self.layers[-i-2].weights.shape)
      bias_grad = np.zeros(self.layers[-i-2].biases.shape)
      
      if i != len(self.layers) - 2:
        activation_derivatives.append(np.zeros(self.layers[-i-2].ipt))
        
      for k in range(self.layers[-i-2].ipt):
        dadw = activation_derivatives[0] * self.layers[-i-2].activation_derivative(activations[-i-2])
        if k == 0:
          bias_grad = dadw * 1
        weight_grad[k] = dadw * activations[-i-3][k]
        if i != len(self.layers) - 2:
          activation_derivatives[-1][k] = np.sum(dadw * self.layers[-i-2].weights[k])

      activation_derivatives.pop(0)
      self.layers[-i-2].weights -= learning_rate * weight_grad
      self.layers[-i-2].biases -= learning_rate * bias_grad


class Inputs:
  def __init__(self, data, expected):
    self.data = np.array(data)
    self.expected = np.array(expected)


i = Inputs([1, -0.5, 3], [0.5, 0.5])
t = Inputs([0, -45, 11], [0.5, 0.5])
m = Model([3, 2, 2])
print(m.process(i))
print(m.process(i, cost=True))
for q in range(100):
  m.train([i, t], 0.1)
print(m.process(i))
print(m.process(t, cost=True))
