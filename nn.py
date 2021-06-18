import numpy as np

#Sigmoid function
def sig(x):
    return 1 / (1 + np.exp(-x)) 

#Read in training data
num_pixels = 784
pixel_cols = [i for i in range(1, num_pixels + 1)]
data = np.genfromtxt("./data/mnist_train_0_1.csv", delimiter=",", usecols=pixel_cols)
labels = np.genfromtxt("./data/mnist_train_0_1.csv", delimiter=",", usecols=0)

#Set initial parameters
hidden_nodes = 30
alpha = 0.2
weights1 = np.random.rand(hidden_nodes, num_pixels) / 100
weights2 = np.random.rand(hidden_nodes) / 100

#Train
for example in range(len(data)):
    #Feed forward
    layer1 = data[example] / 255
    layer2 = sig(np.matmul(weights1, layer1))
    out = sig(np.matmul(weights2, layer2))
    
    #Back propagation
    delta2 = (labels[example] - out) * out * (1 - out)
    delta1 = layer2 * (1 - layer2) * weights2 * delta2
    weights2 += alpha * layer2 * delta2
    weights1 += alpha * np.tile(layer1, (hidden_nodes, 1)) * np.tile(delta1, (num_pixels, 1)).transpose()
    
#Test
test_data = np.genfromtxt("./data/mnist_test_0_1.csv", delimiter=",", usecols=pixel_cols)
test_labels = np.genfromtxt("./data/mnist_test_0_1.csv", delimiter=",", usecols=0)
correct = 0
for example in range(len(test_data)):
    #Feed forward
    layer1 = test_data[example]
    layer2 = sig(np.matmul(weights1, layer1))
    out = sig(np.matmul(weights2, layer2))

    #Count how many predictions were correct
    correct += round(out) == test_labels[example]
correct = correct / len(test_data) * 100

#Print results
print(str(correct) + "% Correct")
