import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

file = "Datasets\cifar-10-batches-py\data_batch_1"

data_batch_1 = unpickle(file)
# print(data_batch_1)
print(len(data_batch_1))
print(data_batch_1.keys())
print(data_batch_1[b'data'])
print(data_batch_1[b'data'].shape)
print(data_batch_1[b'data'][0].shape)

# Reshape the image
image = data_batch_1[b'data'][0]
image = image.reshape(3,32,32)

image = image.transpose(1,2,0)

print(image.shape)

# Visualization
plt.imshow(image)  
plt.show()
