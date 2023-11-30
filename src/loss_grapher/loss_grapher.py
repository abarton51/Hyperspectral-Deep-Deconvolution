import numpy as np
import matplotlib.pyplot as plt

model_path = 'I:\Georgia Institute of Technology\Deep Learning Project Group - General\saved_models\ClassicUnet'

train_path = model_path + '\\trainloss.txt'
val_path = model_path + '\\valloss.txt'
image_path = model_path + '\\loss_curve.png'

train_loss = np.loadtxt(train_path)
val_loss = np.loadtxt(val_path)

plt.plot(val_loss, label="validation loss")
plt.plot(train_loss, label="train loss")
plt.legend(loc='upper right')

plt.title("Train vs. Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")

plt.xlim(0, 1000)

plt.savefig(image_path)