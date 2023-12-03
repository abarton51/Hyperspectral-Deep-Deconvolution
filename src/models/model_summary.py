from model import ClassicUnet
from torchsummary import summary

model = ClassicUnet()
summary(model, (1,128,128))