import math
import torch
from torch.utils.data import DataLoader
from Trainer import Trainer
import numpy as np
import matplotlib.pyplot as plt
import Utils.utils as utils


class Evaluator:
    def __init__(self, trainer: Trainer, testDataLoader: DataLoader):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.testLoader = testDataLoader
        self.trainer = trainer
        self.loss = math.inf
        self.saveDirectory = trainer.get_save_dir() + '/' + 'figs'
        self.samples = 5

    # Create pseudo RGB images by taking slices of the hyperspectral image
    def create_pseudoRGB(self, hyper):
        # this is under the assumption the indices correspond to the wavelengths:
        # 0 - 420nm
        # 14 - 560nm
        # 28 - 700nm
        RGB = np.array(hyper[(28, 14, 0), :, :].to('cpu'))
        return np.transpose(RGB, axes=(1,2,0))

    def greyscale(self, hyper):
        return np.array(hyper.mean(0).to('cpu'))

    # Return the spectral profile of a line in a hyperspectral image
    def create_spectral_profile(self, hyper):
        return None

    def PSNR(self):
        return None

    def plotter(self, mono, pred, gt):

        fig, axs = plt.subplots(nrows=self.samples, ncols=10, tight_layout=True)
        fig.set_size_inches(10,5)
        for i in range(self.samples):
            blurmono = np.array(mono[i, 0, :, :].to('cpu'))
            gtRGB = self.create_pseudoRGB(gt[i])
            predRGB = self.create_pseudoRGB(pred[i])
            grey = self.greyscale(gt[i])

            axs[i, 0].imshow(blurmono, cmap='grey')
            axs[i, 0].axis("off")

            axs[i, 1].imshow(grey, cmap='grey')
            axs[i, 1].axis("off")

            axs[i, 2].imshow(predRGB[:, :, 0], cmap=utils.red_cm())
            axs[i, 2].axis("off")

            axs[i, 3].imshow(gtRGB[:, :, 0], cmap=utils.red_cm())
            axs[i, 3].axis("off")

            axs[i, 4].imshow(predRGB[:, :, 1], cmap=utils.green_cm())
            axs[i, 4].axis("off")

            axs[i, 5].imshow(gtRGB[:, :, 1], cmap=utils.green_cm())
            axs[i, 5].axis("off")

            axs[i, 6].imshow(predRGB[:, :, 2], cmap=utils.blue_cm())
            axs[i, 6].axis("off")

            axs[i, 7].imshow(gtRGB[:, :, 2], cmap=utils.blue_cm())
            axs[i, 7].axis("off")

            axs[i, 8].imshow(predRGB)
            axs[i, 8].axis("off")

            axs[i, 9].imshow(gtRGB)
            axs[i, 9].axis("off")

        axs[0, 0].set_title("Input")
        axs[0, 1].set_title("GT Grey")
        axs[0, 2].set_title("Pred Red")
        axs[0, 3].set_title("GT Red")
        axs[0, 4].set_title("Pred Green")
        axs[0, 5].set_title("GT Green")
        axs[0, 6].set_title("Pred Blue")
        axs[0, 7].set_title("GT Blue")
        axs[0, 8].set_title("Pred pRGB")
        axs[0, 9].set_title("GT pRGB")

        plt.show()

    def test_model(self):
        model = self.trainer.get_model()
        criterion = self.trainer.get_criterion()

        model.eval()
        avgloss = torch.Tensor([0])

        # no grad for validation
        with torch.no_grad():
            for i, (inputs, gt) in enumerate(self.testLoader):
                # data to target device
                inputs = inputs.to(self.device)
                gt = gt.to(self.device)

                # validation loss
                pred = model(inputs)
                loss = criterion(pred, gt)

                if i == 0:
                    self.plotter(inputs[0:self.samples, :, :, :], pred, gt)

                # update avg validation loss
                avgloss += loss.detach().to('cpu')

            avgloss = avgloss / (i + 1)
            avgloss = np.array(avgloss)
            self.loss = avgloss[0]
            return self.loss

    def get_test_loss(self):
        return self.loss

    # Full Evaluation. Test loss, plots, spectral profiles, PSNR
    def evaluate(self):
        return None
