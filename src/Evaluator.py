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
        self.samples = 4
        self.PSNR = 0

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
    def create_spectral_profile(self, pred, gt):
        return None

    def plotter(self, mono, pred, gt, blur, idx, filename, plot_all=False):
        
        n = self.samples
        if plot_all:
            n = mono.shape[0]
        
        for i in range(n):
            fig, axs = plt.subplots(nrows=3, ncols=5, tight_layout=True)
            fig.set_size_inches(5,3)

            blurmono = np.array(mono[i, 0, :, :].to('cpu'))
            gtRGB = self.create_pseudoRGB(gt[i])
            predRGB = self.create_pseudoRGB(pred[i])
            blurRGB = self.create_pseudoRGB(blur[i])
            grey = self.greyscale(gt[i])
            idxx = idx[i].item()

            axs[2, 0].imshow(blurmono, cmap='gray')
            axs[2, 0].axis("off")
            axs[1, 0].imshow(blurmono, cmap='gray')
            axs[1, 0].axis("off")
            axs[0, 0].imshow(grey, cmap='gray')
            axs[0, 0].axis("off")

            axs[2, 1].imshow(predRGB[:, :, 0], cmap=utils.red_cm())
            axs[2, 1].axis("off")
            axs[1, 1].imshow(blurRGB[:, :, 0], cmap=utils.red_cm())
            axs[1, 1].axis("off")
            axs[0, 1].imshow(gtRGB[:, :, 0], cmap=utils.red_cm())
            axs[0, 1].axis("off")

            axs[2, 2].imshow(predRGB[:, :, 1], cmap=utils.green_cm())
            axs[2, 2].axis("off")
            axs[1, 2].imshow(blurRGB[:, :, 1], cmap=utils.green_cm())
            axs[1, 2].axis("off")
            axs[0, 2].imshow(gtRGB[:, :, 1], cmap=utils.green_cm())
            axs[0, 2].axis("off")

            axs[2, 3].imshow(predRGB[:, :, 2], cmap=utils.blue_cm())
            axs[2, 3].axis("off")
            axs[1, 3].imshow(blurRGB[:, :, 2], cmap=utils.blue_cm())
            axs[1, 3].axis("off")
            axs[0, 3].imshow(gtRGB[:, :, 2], cmap=utils.blue_cm())
            axs[0, 3].axis("off")

            axs[2, 4].imshow(predRGB)
            axs[2, 4].axis("off")
            axs[1, 4].imshow(blurRGB)
            axs[1, 4].axis("off")
            axs[0, 4].imshow(gtRGB)
            axs[0, 4].axis("off")

            plt.savefig(self.saveDirectory + '/' + str(idxx) + '_' + filename + str(i) + '.png', bbox_inches='tight')

    def test_model(self, filename=None):
        model = self.trainer.get_model()
        criterion = self.trainer.get_criterion()

        model.eval()
        avgloss = torch.Tensor([0])
        avgpsnr = torch.Tensor([0])
        idx_history = torch.Tensor([])

        # no grad for validation
        with torch.no_grad():
            for i, (inputs, gt, blur, idx) in enumerate(self.testLoader):
                # data to target device
                inputs = inputs.to(self.device)
                gt = gt.to(self.device)
                
                # track idx from OG dataset
                #idx_history = torch.cat([idx_history, idx])
                interesting_idx = idx < 100
                
                # validation loss
                pred = model(inputs)
                loss = criterion(pred, gt)
                psnr = utils.PSNR(pred, gt)
                
                # if there are any test samples from the batch whose OG idx is < 100, plot them
                if interesting_idx.sum().item() > 0:
                    self.plotter(inputs[interesting_idx, :, :, :],
                                pred[interesting_idx, :, :, :],
                                gt[interesting_idx, :, :, :],
                                blur[interesting_idx, :, :, :],
                                idx[interesting_idx],
                                filename,
                                plot_all=True)

                if i == 0 and filename is not None:
                    self.plotter(inputs[0:self.samples, :, :, :],
                                 pred[0:self.samples, :, :, :],
                                 gt[0:self.samples, :, :, :],
                                 blur[0:self.samples, :, :, :],
                                 idx[:self.samples],
                                 filename)

                # update avg test loss
                avgloss += loss.detach().to('cpu')
                avgpsnr += psnr
            
            avgloss = avgloss / (i + 1)
            avgloss = np.array(avgloss)
            avgpsnr = avgpsnr / (i + 1)
            avgpsnr = np.array(avgpsnr)
            self.loss = avgloss[0]
            self.psnr = avgpsnr[0]
            return self.loss, self.psnr

    def get_test_loss(self):
        return self.loss