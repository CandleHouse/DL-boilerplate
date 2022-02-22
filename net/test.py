import logging
import os
import tifffile
from models import xxNet
import torch
import numpy as np

input_dir = ''
pred_dir = ''
# dir_checkpoint = '../checkpoints/epochs/'  # every epoch
# checkpoint_name = 'epoch_77.pth'

final_checkpoint = '../checkpoints/'  # the last epoch
checkpoint_name = f'CP_BS_{32}_LR_{1e-3}_EPOCH_{80}_0222.pth'


def get_file_list(folder, extension):
    file_list = []
    for dir_path, dir_names, file_names in os.walk(folder):
        for file in file_names:
            file_type = os.path.splitext(file)[1]
            if file_type == extension:
                file_fullname = os.path.join(dir_path, file)
                file_list.append(file_fullname)
    return file_list


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = xxNet()
    model.load_state_dict(torch.load(final_checkpoint + checkpoint_name))
    model.to(device)

    images_files_list = np.sort(get_file_list(input_dir, '.npy'))
    samples = len(images_files_list)

    imgs = np.zeros((samples, 1, 237, 324), dtype=np.float32)
    for i in range(len(images_files_list)):
        imgs[i, :, :, :] = np.load(images_files_list[i])

    preds = np.zeros((samples, 1, 237, 324), dtype=np.float32)

    batch_size = 8
    batchs = int(np.floor(samples / batch_size))
    left_sample = samples - (batch_size * batchs)

    for i in range(batchs):
        batch_imgs = imgs[i * batch_size:i * batch_size + batch_size, :, :, :]
        batch_imgs = torch.from_numpy(batch_imgs)
        batch_imgs = batch_imgs.to(device=device, dtype=torch.float32)
        pred = model(batch_imgs)
        preds[i * batch_size:i * batch_size + batch_size, :, :, :] = pred.cpu().detach().numpy()

    if left_sample != 0:
        last_img = imgs[batchs * batch_size:batchs * batch_size + left_sample, :, :, :]
        last_img = torch.from_numpy(last_img)
        last_img = last_img.to(device=device, dtype=torch.float32)
        pred = model(last_img)
        preds[batchs * batch_size:batchs * batch_size + left_sample, :, :, :] = pred.cpu().detach().numpy()

    for i in range(len(images_files_list)):
        pred_filename = images_files_list[i].split("/")[-1].split(".raw")[0].split("_80kVp_")[0] + '_' + str(i).rjust(4, '0')

        # use .tif format for better use
        tifffile.imwrite(os.path.join(pred_dir, pred_filename) + '.tif', np.squeeze(preds[i, :, :, :]))