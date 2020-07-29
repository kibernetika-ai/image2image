"""Main"""
import argparse
import logging
import sys
import time

import matplotlib
from skimage import metrics
import tensorboardX
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset.dataset_class import PreprocessDataset
from dataset.dataset_class import DatasetRepeater
from dataset.video_extraction_conversion import *
from loss.loss_discriminator import *
from loss.loss_generator import *
from network.model import *
from network.resblocks import *


LOG = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', default=1, type=int)
    parser.add_argument('--epochs', default=10, type=int)
    parser.add_argument('--save-checkpoint', type=int, default=1000)
    parser.add_argument('--train-dir', default='train')
    parser.add_argument('--vggface-dir', default='.')
    parser.add_argument('--data-dir')
    parser.add_argument('--frame-shape', default=256, type=int)
    parser.add_argument('--workers', default=4, type=int)

    return parser.parse_args()


def main():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-5s %(name)-10s [-] %(message)s',
        level='INFO'
    )
    logging.root.setLevel(logging.INFO)
    
    args = parse_args()
    """Create dataset and net"""
    cpu = torch.device("cpu")
    device = torch.device("cuda") if torch.cuda.is_available() else cpu
    batch_size = args.batch_size
    frame_shape = args.frame_shape

    dataset = PreprocessDataset(path_to_preprocess=args.data_dir, frame_shape=frame_shape)
    dataset = DatasetRepeater(dataset, num_repeats=10 if len(dataset) < 100 else 2)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
    )

    path_to_chkpt = os.path.join(args.train_dir, 'weights.pkl')
    if os.path.isfile(path_to_chkpt):
        checkpoint = torch.load(path_to_chkpt, map_location=cpu)

    net = nn.DataParallel(Generator(frame_shape, device).to(device))

    net.train()

    optimizer = optim.Adam(
        params=list(net.parameters()),
        lr=5e-4,
        amsgrad=False
    )
    """Loss"""
    loss_fun = LossG(
        os.path.join(args.vggface_dir, 'Pytorch_VGGFACE_IR.py'),
        os.path.join(args.vggface_dir, 'Pytorch_VGGFACE.pth'),
        device
    )

    """Training init"""
    epoch = i_batch = 0

    num_epochs = args.epochs

    # initiate checkpoint if inexistant
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
    if not os.path.isfile(path_to_chkpt):
        def init_weights(m):
            if type(m) == nn.Conv2d:
                torch.nn.init.xavier_uniform(m.weight)

        net.apply(init_weights)

        LOG.info('Initiating new checkpoint...')
        torch.save({
            'epoch': epoch,
            'state_dict': net.module.state_dict(),
            'num_vid': dataset.__len__(),
            'i_batch': i_batch,
            'optimizer': optimizer.state_dict(),
        }, path_to_chkpt)
        LOG.info('...Done')
        prev_step = 0
    else:
        """Loading from past checkpoint"""
        net.module.load_state_dict(checkpoint['state_dict'], strict=False)
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except ValueError:
            pass
        prev_step = checkpoint['i_batch']

    net.train()

    """Training"""
    writer = tensorboardX.SummaryWriter(args.train_dir)
    num_batches = len(dataset) / args.batch_size
    log_step = int(round(0.005 * num_batches + 20))
    log_epoch = 1
    if num_batches <= 100:
        log_step = 50
        log_epoch = 300 // num_batches
    save_checkpoint = args.save_checkpoint
    LOG.info(f"Will log each {log_step} step.")
    LOG.info(f"Will save checkpoint each {save_checkpoint} step.")
    if prev_step != 0:
        LOG.info(f"Starting at {prev_step} step.")

    for epoch in range(0, num_epochs):
        # if epochCurrent > epoch:
        #     pbar = tqdm(dataLoader, leave=True, initial=epoch, disable=None)
        #     continue
        # Reset random generator
        for i_batch, (src_img, _, target_img, target_mark, i) in enumerate(data_loader):

            src_img = src_img.to(device).reshape([-1, *list(src_img.shape[2:])])
            # marks = marks.to(device).reshape([-1, *list(marks.shape[2:])])
            target_mark = target_mark.to(device)
            target_img = target_img.to(device)

            with torch.autograd.enable_grad():
                # zero the parameter gradients
                optimizer.zero_grad()

                # train G and D
                fake = net(src_img, target_mark)

                loss = loss_fun(target_img, fake)
                loss.backward()
                optimizer.step()

            step = epoch * num_batches + i_batch + prev_step

            # Output training stats
            if step % log_step == 0:
                def get_picture(tensor):
                    return (tensor[0] * 127.5 + 127.5).permute([1, 2, 0]).type(torch.int32).to(cpu).numpy()

                def make_grid(tensor):
                    np_image = (tensor * 127.5 + 127.5).permute([0, 2, 3, 1]).type(torch.int32).to(cpu).numpy()
                    np_image = np_image.clip(0, 255).astype(np.uint8)
                    canvas = np.zeros([frame_shape, frame_shape, 3])
                    size = math.ceil(math.sqrt(tensor.shape[0]))
                    im_size = frame_shape // size
                    for i, im in enumerate(np_image):
                        col = i % size
                        row = i // size
                        im = cv2.resize(im, (im_size, im_size))
                        canvas[row * im_size:(row+1) * im_size, col*im_size:(col+1) * im_size] = im

                    return canvas

                out1 = get_picture(fake)
                out2 = get_picture(target_img)
                out3 = get_picture(target_mark)
                out4 = make_grid(src_img)

                accuracy = np.sum(np.squeeze((np.abs(out1 - out2) <= 1))) / np.prod(out1.shape)
                ssim = metrics.structural_similarity(out1.clip(0, 255).astype(np.uint8), out2.clip(0, 255).astype(np.uint8), multichannel=True)
                LOG.info(
                    'Step %d [%d/%d][%d/%d]\tLoss: %.4f\tMatch: %.3f\tSSIM: %.3f'
                    % (step, epoch, num_epochs, i_batch, len(data_loader),
                       loss.item(), accuracy, ssim)
                )

                image = np.hstack((out1, out2, out3, out4)).clip(0, 255).astype(np.uint8)
                writer.add_image(
                    'Result', image,
                    global_step=step,
                    dataformats='HWC'
                )
                writer.add_scalar('loss', loss.item(), global_step=step)
                writer.add_scalar('match', accuracy, global_step=step)
                writer.add_scalar('ssim', ssim, global_step=step)
                writer.flush()

            if step != 0 and step % save_checkpoint == 0:
                LOG.info('Saving latest...')
                torch.save({
                    'epoch': epoch,
                    'state_dict': net.module.state_dict(),
                    'num_vid': dataset.__len__(),
                    'i_batch': step,
                    'optimizer': optimizer.state_dict(),
                },
                    path_to_chkpt
                )
                LOG.info('...Done saving latest')

        if epoch % log_epoch == 0:
            LOG.info('Saving latest...')
            torch.save({
                'epoch': epoch,
                'state_dict': net.module.state_dict(),
                'num_vid': dataset.__len__(),
                'i_batch': step,
                'optimizer': optimizer.state_dict(),
            },
                path_to_chkpt
            )
            LOG.info('...Done saving latest')


if __name__ == '__main__':
    main()
