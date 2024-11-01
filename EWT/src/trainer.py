import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
import torch.nn.functional as F
from tqdm import tqdm


os.environ["CUDA_VISIBLE_DEVICES"] = "3, 4"

class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.lr = args.lr

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

        self.dwt = utility.DWT()

    def _padding(self, x, scale):
        delta_H = 0
        delta_W = 0
        if x.shape[2] % scale != 0:
            delta_H = scale - x.shape[2] % scale
            x = F.pad(x, (0, 0, 0, delta_H), 'reflect')
        if x.shape[3] % scale != 0:
            delta_W = scale - x.shape[3] % scale
            x = F.pad(x, (0, delta_W, 0, 0), 'reflect')
        return x, delta_H, delta_W

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        #lr = self.optimizer.get_lr()

        lr = self.lr * (0.5 ** (epoch // 100))

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            # lr, hr = lr[:, :3, :, :], hr[:, :3, :, :]
            # hr_, _, _ = self._padding(hr, 2)
            # hr_x2 = self.dwt(hr_)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            # sr_, _, _ = self._padding(sr, 2)
            # sr_x2 = self.dwt(sr_)
            # loss = self.loss(sr, hr) + self.loss(sr_x2, hr_x2)
            loss = self.loss(sr, hr)
            # loss = self.loss(sr_x2, hr_x2)
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(self.loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    # print(hr.shape)
                    lr, hr = self.prepare(lr, hr)
                    lr, hr = lr[:, :3, :, :], hr[:, :3, :, :]
                    # lr = lr[:, :, 1000:2000, 500:1800]
                    # hr = hr[:, :, 1000:2000, 500:1800]
                    # lr = lr[:, :, 1000:1200, 1000:1300]
                    # hr = hr[:, :, 1000:1200, 1000:1300]
                    sr = self.model(lr, idx_scale)
                    # sr = sr[:, :, 600:2200, :900]
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )

                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                best = self.ckp.log.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best[0][idx_data, idx_scale],
                        best[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs