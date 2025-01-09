from utils import dpw3d as datasets
from model import AttModel
from utils.opt import Options
from utils import util
from utils import log

from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import numpy as np
import time
import h5py
import torch.optim as optim
import time

def main(opt):
    print('>>> create models')
    in_features = opt.in_features
    d_model = opt.d_model
    kernel_size = opt.kernel_size
    dev = opt.dev
    net_pred = AttModel.AttModel(in_features=in_features, kernel_size=kernel_size, d_model=d_model,
                                 num_stage=opt.num_stage, dct_n=opt.dct_n)
    net_pred.load_state_dict(
        torch.load("./checkpoint/main_3dpw_3d_eval_in50_out12_ks10_dctn22/ckpt_best.pth.tar", map_location=dev)['state_dict']
    )
    net_pred.to(dev)

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in net_pred.parameters()) / 1000000.0))
    print('>>> loading datasets')

    test_dataset = datasets.Datasets(opt, split=2)
    print('>>> Testing dataset length: {:d}'.format(test_dataset.__len__()))
    test_loader = DataLoader(
        test_dataset, batch_size=opt.test_batch_size, shuffle=False, num_workers=0, pin_memory=True
    )
    ret_test = run_model(net_pred, is_train=3, data_loader=test_loader, opt=opt)
    ret_log = np.array([])
    head = np.array([])
    for k in ret_test.keys():
        ret_log = np.append(ret_log, [ret_test[k]])
        head = np.append(head, [k])
    print(ret_test)


def run_model(net_pred, optimizer=None, is_train=0, data_loader=None, epo=1, opt=None):

    dev = opt.dev

    if is_train == 0:
        net_pred.train()
    else:
        net_pred.eval()

    l_p3d = 0
    if is_train <= 1:
        m_p3d_h36 = 0
    else:
        titles = np.array(range(opt.output_n)) + 1
        m_p3d_h36 = np.zeros([opt.output_n])
    n = 0
    in_n = opt.input_n
    out_n = opt.output_n

    dim_used = np.arange(12, 66)
    seq_in = opt.kernel_size

    itera = 1
    idx = np.expand_dims(np.arange(seq_in + out_n), axis=1) + (
            out_n - seq_in + np.expand_dims(np.arange(itera), axis=0))
    st = time.time()
    for i, (p3d_h36) in enumerate(data_loader):
        p3d_h36 = p3d_h36 * 1000
        p3d_h36 = p3d_h36.flatten(2)
        batch_size, seq_n, _ = p3d_h36.shape

        if batch_size == 1 and is_train == 0:
            continue
        n += batch_size
        bt = time.time()
        p3d_h36 = p3d_h36.float().to(dev)
        p3d_sup = p3d_h36.clone()[:, :, dim_used][:, -out_n - seq_in:].reshape(
            [-1, seq_in + out_n, len(dim_used) // 3, 3])
        p3d_src = p3d_h36.clone()[:, :, dim_used]
        p3d_out_all = net_pred(p3d_src, input_n=in_n, output_n=out_n, itera=itera, dev=dev)

        p3d_sup_v = p3d_sup[:, 1:, :] - p3d_sup[:, :-1, :]
        p3d_sup_a = p3d_sup_v[:, 1:, :] - p3d_sup_v[:, :-1, :]
        p3d_sup_v = p3d_sup_v[:, -(seq_in + out_n + 1):, :]
        p3d_sup_a = p3d_sup_a[:, -(seq_in + out_n + 2):, :]
        p3d_out_all_v = p3d_out_all[:, 1:, :] - p3d_out_all[:, :-1, :]
        p3d_out_all_a = p3d_out_all_v[:, 1:, :] - p3d_out_all_v[:, :-1, :]
        p3d_sup_v = p3d_sup_v.reshape(batch_size, seq_in + out_n - 1, -1, 3)
        p3d_sup_a = p3d_sup_a.reshape(batch_size, seq_in + out_n - 2, -1, 3)
        p3d_out_all_v = p3d_out_all_v.reshape(batch_size, seq_in + out_n - 1, itera, -1, 3)
        p3d_out_all_a = p3d_out_all_a.reshape(batch_size, seq_in + out_n - 2, itera, -1, 3)

        p3d_out = p3d_h36.clone()[:, in_n:in_n + out_n]
        p3d_out[:, :, dim_used] = p3d_out_all[:, seq_in:, 0]
        p3d_out = p3d_out.reshape([-1, out_n, 22, 3])

        p3d_h36 = p3d_h36.reshape([-1, in_n + out_n, 22, 3])

        p3d_out_all = p3d_out_all.reshape([batch_size, seq_in + out_n, itera, len(dim_used) // 3, 3])

        grad_norm = 0
        if is_train == 0:
            loss_p3d = torch.mean(torch.norm(p3d_out_all[:, :, 0] - p3d_sup, dim=3))

            loss_p3d_v = torch.mean(torch.norm(p3d_out_all_v[:, :, 0] - p3d_sup_v, dim=3))
            loss_p3d_a = torch.mean(torch.norm(p3d_out_all_a[:, :, 0] - p3d_sup_a, dim=3))

            loss_all = loss_p3d + 0.3 * loss_p3d_v + 0.3 * loss_p3d_a
            optimizer.zero_grad()
            loss_all.backward()
            nn.utils.clip_grad_norm_(list(net_pred.parameters()), max_norm=opt.max_norm)
            optimizer.step()
            l_p3d += loss_p3d.cpu().data.numpy() * batch_size

        if is_train <= 1:  
            mpjpe_p3d_h36 = torch.mean(torch.norm(p3d_h36[:, in_n:in_n + out_n] - p3d_out, dim=3))
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy() * batch_size
        else:
            mpjpe_p3d_h36 = torch.sum(torch.mean(torch.norm(p3d_h36[:, in_n:] - p3d_out, dim=3), dim=2), dim=0)
            m_p3d_h36 += mpjpe_p3d_h36.cpu().data.numpy()
        if i % 1000 == 0:
            print('{}/{}|bt {:.3f}s|tt{:.0f}s|gn{}'.format(i + 1, len(data_loader), time.time() - bt,
                                                           time.time() - st, grad_norm))
    ret = {}
    if is_train == 0:
        ret["l_p3d"] = l_p3d / n

    if is_train <= 1:
        ret["m_p3d_h36"] = m_p3d_h36 / n
    else:
        m_p3d_h36 = m_p3d_h36 / n
        for j in range(out_n):
            ret["#{:d}".format(titles[j])] = round(m_p3d_h36[j], 2)
    return ret


if __name__ == '__main__':
    option = Options().parse()
    main(option)
