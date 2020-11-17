
import torch
import argparse
from model import TE, TEMMA


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TEMMA')

    parser.add_argument('--mask_a_length', type=str, default='50,50')
    parser.add_argument('--mask_b_length', type=str, default='10,10')
    parser.add_argument('--block_num', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--dropout_mmatten', type=float, default=0.5)
    parser.add_argument('--dropout_mtatten', type=float, default=0.2)
    parser.add_argument('--dropout_ff', type=float, default=0.2)
    parser.add_argument('--dropout_subconnect', type=float, default=0.2)
    parser.add_argument('--dropout_position', type=float, default=0.2)
    parser.add_argument('--dropout_embed', type=float, default=0.2)
    parser.add_argument('--dropout_fc', type=float, default=0.2)
    parser.add_argument('--h', type=int, default=4)
    parser.add_argument('--h_mma', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--d_ff', type=int, default=256)
    parser.add_argument('--modal_num', type=int, default=2)
    parser.add_argument('--embed', type=str, default='temporal')
    parser.add_argument('--levels', type=int, default=5)
    parser.add_argument('--ksize', type=int, default=3)
    parser.add_argument('--ntarget', type=int, default=2)

    opts = parser.parse_args()


    nbatch = 2
    seq_len = 10

    opts.modal_num = 1
    opts.mask_a_length = '50'
    opts.mask_b_length = '10'
    x0 = torch.rand(nbatch, seq_len, opts.d_model * opts.modal_num)
    te = TE(opts, opts.d_model * opts.modal_num)
    output = te(x0)
    print(output.shape)

    opts.modal_num = 2
    opts.mask_a_length = '50,50'
    opts.mask_b_length = '10,10'
    x1 = torch.rand(nbatch, seq_len, opts.d_model * opts.modal_num)
    temma = TEMMA(opts, opts.d_model * opts.modal_num)
    output = temma(x1)
    print(output.shape)
