

# Haifeng Chen
# a demo dataset for esay data augment


import torch
import torch.utils.data as data
import numpy as np




class Avec2019(data.Dataset):

    def __init__(self, opts):
        super(Avec2019, self).__init__()

        self.load_dataset()
        self.data_expansion(opts.sampler_len, opts.sampler_hop)


    def load_dataset(self):
        self.video_num = 4
        self.video_real_length = np.array([1751, 1824, 1500, 600])
        img_size = 64

        self.data = np.zeros((self.video_num, max(self.video_real_length), 3, img_size, img_size))


    def data_expansion(self, sampler_len, sampler_hop):
        self.expansion_num = 0
        self.expansion_index = []

        for i in range(self.video_num):
            real_len = self.video_real_length[i]
            i_start = 0
            i_end = i_start + sampler_len

            while (i_end < real_len):
                self.expansion_index.append([i, i_start, i_end])
                self.expansion_num += 1

                i_start = i_start + sampler_hop
                i_end   = i_start + sampler_len

            i_end = real_len
            i_start = i_end - sampler_len
            self.expansion_index.append([i, i_start, i_end])
            self.expansion_num += 1


    def __getitem__(self, index):
        video_index = self.expansion_index[index][0]
        i_start     = self.expansion_index[index][1]
        i_end = self.expansion_index[index][2]

        img_seq = self.data[video_index, i_start:i_end, :,:,:]

        return img_seq


    def __len__(self):
        return self.expansion_num


def avec2019_data_loader(opts):
    dataset = Avec2019(opts)
    data_loader = torch.utils.data.DataLoader(dataset, opts.batch_size)

    return data_loader



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='AffectNet Database')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--sampler_len', type=int, default=300)
    parser.add_argument('--sampler_hop', type=int, default=100)
    opts = parser.parse_args()

    data_loader = avec2019_data_loader(opts)

    for i, img_seq in enumerate(data_loader):
        print(img_seq.shape)
