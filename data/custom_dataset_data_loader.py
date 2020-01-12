import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    from data.aligned_dataset import AlignedDataset
    dataset = AlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def CreateSelfDataset(opt):
    dataset = None
    from data.label_aligned_dataset import LabelAlignedDataset
    dataset = LabelAlignedDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

def CreateSingleDataset(opt):
    dataset = None
    from data.single_dataset import SingleDataset
    dataset = SingleDataset()

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, selfdata = 0):
        BaseDataLoader.initialize(self, opt)
        if selfdata == 0:
            self.dataset = CreateDataset(opt)
        elif selfdata == 2:
            self.dataset = CreateSingleDataset(opt)
        else:
            self.dataset = CreateSelfDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
