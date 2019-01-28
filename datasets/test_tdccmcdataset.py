"""
test_tdccmcdataset.py
"""
import torch
from torch.utils.data import DataLoader

from .tdccmcdataset import TDCCMCDataset

dataset = TDCCMCDataset(
    filenames=["./videos/2AYaxTiWKoY.mp4"], trims=[(0, 200)], crops=[(0, 0, 140, 140)]
)
loader = DataLoader(dataset, batch_size=32, num_workers=4)
batch = next(iter(loader))
batch_v, batch_w, batch_a, batch_tdc_label, batch_cmc_label = batch


def test_batch_type():
    """
    Test if the batch returned by loader is of correct type.
    """
    print(type(batch_v))
    print(type(batch_w))
    print(type(batch_a))
    print(batch_tdc_label)
    print(batch_cmc_label)
    assert type(batch_v) == torch.Tensor
    assert type(batch_w) == torch.Tensor
    assert type(batch_a) == torch.Tensor
    assert type(batch_tdc_label) == torch.Tensor
    assert type(batch_cmc_label) == torch.Tensor


def test_batch_dtype():
    """
    Test if the batch returned by loader is of correct datatype.
    """
    print(batch_v.dtype)
    print(batch_w.dtype)
    print(batch_a.dtype)
    print(batch_tdc_label.dtype)
    print(batch_cmc_label.dtype)
    assert batch_v.dtype == torch.float32
    assert batch_w.dtype == torch.float32
    assert batch_a.dtype == torch.float32
    assert batch_tdc_label.dtype == torch.int64
    assert batch_cmc_label.dtype == torch.int64


def test_batch_shape():
    """
    Test if the batch returned by loader is of correct shape.
    """
    print(batch_v.shape)
    print(batch_w.shape)
    print(batch_a.shape)
    print(batch_tdc_label.shape)
    print(batch_cmc_label.shape)
    assert batch_v.shape == torch.Size([32, 12, 128, 128])
    assert batch_w.shape == torch.Size([32, 12, 128, 128])
    # TODO Find correct shape
    # assert batch_a.shape == torch.Size([])
    assert batch_tdc_label.shape == torch.Size([32, 1])
    assert batch_cmc_label.shape == torch.Size([32, 1])
