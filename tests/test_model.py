import torch

from sinalnet.model import CLASSES, NUM_CLASSES, SinalNet


def test_num_classes_bate_com_a_lista():
    assert NUM_CLASSES == len(CLASSES) == 43


def test_forward_retorna_logits_por_classe():
    model = SinalNet()
    x = torch.rand(4, 3, 48, 48)
    logits = model(x)
    assert logits.shape == (4, NUM_CLASSES)
