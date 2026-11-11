import numpy as np
import pytest
import torch

from multicam_tracker.config import ReIDConfig
from multicam_tracker.detection.base import FrameContext
from multicam_tracker.reid.evaluate import evaluate_rank
from multicam_tracker.reid.losses import batch_hard_triplet_loss
from multicam_tracker.reid.model import ReIDNet
from multicam_tracker.reid.sampler import PKSampler

from builders import write_market


def test_train_forward_returns_logits_and_features():
    model = ReIDNet("resnet18", num_classes=5, pretrained=False).train()
    logits, feats = model(torch.randn(4, 3, 64, 32))
    assert logits.shape == (4, 5) and feats.shape == (4, 512)


def test_eval_forward_returns_unit_embeddings():
    model = ReIDNet("resnet18", num_classes=5, pretrained=False).eval()
    with torch.no_grad():
        emb = model(torch.randn(3, 3, 64, 32))
    np.testing.assert_allclose(emb.norm(dim=1).numpy(), 1.0, rtol=1e-5)


def test_last_stride_one_doubles_final_resolution():
    x = torch.randn(1, 3, 256, 128)
    assert ReIDNet("resnet18", 5, pretrained=False, last_stride=1).backbone(x).shape[-2:] == (16, 8)
    assert ReIDNet("resnet18", 5, pretrained=False, last_stride=2).backbone(x).shape[-2:] == (8, 4)


def test_batch_hard_triplet_loss_hand_computed():
    feats = torch.tensor([[0.0], [5.0], [4.0], [9.0]])
    labels = torch.tensor([0, 0, 1, 1])
    # Hardest positive / negative per anchor: (5,4) (5,1) (5,1) (5,4) -> 1.3, 4.3, 4.3, 1.3
    assert batch_hard_triplet_loss(feats, labels, margin=0.3).item() == pytest.approx(2.8)


def test_batch_hard_triplet_loss_is_zero_when_separated():
    feats = torch.tensor([[0.0], [1.0], [10.0], [12.0]])
    assert batch_hard_triplet_loss(feats, torch.tensor([0, 0, 1, 1]), margin=0.3).item() == 0.0


def test_pk_sampler_batches_p_identities_with_k_images_each():
    labels = [0, 0, 0, 1, 1, 2, 3, 3]
    sampler = PKSampler(labels, p=2, k=3, seed=0)
    batches = list(sampler)
    assert len(batches) == len(sampler) == 2
    for batch in batches:
        assert len(batch) == 6
        batch_labels = [labels[i] for i in batch]
        assert len(set(batch_labels)) == 2
        assert all(batch_labels.count(lab) == 3 for lab in set(batch_labels))
    assert sorted({labels[i] for b in batches for i in b}) == [0, 1, 2, 3]


def test_market_ranking_protocol():
    # Gallery 0 shares the query's camera and identity, so it is excluded. Ranking then goes
    # wrong, right, right: AP = (1/2 + 2/3) / 2.
    q = np.array([[1.0, 0.0]])
    g = np.array([[1.0, 0.0], [0.995, 0.0998], [0.98, 0.199], [0.955, 0.296]])
    cmc, mAP = evaluate_rank(q, np.array([1]), np.array([1]), g, np.array([1, 2, 1, 1]), np.array([1, 2, 2, 3]))
    assert cmc[0] == 0.0 and cmc[1] == 1.0 and cmc[4] == 1.0
    assert mAP == pytest.approx((1 / 2 + 2 / 3) / 2)


def test_query_without_valid_match_is_skipped():
    q = np.array([[1.0, 0.0], [0.0, 1.0]])
    g = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Query 0's only true match is in its own camera; query 1 matches gallery 1 at rank 1.
    cmc, mAP = evaluate_rank(q, np.array([1, 2]), np.array([1, 1]), g, np.array([1, 2]), np.array([1, 2]))
    assert cmc[0] == 1.0 and mAP == pytest.approx(1.0)


def tiny_market(root):
    rng = np.random.default_rng(0)
    names = {"bounding_box_train": [], "query": [], "bounding_box_test": []}
    for pid in range(1, 5):
        for i in range(4):
            names["bounding_box_train"].append(f"{pid:04d}_c{1 + i % 2}s1_{i:06d}_00.jpg")
        names["query"].append(f"{pid:04d}_c1s1_{99:06d}_00.jpg")
        names["bounding_box_test"].append(f"{pid:04d}_c2s1_{98:06d}_00.jpg")
    write_market(root, names)
    import cv2

    # Give each identity a distinct colour so the model has something to learn.
    for sub in names:
        for n in names[sub]:
            pid = int(n[:4])
            img = np.full((64, 32, 3), 40 * pid, np.uint8) + rng.integers(0, 20, (64, 32, 3), dtype=np.uint8)
            cv2.imwrite(str(root / sub / n), img)


def test_training_reduces_loss_and_writes_a_loadable_checkpoint(tmp_path):
    from multicam_tracker.reid.extractor import TorchReIDEmbedder
    from multicam_tracker.reid.train import ReIDTrainConfig, train

    tiny_market(tmp_path / "market")
    cfg = ReIDTrainConfig(
        data_root=str(tmp_path / "market"), backbone="resnet18", pretrained=False, input_size=[64, 32],
        p=2, k=4, epochs=8, lr=1e-3, warmup_epochs=0, milestones=[100], eval_every=8,
        num_workers=0, amp=False, device="cpu", output_dir=str(tmp_path / "run"),
    )
    history = train(cfg)
    assert np.mean(history["loss"][-2:]) < np.mean(history["loss"][:2])
    assert "mAP" in history["eval"][-1]

    ckpt = tmp_path / "run" / "last.pt"
    assert ckpt.exists()
    emb = TorchReIDEmbedder(ReIDConfig(enabled=True, checkpoint=str(ckpt), input_size=[64, 32], device="cpu"))
    image = np.zeros((100, 100, 3), np.uint8)
    boxes = np.array([[10.0, 10, 40, 90], [-20.0, 50, 30, 130], [50.0, 50, 50.5, 50.5]])
    out = emb.embed(FrameContext(sequence="s", camera=0, frame=0, image_fn=lambda: image), boxes)
    assert out.shape == (3, 512)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, rtol=1e-5)


def test_init_from_copies_backbone_but_not_the_classifier(tmp_path):
    from multicam_tracker.reid.train import ReIDTrainConfig, build_model

    source = ReIDNet("resnet18", num_classes=7, pretrained=False)
    torch.save({"model": source.state_dict(), "meta": source.config}, tmp_path / "src.pt")
    cfg = ReIDTrainConfig(backbone="resnet18", pretrained=False, init_from=str(tmp_path / "src.pt"))
    model = build_model(cfg, num_classes=3)
    assert torch.equal(model.backbone[0].weight, source.backbone[0].weight)
    assert model.classifier.weight.shape == (3, 512)
