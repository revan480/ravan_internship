"""
Phase 3 CPU checks for all four frameworks (MoCo, BYOL, LooC, SimCLR).

For each: forward returns finite ssl_loss + h1/h2 with grad; the relational-loss
backward populates grads on the TRAINABLE backbone but NOT on the momentum/target
encoder; and the trunk-only checkpoint loads into a plain torchvision resnet with
missing == {fc.weight, fc.bias}.

Run:  python -m pytest relssl/tests/test_frameworks.py -q
"""

import os
import sys

import pytest
import torch
import torchvision.models as models

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from relssl.losses import RelPairLoss  # noqa: E402
from relssl.models.frameworks import (TRAINABLE_BACKBONE_ATTR,  # noqa: E402
                                      backbone_state_dict, build_model)
from relssl.models.rel_head import RelHead  # noqa: E402

FRAMEWORKS = ["simclr", "moco", "byol", "looc"]
# the no-grad encoder paired with each trainable backbone
MOMENTUM_ATTR = {"moco": "encoder_k", "byol": "target_backbone", "looc": "backbone_k"}


def _cfg(framework, arch="resnet18", rel_lambda=0.5):
    return {
        "arch": arch, "framework": framework, "rel_lambda": rel_lambda,
        "rel_head_hidden": 64,
        "rel_factors": ["rotation", "hflip", "brightness", "contrast",
                        "saturation", "hue", "grayscale", "blur"],
        # framework-specific (small queue for CPU)
        "simclr_dim": 128, "temperature": 0.5,
        "moco_dim": 128, "moco_k": 64, "moco_m": 0.999, "moco_t": 0.2,
        "proj_hidden_dim": 256, "proj_dim": 64, "tau_base": 0.996,
        "dim": 128, "K": 64, "m": 0.999, "T": 0.2, "n_aug": 0, "full_multiview": False,
    }


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_forward_h1h2_grad(fw):
    model = build_model(_cfg(fw)).train()
    v1, v2 = torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64)
    out = model(v1, v2)
    assert torch.isfinite(out.ssl_loss)
    assert out.h1.shape == (4, model.feat_dim)
    assert out.h2 is not None and out.h2.shape == (4, model.feat_dim)
    assert out.h1.requires_grad and out.h2.requires_grad


@pytest.mark.parametrize("fw", FRAMEWORKS)
def test_relloss_grads_trainable_backbone_only(fw):
    model = build_model(_cfg(fw)).train()
    head = RelHead(model.feat_dim, num_factors=8, hidden=64)
    crit = RelPairLoss()
    v1, v2 = torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64)
    out = model(v1, v2)
    labels = (torch.rand(4, 8) > 0.5).float()
    mask = torch.ones(4, 8)
    rel_loss, _, _ = crit(head(out.h1, out.h2), labels, mask)
    (out.ssl_loss + 0.5 * rel_loss).backward()

    trunk = getattr(model, TRAINABLE_BACKBONE_ATTR[fw])
    assert any(p.grad is not None for p in trunk.parameters()), "trainable backbone must get grad"
    assert any(p.grad is not None for p in head.parameters())
    # the momentum / target encoder must never receive gradient
    if fw in MOMENTUM_ATTR:
        mom = getattr(model, MOMENTUM_ATTR[fw])
        assert all(p.grad is None for p in mom.parameters()), "momentum encoder must NOT get grad"


@pytest.mark.parametrize("fw", FRAMEWORKS)
@pytest.mark.parametrize("arch", ["resnet18", "resnet50"])
def test_checkpoint_trunk_only_eval_compat(fw, arch):
    model = build_model(_cfg(fw, arch=arch))
    sd = backbone_state_dict(model, fw)
    assert all(not k.startswith("backbone.fc.") for k in sd)
    stripped = {k[len("backbone."):]: v for k, v in sd.items()}
    plain = models.__dict__[arch]()
    msg = plain.load_state_dict(stripped, strict=False)
    assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}, (fw, arch, msg.missing_keys)
    assert msg.unexpected_keys == [], (fw, arch, msg.unexpected_keys)


def test_moco_baseline_skips_extra_forward():
    # rel_lambda == 0 -> pair_feats False -> h2 is None (no extra query forward)
    model = build_model(_cfg("moco", rel_lambda=0.0)).train()
    out = model(torch.randn(4, 3, 64, 64), torch.randn(4, 3, 64, 64))
    assert out.h2 is None and out.h1 is not None


def test_looc_rejects_multiview():
    cfg = _cfg("looc")
    cfg["full_multiview"] = True
    with pytest.raises(NotImplementedError):
        build_model(cfg)


def test_byol_tau_schedule():
    model = build_model(_cfg("byol"))
    assert model._current_tau() == model.tau_base   # no total_steps -> fixed tau_base
    model.set_total_steps(1000)
    t0 = model._current_tau()
    assert abs(t0 - model.tau_base) < 1e-6           # step 0 -> tau_base
    model._step += 500                               # halfway -> tau between base and 1
    assert model.tau_base < model._current_tau() < 1.0


if __name__ == "__main__":
    import traceback
    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for name, fn in fns:
        try:
            # crude param expansion for __main__ runs
            if "fw" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                for fw in FRAMEWORKS:
                    if "arch" in fn.__code__.co_varnames[:fn.__code__.co_argcount]:
                        for arch in ("resnet18", "resnet50"):
                            fn(fw, arch)
                    else:
                        fn(fw)
            else:
                fn()
            print(f"PASS  {name}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {name}"); traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} test-fns passed")
    sys.exit(1 if failed else 0)
