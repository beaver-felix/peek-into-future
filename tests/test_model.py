import torch
from peek_future.models.encoder import ConformerEncoder
from peek_future.models.decoder import AttentionDecoder
from peek_future.models.aed import AEDModel
from peek_future.models.mtp_heads import MTPModule
from peek_future.models.entity_scorer import EntityScorer
from peek_future.models.biasing_model import BiasingModel
from peek_future.baselines.clas import CLASModel


def test_encoder():
    enc = ConformerEncoder(feat_dim=80, d_model=128, num_layers=2, n_heads=4, ff_expansion=2)
    x = torch.randn(2, 100, 80)
    lens = torch.tensor([100, 80])
    out, out_lens = enc(x, lens)
    assert out.size(0) == 2
    assert out.size(2) == 128
    assert out_lens[0] == (100 + 1) // 4


def test_decoder():
    dec = AttentionDecoder(vocab_size=100, d_model=128, num_layers=2, n_heads=4)
    tokens = torch.randint(0, 100, (2, 20))
    memory = torch.randn(2, 25, 128)
    h_d = dec(tokens, memory)
    assert h_d.shape == (2, 20, 128)


def test_mtp_module():
    mtp = MTPModule(k_heads=4, d_model=128, head_type="linear")
    h_d = torch.randn(2, 15, 128)
    lm_head = torch.nn.Linear(128, 100, bias=False)
    logits = mtp(h_d, lm_head)
    assert logits.shape == (2, 15, 4, 100)


def test_entity_scorer():
    scorer = EntityScorer(k_heads=4, scorer_type="learned", hidden_dim=32)
    mtp_logits = torch.randn(2, 10, 4, 100)
    bias_tokens = torch.randint(0, 100, (2, 5, 4))
    z_scores, P_e = scorer(mtp_logits, bias_tokens)
    assert z_scores.shape == (2, 10, 5)
    assert P_e.shape == (2, 10, 5)
    # Check probabilities sum to 1 across entities
    sums = P_e.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_biasing_model_forward():
    model = BiasingModel(
        vocab_size=100,
        feat_dim=80,
        d_model=128,
        enc_layers=2,
        dec_layers=2,
        n_heads=4,
        k_heads=4,
    )
    x = torch.randn(2, 60, 80)
    x_lens = torch.tensor([60, 50])
    y = torch.randint(4, 95, (2, 16))
    y[:, 0] = 2  # BOS
    bias_tokens = torch.randint(0, 100, (2, 6, 4))
    tau_labels = torch.randint(0, 6, (2, 15))

    out = model(x, x_lens, y, bias_tokens=bias_tokens, tau_labels=tau_labels)
    assert "loss_total" in out
    assert "loss_mtp" in out
    assert "loss_entity" in out
    assert out["loss_total"].item() > 0

    # Test backward pass
    out["loss_total"].backward()


def test_clas_model_forward():
    model = CLASModel(
        vocab_size=100,
        feat_dim=80,
        d_model=128,
        bias_hidden_dim=64,
        enc_layers=2,
        dec_layers=2,
        n_heads=4,
    )
    x = torch.randn(2, 60, 80)
    x_lens = torch.tensor([60, 50])
    y = torch.randint(4, 95, (2, 16))
    bias_tokens = torch.randint(0, 100, (2, 5, 8))
    logits, loss = model(x, x_lens, y, bias_tokens=bias_tokens)
    assert logits.shape == (2, 15, 100)
    assert loss.item() > 0


if __name__ == "__main__":
    print("Running model tests...")
    test_encoder()
    print("Encoder passed.")
    test_decoder()
    print("Decoder passed.")
    test_mtp_module()
    print("MTP module passed.")
    test_entity_scorer()
    print("Entity scorer passed.")
    test_biasing_model_forward()
    print("Biasing model forward passed.")
    test_clas_model_forward()
    print("CLAS model forward passed.")
    print("All model tests passed successfully!")
