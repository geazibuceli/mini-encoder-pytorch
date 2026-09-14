import pytest
import torch

from miniencoder.attention import ManualMultiHeadSelfAttention, SDPAMultiHeadSelfAttention
from miniencoder.config import validate_config
from miniencoder.data import SentimentDataset
from miniencoder.model import TransformerClassifier
from miniencoder.positional_encoding import SinusoidalPositionalEncoding
from miniencoder.tokenizer import RegexTokenizer, Vocabulary, build_vocabulary
from miniencoder.train import build_model


@pytest.mark.parametrize("all_masked", [False, True])
def test_attention_outputs_and_gradients_match(all_masked):
    torch.manual_seed(42)
    manual = ManualMultiHeadSelfAttention(8, 2).eval()
    sdpa = SDPAMultiHeadSelfAttention(8, 2).eval()
    sdpa.load_state_dict(manual.state_dict())
    x = torch.randn(2, 4, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    mask = (
        torch.zeros(2, 4, dtype=torch.bool)
        if all_masked
        else torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    )
    a, b = manual(x, mask), sdpa(y, mask)
    torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-5)
    a.square().sum().backward()
    b.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-6, rtol=1e-5)
    for p, q in zip(manual.parameters(), sdpa.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("backend", ["manual", "sdpa"])
@pytest.mark.parametrize("pooling", ["cls", "mean"])
def test_padding_cannot_change_predictions(backend, pooling):
    model = TransformerClassifier(
        10,
        embedding_dimension=8,
        heads=2,
        layers=1,
        dropout=0,
        pooling=pooling,
        attention_backend=backend,
    ).eval()
    short = torch.tensor([[2, 3, 4]])
    padded = torch.tensor([[2, 3, 4, 8, 9]])
    torch.testing.assert_close(
        model(short, torch.ones_like(short, dtype=torch.bool)),
        model(padded, torch.tensor([[1, 1, 1, 0, 0]], dtype=torch.bool)),
    )


def test_tokenizer_vocabulary_and_truncation():
    tokenizer = RegexTokenizer()
    assert tokenizer.tokenize("It's GOOD!") == ["it's", "good", "!"]
    vocabulary = build_vocabulary(["good movie", "bad movie"], tokenizer)
    reversed_vocab = build_vocabulary(["bad movie", "good movie"], tokenizer)
    assert vocabulary.fingerprint() == reversed_vocab.fingerprint()
    row = SentimentDataset(["good movie extra"], [1], tokenizer, vocabulary, 2)[0]
    assert row["input_ids"].tolist() == [vocabulary.cls_id, vocabulary.token_to_id["good"]]
    assert vocabulary.encode(["missing"]) == [vocabulary.unk_id]
    with pytest.raises(ValueError, match="same length"):
        SentimentDataset(["good"], [], tokenizer, vocabulary, 2)
    with pytest.raises(ValueError, match="max_length"):
        SentimentDataset(["good"], [1], tokenizer, vocabulary, 0)
    with pytest.raises(ValueError, match="unique"):
        Vocabulary({"[PAD]": 0, "[UNK]": 1, "[CLS]": 1})


def test_positions_odd_width_and_capacity():
    position = SinusoidalPositionalEncoding(7, 4)
    result = position(torch.zeros(1, 4, 7))
    assert result.shape == (1, 4, 7)
    assert not torch.equal(result[:, 0], result[:, 1])
    with pytest.raises(ValueError, match="capacity"):
        position(torch.zeros(1, 5, 7))


@pytest.mark.parametrize(
    "config",
    [
        None,
        [],
        {"modle": {}},
        {"model": None},
        {"model": {"type": "typo"}},
        {"model": {"heads": 0}},
        {"model": {"embedding_dimension": 7}},
        {"model": {"dropout": 1}},
        {"training": {"epochs": 0}},
        {"training": {"batch_size": -1}},
        {"training": {"learning_rate": float("nan")}},
        {"training": {"unknown": 1}},
        {"model": {"positional_encoding": "false"}},
    ],
)
def test_invalid_config_is_rejected(config):
    with pytest.raises(ValueError):
        validate_config(config)


def test_factory_rejects_unknown_model():
    with pytest.raises(ValueError, match="model.type"):
        build_model({"type": "typo"}, 5)
