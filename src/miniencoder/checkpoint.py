"""State-dict checkpoints with compatibility-aware loading."""

import inspect
from pathlib import Path

import torch


def save_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, **metadata):
    if hasattr(model, "model_config"):
        metadata.setdefault("model_config", model.model_config)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict() if optimizer else None,
               "scheduler_state": scheduler.state_dict() if scheduler else None,
               "scaler_state": scaler.state_dict() if scaler else None, **metadata}
    torch.save(payload, path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, scaler=None, map_location="cpu"):
    kwargs = {"map_location": map_location}
    if "weights_only" in inspect.signature(torch.load).parameters: kwargs["weights_only"] = False
    payload = torch.load(path, **kwargs)
    model.load_state_dict(payload["model_state"])
    if optimizer and payload.get("optimizer_state"): optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler and payload.get("scheduler_state"): scheduler.load_state_dict(payload["scheduler_state"])
    if scaler and payload.get("scaler_state"): scaler.load_state_dict(payload["scaler_state"])
    return payload
