import unittest

import torch
import yaml

from src.models import build_pignn_from_config, build_mlp_baseline_from_config
from src.models.losses import PIGNNLoss, LossWeights
from src.utils.normalization import affine_parameters


class NormalizationTests(unittest.TestCase):
    def test_validation(self):
        for scales in ([0, 1], [float("nan"), 1], [1]):
            with self.assertRaises(ValueError):
                affine_parameters(2, scales=scales)

    def test_equal_scaled_errors_and_missing_target(self):
        scales = torch.tensor([2., .001, 1., 1.])
        prediction = scales.repeat(3, 1).requires_grad_()
        target = torch.zeros_like(prediction)
        target[0, 1] = float("nan")
        criterion = PIGNNLoss(weights=LossWeights(), channel_scales=scales.tolist())
        result = criterion(prediction, target, torch.tensor([[0, 1], [1, 2]]), torch.zeros(3), torch.zeros(3, 2))
        self.assertAlmostEqual(result["data"].item(), 1.)
        result["total"].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_models_keep_physical_residuals_and_checkpoint_scales(self):
        with open("configs/default.yaml") as handle:
            config = yaml.safe_load(handle)
        edges = torch.tensor([[0, 1], [1, 2]])
        for model in (build_pignn_from_config(config["model"]), build_mlp_baseline_from_config(config["baselines"]["mlp"])):
            model.eval()
            for parameter in model.parameters():
                parameter.data.zero_()
            x = model.input_offset.expand(3, -1).clone()
            x[:, 5] = .002
            output = model(x, edges, torch.ones(2, 3)) if hasattr(model, "edge_scale") else model(x)
            torch.testing.assert_close(output[:, 0], x[:, 0])
            torch.testing.assert_close(output[:, 1], x[:, 5] + torch.nn.functional.softplus(torch.tensor(0.)) * .001)
            self.assertIn("input_scale", model.state_dict())
