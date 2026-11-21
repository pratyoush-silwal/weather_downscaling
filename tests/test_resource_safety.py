import tempfile
import unittest
import zipfile
from pathlib import Path

import torch

from src.data.dataset import MonthlySampler
from src.data.preprocess_era5 import _single_level_paths
from src.training.metrics import StreamingMetrics, summarize_metrics
from src.training.train_baseline import rows_from_dataset


class ResourceSafetyTests(unittest.TestCase):
    def test_training_validation_checkpoint_smoke(self):
        from src.models import build_pignn_from_config, PIGNNLoss
        from src.training.train import make_loader, evaluate_epoch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            graph = root / "graph.pt"
            torch.save({"pos": torch.tensor([[27., 85.], [27.1, 85.], [27.2, 85.]]),
                        "static_x": torch.zeros(3, 3), "edge_index": torch.tensor([[0, 1, 2], [1, 2, 0]]),
                        "edge_attr": torch.ones(3, 3)}, graph)
            dynamic, target = root / "dynamic.pt", root / "target.pt"
            torch.save({"x_dynamic": torch.randn(4, 3, 6), "time_features": torch.zeros(4, 2),
                        "timestamps": [str(i) for i in range(4)]}, dynamic)
            torch.save({"y": torch.randn(4, 3, 4)}, target)
            loader = make_loader(graph, [dynamic], [target], 1, True)
            model = build_pignn_from_config({"hidden_channels": 8, "edge_hidden_channels": 4,
                                            "message_hidden_channels": 8, "num_layers": 1})
            optimizer = torch.optim.Adam(model.parameters())
            criterion = PIGNNLoss()
            for batch in loader:
                optimizer.zero_grad()
                prediction = model(batch["x"], batch["edge_index"], batch["edge_attr"])
                loss = criterion(prediction, batch["y"], batch["edge_index"], batch["x"][..., 6], batch["pos"])["total"]
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                optimizer.step()
            loss, metrics = evaluate_epoch(model, loader, criterion, torch.device("cpu"))
            self.assertEqual(len(metrics), 12)
            torch.save(model.state_dict(), root / "checkpoint.pt")
            model.load_state_dict(torch.load(root / "checkpoint.pt", weights_only=True))

    def test_streaming_matches_full(self):
        prediction, target = torch.randn(7, 5, 4), torch.randn(7, 5, 4)
        target[1, 2, 1] = float("nan")
        mask = torch.tensor([True, False, True, True, False])
        metrics = StreamingMetrics()
        for p, t in zip(prediction, target):
            metrics.update(p, t, mask)
        expected = summarize_metrics(prediction, target, mask)
        for name, value in metrics.compute().items():
            self.assertAlmostEqual(value, expected[name], places=6)
        self.assertEqual(metrics.totals.shape, (4, 4))

    def test_month_locality_and_row_budget(self):
        class Dataset:
            _lengths = [3, 4]
            _cumulative = [3, 7]
            static_x = torch.zeros(5, 3)

            def __len__(self):
                return 7

            def __getitem__(self, index):
                return {"x": torch.full((5, 11), float(index)), "y": torch.zeros(5, 4)}

        data = Dataset()
        order = list(MonthlySampler(data))
        self.assertEqual(sorted(order), list(range(7)))
        months = [index >= 3 for index in order]
        self.assertEqual(sum(a != b for a, b in zip(months, months[1:])), 1)
        x, y = rows_from_dataset(data, 9, 42)
        self.assertEqual(x.shape, (9, 11))
        self.assertEqual(y.shape, (9, 4))

    def test_extraction_repair_and_freshness(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw.nc"
            def write(value):
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("nested/stepType-instant.nc", value)
                    archive.writestr("nested/stepType-accum.nc", value)
            write("original")
            paths = _single_level_paths(path)
            paths["instant"].write_text("bad")
            self.assertEqual(_single_level_paths(path)["instant"].read_text(), "original")
            write("replacement")
            self.assertEqual(_single_level_paths(path)["accum"].read_text(), "replacement")
