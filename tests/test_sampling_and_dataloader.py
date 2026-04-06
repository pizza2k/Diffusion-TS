import importlib.util
import math
import sys
import types
import unittest
from contextlib import contextmanager
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@contextmanager
def patched_modules(mapping):
    original = {}
    for name, module in mapping.items():
        original[name] = sys.modules.get(name)
        sys.modules[name] = module
    try:
        yield
    finally:
        for name, old in original.items():
            if old is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = old


def load_module(module_name, file_path, extra_modules):
    with patched_modules(extra_modules):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module


class BuildDataloaderTests(unittest.TestCase):
    def _load_build_dataloader_module(self, dataset_obj):
        def fake_dataloader(dataset, **kwargs):
            return {'dataset': dataset, 'kwargs': kwargs}

        torch_mod = types.ModuleType('torch')
        torch_mod.utils = types.SimpleNamespace(
            data=types.SimpleNamespace(DataLoader=fake_dataloader)
        )

        utils_pkg = types.ModuleType('Utils')
        io_utils_mod = types.ModuleType('Utils.io_utils')
        io_utils_mod.instantiate_from_config = lambda cfg: dataset_obj

        return load_module(
            'build_dataloader_test_module',
            ROOT / 'Data' / 'build_dataloader.py',
            {
                'torch': torch_mod,
                'Utils': utils_pkg,
                'Utils.io_utils': io_utils_mod,
            },
        )

    def test_build_dataloader_adds_fallback_column_names(self):
        dataset = types.SimpleNamespace(var_num=3)
        mod = self._load_build_dataloader_module(dataset)

        config = {
            'dataloader': {
                'batch_size': 2,
                'shuffle': True,
                'train_dataset': {'params': {}},
            }
        }
        args = types.SimpleNamespace(save_dir='/tmp/output')

        info = mod.build_dataloader(config, args)

        self.assertEqual(info['column_names'], ['feature_0', 'feature_1', 'feature_2'])
        self.assertEqual(dataset.column_names, ['feature_0', 'feature_1', 'feature_2'])

    def test_build_dataloader_keeps_existing_column_names(self):
        dataset = types.SimpleNamespace(var_num=3, column_names=['a', 'b', 'c'])
        mod = self._load_build_dataloader_module(dataset)

        config = {
            'dataloader': {
                'batch_size': 2,
                'shuffle': False,
                'train_dataset': {'params': {}},
            }
        }
        args = types.SimpleNamespace(save_dir='/tmp/output')

        info = mod.build_dataloader(config, args)

        self.assertEqual(info['column_names'], ['a', 'b', 'c'])


class TrainerSampleTests(unittest.TestCase):
    def _load_solver_module(self):
        fake_np = types.ModuleType('numpy')
        fake_np.ceil = math.ceil
        fake_np.empty = lambda shape: []
        fake_np.concatenate = lambda arrays, axis=0: [item for arr in arrays for item in arr]

        torch_mod = types.ModuleType('torch')
        torch_mod.cuda = types.SimpleNamespace(empty_cache=lambda: None)

        torch_nn_mod = types.ModuleType('torch.nn')
        torch_nn_func_mod = types.ModuleType('torch.nn.functional')

        torch_optim_mod = types.ModuleType('torch.optim')
        torch_optim_mod.Adam = object

        torch_nn_utils_mod = types.ModuleType('torch.nn.utils')
        torch_nn_utils_mod.clip_grad_norm_ = lambda *args, **kwargs: None

        tqdm_mod = types.ModuleType('tqdm')
        tqdm_auto_mod = types.ModuleType('tqdm.auto')
        tqdm_auto_mod.tqdm = lambda *args, **kwargs: None

        ema_mod = types.ModuleType('ema_pytorch')
        ema_mod.EMA = object

        utils_pkg = types.ModuleType('Utils')
        io_utils_mod = types.ModuleType('Utils.io_utils')
        io_utils_mod.instantiate_from_config = lambda cfg: None
        io_utils_mod.get_model_parameters_info = lambda model: 'params'

        return load_module(
            'solver_test_module',
            ROOT / 'engine' / 'solver.py',
            {
                'numpy': fake_np,
                'torch': torch_mod,
                'torch.nn': torch_nn_mod,
                'torch.nn.functional': torch_nn_func_mod,
                'torch.optim': torch_optim_mod,
                'torch.nn.utils': torch_nn_utils_mod,
                'tqdm': tqdm_mod,
                'tqdm.auto': tqdm_auto_mod,
                'ema_pytorch': ema_mod,
                'Utils': utils_pkg,
                'Utils.io_utils': io_utils_mod,
            },
        )

    def test_sample_generates_exact_requested_count(self):
        mod = self._load_solver_module()

        class FakeTensor:
            def __init__(self, batch_size):
                self._batch_size = batch_size

            def detach(self):
                return self

            def cpu(self):
                return self

            def numpy(self):
                return [self._batch_size] * self._batch_size

        class FakeModel:
            def __init__(self):
                self.batch_sizes = []

            def generate_mts(self, batch_size, model_kwargs=None, cond_fn=None):
                self.batch_sizes.append(batch_size)
                return FakeTensor(batch_size)

        trainer = mod.Trainer.__new__(mod.Trainer)
        trainer.logger = None
        trainer.ema = types.SimpleNamespace(ema_model=FakeModel())

        samples = trainer.sample(num=10, size_every=4, shape=[2, 3])

        self.assertEqual(trainer.ema.ema_model.batch_sizes, [4, 4, 2])
        self.assertEqual(len(samples), 10)
        self.assertEqual(samples[-2:], [2, 2])


if __name__ == '__main__':
    unittest.main()
