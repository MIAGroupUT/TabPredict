"""
Public API smoke tests — verify every symbol advertised in __all__ is importable.
If an __init__.py export breaks (typo, deleted file, circular import) this catches it.
"""

from __future__ import annotations


class TestModelsPackage:
    def test_model_class(self):
        from src.models import ReinterventionRiskModel
        assert callable(ReinterventionRiskModel)

    def test_helpers(self):
        from src.models import load_checkpoint, get_model, disable_checkpointing
        assert callable(load_checkpoint)
        assert callable(get_model)
        assert callable(disable_checkpointing)

    def test_constants(self):
        from src.models import COMPLICATIONS_NO_COMP, COMPLICATIONS_FULL, AAA_METRIC_NAMES
        assert COMPLICATIONS_NO_COMP == []
        assert len(COMPLICATIONS_FULL) == 7
        assert len(AAA_METRIC_NAMES) == 5


class TestDataPackage:
    def test_io_functions(self):
        from src.data import load_splits_and_config, compute_clin_stats, build_data_list
        assert callable(load_splits_and_config)
        assert callable(compute_clin_stats)
        assert callable(build_data_list)

    def test_loader_functions(self):
        from src.data import build_loaders
        assert callable(build_loaders)

    def test_transforms(self):
        from src.data import (
            get_cached_transforms,
            get_train_runtime_transforms,
            get_val_runtime_transforms,
        )
        assert callable(get_cached_transforms)
        assert callable(get_train_runtime_transforms)
        assert callable(get_val_runtime_transforms)

    def test_classes(self):
        from src.data import BuildCTAd, NormalizeClinicald, RuntimeDataset
        assert callable(BuildCTAd)
        assert callable(NormalizeClinicald)
        assert callable(RuntimeDataset)


class TestTrainingPackage:
    def test_losses(self):
        from src.training import ReinterventionLoss, MultiTaskReinterventionLoss
        assert callable(ReinterventionLoss)
        assert callable(MultiTaskReinterventionLoss)

    def test_stage_functions(self):
        from src.training import (
            set_all_seeds, set_stage_a, set_stage_b, build_optimizer,
            causal_modality_dropout, maybe_apply_modality_dropout,
            count_trainable, print_model_param_overview,
        )
        for fn in (set_all_seeds, set_stage_a, set_stage_b, build_optimizer,
                   causal_modality_dropout, maybe_apply_modality_dropout,
                   count_trainable, print_model_param_overview):
            assert callable(fn)

    def test_pipeline(self):
        from src.training import run_pipeline, validate
        assert callable(run_pipeline)
        assert callable(validate)


class TestXAIPackage:
    def test_xai_class(self):
        from src.xai import ReinterventionXAI
        assert callable(ReinterventionXAI)


class TestEvaluationPackage:
    def test_metrics(self):
        from src.evaluation import compute_overall_metrics, save_metrics, label_type, format_xai_report
        for fn in (compute_overall_metrics, save_metrics, label_type, format_xai_report):
            assert callable(fn)

    def test_runner(self):
        from src.evaluation import run_test
        assert callable(run_test)
