"""CPU checks using the installed ColQwen2_5 class with a tiny random config."""

from pathlib import Path
import copy
import tempfile
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "programs"))
from rc_prellm_m_adapter_v1 import (  # noqa: E402
    QualityResidualAdapter,
    cache_merged_visual_tokens,
    capture_frozen_hidden,
    conditioned_colnomic_forward,
    project_cached_hidden,
    _load_frozen_colnomic_weights,
)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(9124)
        torch.set_num_threads(1)

    def test_zero_init_exact_and_backward_through_frozen_downstream(self):
        x = torch.randn(5, 32)
        adapter = QualityResidualAdapter(32, 8)
        self.assertTrue(torch.equal(adapter(x, 0.03), x))
        downstream = torch.nn.Sequential(torch.nn.Linear(32, 11), torch.nn.Tanh()).requires_grad_(False)
        before = {name: value.detach().clone() for name, value in downstream.named_parameters()}
        loss = downstream(adapter(x, 0.03)).square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(adapter.up.weight.grad).all())
        self.assertGreater(float(adapter.up.weight.grad.norm()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in downstream.parameters()))
        optimizer = torch.optim.SGD(adapter.parameters(), lr=0.5)
        optimizer.step()
        optimizer.zero_grad()
        downstream(adapter(x, 0.03)).square().mean().backward()
        self.assertGreater(float(adapter.down.weight.grad.norm()), 0)
        self.assertTrue(all(torch.equal(value, before[name]) for name, value in downstream.named_parameters()))

    def test_learned_residual_uses_content_and_mass_and_constant_control(self):
        adapter = QualityResidualAdapter(32, 8, mass_log_mean=-3.0, mass_log_std=0.8)
        torch.nn.init.normal_(adapter.up.weight, std=0.05)
        x = torch.randn(5, 32)
        a, b = adapter(x, 0.01), adapter(x, 0.2)
        self.assertFalse(torch.equal(a, b))
        self.assertFalse(torch.allclose(a[0] - x[0], a[1] - x[1]))
        constant = QualityResidualAdapter(32, 8, conditioning="constant")
        constant.load_state_dict(adapter.state_dict())
        self.assertTrue(torch.equal(constant(x, 0.01), constant(x, 0.2)))
        self.assertEqual(sum(p.numel() for p in adapter.parameters()), sum(p.numel() for p in constant.parameters()))

    def test_validation_teacher_detach_and_bfloat16_identity(self):
        adapter = QualityResidualAdapter(32, 8)
        x = torch.randn(5, 32).bfloat16()
        teacher = torch.tensor(0.1, requires_grad=True)
        self.assertTrue(torch.equal(adapter(x, teacher), x))
        adapter(x, teacher).float().square().mean().backward()
        self.assertIsNone(teacher.grad)
        for invalid in (-0.1, 1.1, float("nan"), float("inf"), [0.1, 0.2]):
            with self.assertRaises(ValueError):
                adapter(x, invalid)


class ActualColQwenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from colpali_engine.models import ColQwen2_5
        from transformers import Qwen2_5_VLConfig

        torch.set_num_threads(1)
        torch.manual_seed(2912)
        config = Qwen2_5_VLConfig(
            image_token_id=1, video_token_id=2, vision_start_token_id=3, vision_end_token_id=4,
            text_config={
                "vocab_size": 32, "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
                "bos_token_id": 5, "eos_token_id": 6, "pad_token_id": 0,
                "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [1, 1, 2]},
            },
            vision_config={
                "depth": 1, "hidden_size": 32, "intermediate_size": 64, "num_heads": 4,
                "patch_size": 2, "temporal_patch_size": 2, "spatial_merge_size": 2,
                "window_size": 8, "out_hidden_size": 32, "fullatt_block_indexes": [0],
            },
        )
        config._attn_implementation = "eager"
        cls.model = ColQwen2_5(config).eval().requires_grad_(False)
        cls.batch = {
            "input_ids": torch.tensor([[0, 5, 3, 1, 1, 1, 1, 4, 7]]),
            "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1, 1]]),
            "mm_token_type_ids": torch.tensor([[0, 0, 0, 1, 1, 1, 1, 0, 0]]),
            "image_grid_thw": torch.tensor([[1, 4, 4]]),
            "pixel_values": torch.randn(1, 16, 24),
        }
        with torch.no_grad():
            cls.original = cls.model(**cls.batch)
        cls.visual = cache_merged_visual_tokens(cls.model, cls.batch)
        cls.hidden = capture_frozen_hidden(cls.model, cls.batch, cls.visual)

    def setUp(self):
        torch.manual_seed(123)
        self.model.mask_non_image_embeddings = False

    def adapter(self):
        return QualityResidualAdapter(32, 8, mass_log_mean=-3.0, mass_log_std=1.0)

    def test_cache_and_zero_adapter_match_original_exactly(self):
        self.assertFalse(self.visual.requires_grad)
        self.assertFalse(self.hidden.requires_grad)
        adapter = self.adapter()
        for location in ("prellm", "postllm"):
            actual = conditioned_colnomic_forward(self.model, self.batch, adapter, 0.04, location=location, cached_visual=self.visual)
            self.assertTrue(torch.equal(actual, self.original), location)
        projected = project_cached_hidden(self.model, self.hidden, self.batch, adapter, 0.04)
        self.assertTrue(torch.equal(projected, self.original))

    def test_frozen_backbone_gradient_through_language_model(self):
        adapter = self.adapter()
        output = conditioned_colnomic_forward(self.model, self.batch, adapter, 0.04, cached_visual=self.visual)
        weights = torch.randn_like(output)
        (output * weights).sum().backward()
        self.assertTrue(torch.isfinite(adapter.up.weight.grad).all())
        self.assertGreater(float(adapter.up.weight.grad.norm()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_nonzero_postllm_cache_equals_full_forward_and_protects_text(self):
        adapter = self.adapter()
        torch.nn.init.normal_(adapter.up.weight, std=0.1)
        cached = project_cached_hidden(self.model, self.hidden, self.batch, adapter, 0.04)
        live = conditioned_colnomic_forward(self.model, self.batch, adapter, 0.04, location="postllm", cached_visual=self.visual)
        self.assertTrue(torch.equal(cached, live))
        image_mask = self.batch["input_ids"].eq(self.model.config.image_token_id)
        self.assertTrue(torch.equal(cached[~image_mask], self.original[~image_mask]))
        self.assertFalse(torch.equal(cached[image_mask], self.original[image_mask]))
        cached.square().sum().backward()
        self.assertTrue(torch.isfinite(adapter.up.weight.grad).all())

    def test_prellm_preserves_text_inputs_positions_masks_and_batch(self):
        observed = []

        def capture(module, args, kwargs):
            observed.append({key: value.detach().clone() for key, value in kwargs.items() if isinstance(value, torch.Tensor)})

        before = {key: value.clone() for key, value in self.batch.items()}
        handle = self.model.language_model.register_forward_pre_hook(capture, with_kwargs=True)
        adapter = self.adapter()
        torch.nn.init.normal_(adapter.up.weight, std=0.1)
        try:
            conditioned_colnomic_forward(self.model, self.batch, None, 0.04, cached_visual=self.visual)
            conditioned_colnomic_forward(self.model, self.batch, adapter, 0.04, cached_visual=self.visual)
        finally:
            handle.remove()
        self.assertEqual(len(observed), 2)
        for key in ("position_ids", "attention_mask"):
            self.assertTrue(torch.equal(observed[0][key], observed[1][key]), key)
        image_mask = self.batch["input_ids"].eq(self.model.config.image_token_id)
        self.assertTrue(torch.equal(observed[0]["inputs_embeds"][~image_mask], observed[1]["inputs_embeds"][~image_mask]))
        self.assertFalse(torch.equal(observed[0]["inputs_embeds"][image_mask], observed[1]["inputs_embeds"][image_mask]))
        for key in self.batch:
            self.assertTrue(torch.equal(self.batch[key], before[key]), key)

    def test_image_only_flag_and_post_hook_removal_on_failure(self):
        self.model.mask_non_image_embeddings = True
        try:
            with torch.no_grad():
                native = self.model(**self.batch)
            projected = project_cached_hidden(self.model, self.hidden, self.batch, None, 0.04)
            self.assertTrue(torch.equal(native, projected))
            adapted = conditioned_colnomic_forward(self.model, self.batch, self.adapter(), 0.04, cached_visual=self.visual)
            self.assertTrue(torch.equal(native, adapted))
        finally:
            self.model.mask_non_image_embeddings = False
        before = len(self.model.custom_text_proj._forward_pre_hooks)
        with self.assertRaises(ValueError):
            conditioned_colnomic_forward(self.model, self.batch, self.adapter(), float("nan"), location="postllm", cached_visual=self.visual)
        self.assertEqual(len(self.model.custom_text_proj._forward_pre_hooks), before)

    def test_no_visual_recompute_with_caches_and_invalid_cache_rejected(self):
        calls = []
        handle = self.model.visual.register_forward_pre_hook(lambda *args: calls.append(1))
        try:
            for location in ("prellm", "postllm"):
                conditioned_colnomic_forward(self.model, self.batch, self.adapter(), 0.04, location=location, cached_visual=self.visual)
            project_cached_hidden(self.model, self.hidden, self.batch, self.adapter(), 0.04)
        finally:
            handle.remove()
        self.assertEqual(calls, [])
        with self.assertRaises(ValueError):
            conditioned_colnomic_forward(self.model, self.batch, self.adapter(), 0.04, cached_visual=self.visual[:-1])
        with self.assertRaises(ValueError):
            conditioned_colnomic_forward(self.model, self.batch, self.adapter(), [0.03, 0.04], cached_visual=self.visual)

    def test_actual_peft_lora_projection_and_bfloat16_grad_replay(self):
        from peft import LoraConfig, get_peft_model

        model = copy.deepcopy(self.model).to(dtype=torch.bfloat16)
        # Same target regex as the installed ColNomic adapter_config.json.
        model = get_peft_model(model, LoraConfig(
            r=2, lora_alpha=4, lora_dropout=0.1,
            target_modules=r"(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)",
        ), autocast_adapter_dtype=False)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(std=0.03)
        model.eval().requires_grad_(False)
        self.assertTrue(any("lora_" in name for name, _ in model.custom_text_proj.named_parameters()))
        visual = cache_merged_visual_tokens(model, self.batch)
        hidden = capture_frozen_hidden(model, self.batch, visual)
        adapter = self.adapter()
        with torch.no_grad():
            native = model(**self.batch)
            replay_nograd = conditioned_colnomic_forward(model, self.batch, adapter, 0.04, cached_visual=visual)
        replay_grad = conditioned_colnomic_forward(model, self.batch, adapter, 0.04, cached_visual=visual)
        self.assertTrue(torch.equal(native, replay_nograd))
        self.assertTrue(torch.equal(replay_nograd, replay_grad))
        self.assertTrue(torch.equal(native, project_cached_hidden(model, hidden, self.batch, adapter, 0.04)))
        (replay_grad.float() * torch.randn_like(replay_grad.float())).sum().backward()
        self.assertGreater(float(adapter.up.weight.grad.norm()), 0)
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_strict_loader_original_checkpoint_layout_and_missing_weight_fail(self):
        from peft import LoraConfig, get_peft_model
        from peft.utils.save_and_load import get_peft_model_state_dict
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory(prefix="rc-colnomic-loader-") as temp:
            base_dir, adapter_dir = Path(temp) / "base", Path(temp) / "adapter"
            base_dir.mkdir(); adapter_dir.mkdir()
            base = copy.deepcopy(self.model).to(dtype=torch.bfloat16)
            base.config.save_pretrained(base_dir)
            legacy_base = {name.replace("language_model.", "model.", 1): tensor.detach().clone()
                           for name, tensor in base.state_dict().items()}
            legacy_base["lm_head.weight"] = torch.randn(32, 32).bfloat16()
            save_file(legacy_base, base_dir / "model.safetensors")
            config = LoraConfig(r=2, lora_alpha=4, lora_dropout=0.1,
                target_modules=r"(.*(model).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$|.*(custom_text_proj).*$)")
            reference = get_peft_model(base, config, autocast_adapter_dtype=False)
            reference.peft_config["default"].base_model_name_or_path = str(base_dir)
            reference.peft_config["default"].save_pretrained(adapter_dir)
            with torch.no_grad():
                for name, parameter in reference.named_parameters():
                    if "lora_B" in name:
                        parameter.normal_(std=0.03)
            reference.eval().requires_grad_(False)
            expected = reference(**self.batch)
            legacy_adapter = {name.replace("base_model.model.language_model.", "base_model.model.model.", 1): tensor.float()
                              for name, tensor in get_peft_model_state_dict(reference).items()}
            save_file(legacy_adapter, adapter_dir / "adapter_model.safetensors")
            loaded, report = _load_frozen_colnomic_weights(adapter_dir, "cpu", attention="eager")
            self.assertEqual(report["base_missing_keys"], [])
            self.assertEqual(report["base_unused_keys"], ["lm_head.weight"])
            self.assertEqual(report["adapter_dtype"], ["torch.bfloat16"])
            self.assertTrue(torch.equal(loaded(**self.batch), expected))
            # Missing a real embedding weight must never produce a random model.
            del legacy_base["model.embed_tokens.weight"]
            save_file(legacy_base, base_dir / "model.safetensors")
            with self.assertRaisesRegex(ValueError, "incomplete base"):
                _load_frozen_colnomic_weights(adapter_dir, "cpu", attention="eager")


if __name__ == "__main__":
    unittest.main(verbosity=2)
