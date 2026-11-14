from conftest import ROOT

from crc_tissuenet.config import apply_overrides, deep_merge, get, load_config, parse_value


def test_inheritance_chain():
    cfg = load_config(ROOT / "configs/models/p5_deit_small.yaml")
    assert cfg["model"]["name"] == "deit_small"
    assert get(cfg, "train.finetune.layer_decay") == 0.75       # transformer_base
    assert get(cfg, "data.size") == 224                          # pretrained_base
    assert get(cfg, "eval.test") is False                        # default


def test_overrides_and_parsing():
    assert parse_value("1e-3") == 1e-3
    assert parse_value("[1, 2]") == [1, 2]
    assert parse_value("relu") == "relu"
    assert parse_value("true") is True
    cfg = apply_overrides({"a": {"b": 1}}, ["a.b=2", "a.c.d=x"])
    assert cfg == {"a": {"b": 2, "c": {"d": "x"}}}


def test_deep_merge_does_not_mutate():
    base = {"a": {"b": 1}}
    out = deep_merge(base, {"a": {"c": 2}})
    assert base == {"a": {"b": 1}} and out == {"a": {"b": 1, "c": 2}}


def test_all_configs_load():
    for p in (ROOT / "configs/models").glob("*.yaml"):
        load_config(p)


def test_plan_extends_only_stages_and_extra():
    from crc_tissuenet.ablation import load_plan

    cnn = load_plan(ROOT / "configs/ablation/cnn_ablation_cpu.yaml")
    assert len(cnn["stages"]) == 18 and cnn["common_overrides"]["data.subset_fraction"] == 0.2
    assert cnn["common_overrides"]["train.save_checkpoint"] == "none"        # inherited from parent
    assert cnn["best_config_out"].endswith("b1_cnn_tuned_cpu.yaml")
    tr = load_plan(ROOT / "configs/ablation/transfer_ablation_cpu.yaml")
    assert [s["name"] for s in tr["stages"]] == ["finetune_strategy", "learning_rate", "augmentation",
                                                 "label_smoothing", "input_resolution"]
    assert tr["base"].endswith("l1_resnet18_pt.yaml")


def test_capstone_plans_and_suite():
    from crc_tissuenet.ablation import load_plan
    from crc_tissuenet.suite import load_suite

    cnn = load_plan(ROOT / "configs/ablation/capstone_cnn.yaml")
    assert [s["name"] for s in cnn["stages"]] == ["optimizer", "learning_rate", "batch_size", "activation",
                                                  "filter_size", "dropout"]
    assert sum(len(s["candidates"]) for s in cnn["stages"]) - len(cnn["stages"]) + 1 == 12   # unique runs (stage 1 has no repeat of the starting setting)
    tr = load_plan(ROOT / "configs/ablation/capstone_transfer.yaml")
    assert [s["name"] for s in tr["stages"]] == ["finetune_strategy", "learning_rate"]
    suite = load_suite(ROOT / "configs/suites/capstone_benchmark.yaml")
    assert [r["id"] for r in suite["runs"]][:4] == ["CNN_baseline_default", "CNN_baseline_tuned",
                                                "MobileNetV3_small_pretrained", "EfficientNet_B0_pretrained"]
    for f in ("l3_mobilenetv3_small_pt", "l4_efficientnet_b0_pt"):
        load_config(ROOT / f"configs/models/{f}.yaml")
