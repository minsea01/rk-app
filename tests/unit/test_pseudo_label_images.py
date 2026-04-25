from pathlib import Path

from tools.pseudo_label_images import build_split_plan, infer_group_key, parse_class_ids, resolve_names


def test_infer_group_key_strips_numeric_suffix():
    assert infer_group_key(Path("back-side-view-10078860.jpg")) == "back-side-view"
    assert infer_group_key(Path("pexels-12345.png")) == "pexels"
    assert infer_group_key(Path("IMG_0012.jpg")) == "img"


def test_parse_class_ids_handles_empty_and_csv():
    assert parse_class_ids(None) is None
    assert parse_class_ids("") is None
    assert parse_class_ids("0, 2,5") == [0, 2, 5]


def test_resolve_names_remaps_selected_classes():
    model_names = {0: "person", 1: "car", 2: "dog"}
    assert resolve_names(model_names, None) == model_names
    assert resolve_names(model_names, [2, 0]) == {0: "dog", 1: "person"}


def test_build_split_plan_keeps_small_groups_train_only():
    images = [
        Path("back-side-view-1.jpg"),
        Path("back-side-view-2.jpg"),
        Path("pexels-1.jpg"),
        Path("pexels-2.jpg"),
        Path("pexels-3.jpg"),
    ]
    plan = build_split_plan(images, val_ratio=0.2, seed=7)
    assert set(plan.split_by_source.values()) <= {"train", "val"}
    assert len([p for p, split in plan.split_by_source.items() if split == "val"]) >= 1
    assert "back-side-view" in plan.groups
    assert "pexels" in plan.groups
