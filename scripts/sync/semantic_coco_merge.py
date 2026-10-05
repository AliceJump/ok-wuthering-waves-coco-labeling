#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


def git(*args: str, check: bool = True, binary: bool = False):
    proc = subprocess.run(
        ["git", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    if check and proc.returncode:
        err = proc.stderr.decode() if binary else proc.stderr
        raise RuntimeError(f"git {' '.join(args)} failed: {err.strip()}")
    return proc.stdout


def git_bytes(ref: str, path: str) -> bytes:
    return git("show", f"{ref}:{path}", binary=True)


def git_path_exists(ref: str, path: str) -> bool:
    return subprocess.run(
        ["git", "cat-file", "-e", f"{ref}:{path}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def load_json_at(ref: str, path: str) -> dict[str, Any]:
    return json.loads(git("show", f"{ref}:{path}"))


def by_id(items: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(item["id"]): item for item in items}


def ensure_source_only_additions(base: dict[str, Any], source: dict[str, Any]) -> None:
    for section in ("images", "annotations", "categories"):
        base_items = by_id(base.get(section, []))
        source_items = by_id(source.get(section, []))
        removed = sorted(set(base_items) - set(source_items))
        changed = sorted(
            key
            for key in set(base_items) & set(source_items)
            if base_items[key] != source_items[key]
        )
        if removed or changed:
            raise RuntimeError(
                f"Unsupported source edits in {section}: removed={removed}, changed={changed}. "
                "The automatic resolver currently imports source additions only."
            )


def next_free(used: set[int]) -> int:
    candidate = max(used, default=0) + 1
    while candidate in used:
        candidate += 1
    used.add(candidate)
    return candidate


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def worktree_or_ref_bytes(ref: str, path: str) -> bytes | None:
    p = Path(path)
    if p.is_file():
        return p.read_bytes()
    if git_path_exists(ref, path):
        return git_bytes(ref, path)
    return None


def unique_hashed_name(
    original: str,
    data: bytes,
    target_ref: str,
    result_images_by_name: dict[str, dict[str, Any]],
) -> tuple[str, int | None, bool]:
    p = Path(original)
    digest = sha256(data)
    for length in (8, 12, 16, 24, 32, 64):
        candidate = p.with_name(f"{p.stem}_{digest[:length]}{p.suffix}").as_posix()
        existing_data = worktree_or_ref_bytes(target_ref, candidate)
        if existing_data is None:
            return candidate, None, True
        if existing_data == data:
            existing = result_images_by_name.get(candidate)
            return candidate, int(existing["id"]) if existing else None, False
    raise RuntimeError(f"Unable to derive a unique deterministic name for {original}")


def resolve_source_image_path(
    file_name: str,
    source_ref: str,
    target_ref: str,
    result_images_by_name: dict[str, dict[str, Any]],
) -> tuple[str, int | None, str]:
    source_data = git_bytes(source_ref, file_name)
    target_metadata = result_images_by_name.get(file_name)
    target_file_exists = git_path_exists(target_ref, file_name)

    if not target_file_exists:
        if target_metadata is not None:
            raise RuntimeError(
                f"Target COCO references missing image {file_name}; cannot safely compare "
                "it with a source image of the same name"
            )
        Path(file_name).parent.mkdir(parents=True, exist_ok=True)
        Path(file_name).write_bytes(source_data)
        git("add", "--", file_name)
        return file_name, None, "kept-source-name"

    target_data = git_bytes(target_ref, file_name)
    git("checkout", target_ref, "--", file_name, check=False)
    git("add", "--", file_name)
    if source_data == target_data:
        if target_metadata is not None:
            return file_name, int(target_metadata["id"]), "reused-identical-target"
        return file_name, None, "reused-identical-unregistered-target"

    new_name, reuse_id, needs_write = unique_hashed_name(
        file_name, source_data, target_ref, result_images_by_name
    )
    if reuse_id is not None:
        return new_name, reuse_id, "reused-existing-hash-name"
    if needs_write:
        Path(new_name).parent.mkdir(parents=True, exist_ok=True)
        Path(new_name).write_bytes(source_data)
        git("add", "--", new_name)
        return new_name, None, "renamed-source-different"
    return new_name, None, "reused-existing-file"


def invalid_references(data: dict[str, Any]) -> set[tuple[int, str, int]]:
    image_ids = {int(x["id"]) for x in data.get("images", [])}
    category_ids = {int(x["id"]) for x in data.get("categories", [])}
    invalid: set[tuple[int, str, int]] = set()
    for ann in data.get("annotations", []):
        ann_id = int(ann["id"])
        image_id = int(ann["image_id"])
        category_id = int(ann["category_id"])
        if image_id not in image_ids:
            invalid.add((ann_id, "image_id", image_id))
        if category_id not in category_ids:
            invalid.add((ann_id, "category_id", category_id))
    return invalid


def validate(data: dict[str, Any], target: dict[str, Any]) -> None:
    for section in ("images", "annotations", "categories"):
        ids = [int(x["id"]) for x in data.get(section, [])]
        if len(ids) != len(set(ids)):
            raise RuntimeError(f"Duplicate IDs found in {section}")

    target_names = Counter(x["file_name"] for x in target.get("images", []))
    result_names = Counter(x["file_name"] for x in data.get("images", []))
    increased_duplicates = {
        name: (target_names[name], count)
        for name, count in result_names.items()
        if count > max(1, target_names[name])
    }
    if increased_duplicates:
        raise RuntimeError(
            f"Merge introduced additional duplicate image file_name values: {increased_duplicates}"
        )

    target_missing = {name for name in target_names if not Path(name).is_file()}
    result_missing = {name for name in result_names if not Path(name).is_file()}
    introduced_missing = sorted(result_missing - target_missing)
    if introduced_missing:
        raise RuntimeError(
            f"Merge introduced COCO image paths missing from worktree: {introduced_missing[:20]}"
        )

    introduced_invalid_refs = sorted(invalid_references(data) - invalid_references(target))
    if introduced_invalid_refs:
        raise RuntimeError(
            f"Merge introduced invalid annotation references: {introduced_invalid_refs[:20]}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Apply source COCO additions onto a target while preserving target IDs and names."
    )
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--target-ref", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--coco", default="coco_annotations.json")
    args = parser.parse_args()

    base = load_json_at(args.base_ref, args.coco)
    target = load_json_at(args.target_ref, args.coco)
    source = load_json_at(args.source_ref, args.coco)
    ensure_source_only_additions(base, source)

    base_images = by_id(base.get("images", []))
    base_annotations = by_id(base.get("annotations", []))
    base_categories = by_id(base.get("categories", []))
    source_images = by_id(source.get("images", []))
    source_annotations = by_id(source.get("annotations", []))
    source_categories = by_id(source.get("categories", []))

    added_images = [x for k, x in source_images.items() if k not in base_images]
    added_annotations = [x for k, x in source_annotations.items() if k not in base_annotations]
    added_categories = [x for k, x in source_categories.items() if k not in base_categories]

    result = copy.deepcopy(target)
    result_images = result.setdefault("images", [])
    result_annotations = result.setdefault("annotations", [])
    result_categories = result.setdefault("categories", [])

    used_image_ids = {int(x["id"]) for x in result_images}
    used_annotation_ids = {int(x["id"]) for x in result_annotations}
    used_category_ids = {int(x["id"]) for x in result_categories}
    result_images_by_name: dict[str, dict[str, Any]] = {}
    for item in result_images:
        result_images_by_name.setdefault(item["file_name"], item)
    result_categories_by_name: dict[str, dict[str, Any]] = {}
    for item in result_categories:
        result_categories_by_name.setdefault(item["name"], item)

    category_id_map: dict[int, int] = {}
    image_id_map: dict[int, int] = {}
    report: list[str] = []

    for category in added_categories:
        old_id = int(category["id"])
        existing = result_categories_by_name.get(category["name"])
        if existing is not None:
            category_id_map[old_id] = int(existing["id"])
            report.append(f"category {category['name']}: reused target id {existing['id']}")
            continue
        new_item = copy.deepcopy(category)
        if old_id in used_category_ids:
            new_id = next_free(used_category_ids)
        else:
            new_id = old_id
            used_category_ids.add(new_id)
        new_item["id"] = new_id
        result_categories.append(new_item)
        result_categories_by_name[new_item["name"]] = new_item
        category_id_map[old_id] = new_id
        report.append(f"category {category['name']}: source {old_id} -> result {new_id}")

    for image in added_images:
        old_id = int(image["id"])
        old_name = image["file_name"]
        new_name, reuse_id, action = resolve_source_image_path(
            old_name, args.source_ref, args.target_ref, result_images_by_name
        )
        if reuse_id is not None:
            image_id_map[old_id] = reuse_id
            report.append(f"image {old_name}: {action}, source id {old_id} -> target id {reuse_id}")
            continue
        new_item = copy.deepcopy(image)
        if old_id in used_image_ids:
            new_id = next_free(used_image_ids)
        else:
            new_id = old_id
            used_image_ids.add(new_id)
        new_item["id"] = new_id
        new_item["file_name"] = new_name
        result_images.append(new_item)
        result_images_by_name.setdefault(new_name, new_item)
        image_id_map[old_id] = new_id
        report.append(
            f"image {old_name}: {action} as {new_name}, source id {old_id} -> result {new_id}"
        )

    for ann in added_annotations:
        new_item = copy.deepcopy(ann)
        old_id = int(ann["id"])
        if old_id in used_annotation_ids:
            new_id = next_free(used_annotation_ids)
        else:
            new_id = old_id
            used_annotation_ids.add(new_id)
        new_item["id"] = new_id
        old_image_id = int(new_item["image_id"])
        old_category_id = int(new_item["category_id"])
        new_item["image_id"] = image_id_map.get(old_image_id, old_image_id)
        new_item["category_id"] = category_id_map.get(old_category_id, old_category_id)
        result_annotations.append(new_item)
        report.append(
            f"annotation {old_id} -> {new_id}, image {old_image_id} -> {new_item['image_id']}, "
            f"category {old_category_id} -> {new_item['category_id']}"
        )

    Path(args.coco).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    git("add", "--", args.coco)
    validate(result, target)

    unresolved = [x for x in git("diff", "--name-only", "--diff-filter=U").splitlines() if x]
    if unresolved:
        raise RuntimeError(f"Unsupported unresolved merge conflicts remain: {unresolved}")

    print("Semantic COCO merge completed (target priority)")
    for line in report:
        print(f"- {line}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
