#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unicodedata
from typing import Any

from coco_audit import audit_coco, format_audit, issue_fingerprints


def git(*args: str, check: bool = True, binary: bool = False):
    proc = subprocess.run(["git", *args], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=not binary)
    if check and proc.returncode:
        err = proc.stderr.decode(errors="replace") if binary else proc.stderr
        raise RuntimeError(f"git {' '.join(args)} failed: {err.strip()}")
    return proc.stdout


def git_bytes(ref: str, path: str) -> bytes:
    return git("show", f"{ref}:{path}", binary=True)


def git_path_exists(ref: str, path: str) -> bool:
    return subprocess.run(["git", "cat-file", "-e", f"{ref}:{path}"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def git_paths(ref: str) -> list[str]:
    return [x for x in git("ls-tree", "-r", "--name-only", ref).splitlines() if x]


def load_json_at(ref: str, path: str) -> dict[str, Any]:
    return json.loads(git("show", f"{ref}:{path}"))


def ref_reader(ref: str):
    def read(path: str) -> bytes | None:
        return git_bytes(ref, path) if git_path_exists(ref, path) else None
    return read


def worktree_reader(path: str) -> bytes | None:
    p = Path(path)
    return p.read_bytes() if p.is_file() else None


def norm_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def norm_path(value: str) -> str:
    return norm_text(value.replace("\\", "/"))


def by_id_checked(items: list[dict[str, Any]], section: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for item in items:
        item_id = int(item["id"])
        if item_id in result:
            raise RuntimeError(f"Ambiguous {section}: duplicate id {item_id}")
        result[item_id] = item
    return result


def ensure_source_only_additions(base: dict[str, Any], source: dict[str, Any]) -> None:
    for section in ("images", "annotations", "categories"):
        base_items = by_id_checked(base.get(section, []), f"base {section}")
        source_items = by_id_checked(source.get(section, []), f"source {section}")
        removed = sorted(set(base_items) - set(source_items))
        changed = sorted(key for key in set(base_items) & set(source_items) if base_items[key] != source_items[key])
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


def category_payload_without_identity(category: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(category)
    result.pop("id", None)
    result.pop("name", None)
    return result


def image_payload_without_identity(image: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(image)
    result.pop("id", None)
    result.pop("file_name", None)
    return result


def annotation_payload_without_id(annotation: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(annotation)
    result.pop("id", None)
    return result


def index_by_normalized_name(items: list[dict[str, Any]], field: str, normalizer) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        value = item.get(field)
        if isinstance(value, str):
            result.setdefault(normalizer(value), []).append(item)
    return result


def index_paths_by_norm(paths: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in paths:
        result.setdefault(norm_path(path), []).append(path)
    return result


def unique_hashed_name(original: str, data: bytes, occupied_paths_by_norm: dict[str, list[str]]) -> str:
    p = Path(original)
    digest = sha256(data)
    for length in (8, 12, 16, 24, 32, 64):
        candidate = p.with_name(f"{p.stem}_{digest[:length]}{p.suffix}").as_posix()
        if norm_path(candidate) not in occupied_paths_by_norm:
            return candidate
    raise RuntimeError(f"Unable to derive a unique deterministic name for {original}")


def restore_target_file(target_ref: str, target_path: str) -> None:
    data = git_bytes(target_ref, target_path)
    p = Path(target_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    git("add", "--", target_path)


def resolve_source_image_path(
    image: dict[str, Any],
    *,
    source_ref: str,
    target_ref: str,
    result_images_by_norm: dict[str, list[dict[str, Any]]],
    target_paths_by_norm: dict[str, list[str]],
    occupied_paths_by_norm: dict[str, list[str]],
) -> tuple[str, int | None, str]:
    source_name = str(image["file_name"])
    source_data = git_bytes(source_ref, source_name)
    key = norm_path(source_name)
    metadata_matches = result_images_by_norm.get(key, [])
    physical_matches = target_paths_by_norm.get(key, [])
    if len(metadata_matches) > 1:
        raise RuntimeError(f"Target has ambiguous normalized image name {source_name!r}; matching image ids={[x.get('id') for x in metadata_matches]}")
    if len(physical_matches) > 1:
        raise RuntimeError(f"Target Git tree has multiple paths colliding with {source_name!r} after normalization: {physical_matches}")
    target_metadata = metadata_matches[0] if metadata_matches else None
    target_path = physical_matches[0] if physical_matches else None
    if target_metadata is not None and target_path is None:
        raise RuntimeError(f"Target COCO registers {target_metadata['file_name']!r} but no matching file exists in target Git tree")
    if target_path is None:
        p = Path(source_name)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(source_data)
        git("add", "--", source_name)
        occupied_paths_by_norm.setdefault(key, []).append(source_name)
        return source_name, None, "kept-source-name"
    target_data = git_bytes(target_ref, target_path)
    restore_target_file(target_ref, target_path)
    if source_name != target_path:
        git("rm", "-f", "--ignore-unmatch", "--", source_name, check=False)
        restore_target_file(target_ref, target_path)
    if source_data == target_data:
        if target_metadata is not None:
            if image_payload_without_identity(image) != image_payload_without_identity(target_metadata):
                raise RuntimeError(
                    f"Image {source_name!r} has identical bytes to target {target_path!r} but conflicting COCO metadata: "
                    f"source={image_payload_without_identity(image)}, target={image_payload_without_identity(target_metadata)}"
                )
            return target_path, int(target_metadata["id"]), "reused-identical-target"
        return target_path, None, "reused-identical-unregistered-target"
    new_name = unique_hashed_name(source_name, source_data, occupied_paths_by_norm)
    p = Path(new_name)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(source_data)
    git("add", "--", new_name)
    occupied_paths_by_norm.setdefault(norm_path(new_name), []).append(new_name)
    return new_name, None, "renamed-source-different"


def audit_or_raise_source_delta(base: dict[str, Any], source: dict[str, Any], *, base_ref: str, source_ref: str):
    base_issues = audit_coco(base, label="base", file_reader=ref_reader(base_ref))
    source_issues = audit_coco(source, label="source", file_reader=ref_reader(source_ref))
    new_source_errors = issue_fingerprints(source_issues) - issue_fingerprints(base_issues)
    if new_source_errors:
        details = [x for x in source_issues if x.severity == "error" and x.fingerprint in new_source_errors]
        raise RuntimeError("Source introduces integrity errors relative to merge base:\n" + format_audit(details, title="source delta errors"))
    return base_issues, source_issues


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply source COCO additions onto a target while preserving target IDs, names, and files.")
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--target-ref", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--coco", default="coco_annotations.json")
    args = parser.parse_args()

    base = load_json_at(args.base_ref, args.coco)
    target = load_json_at(args.target_ref, args.coco)
    source = load_json_at(args.source_ref, args.coco)
    _, source_issues = audit_or_raise_source_delta(base, source, base_ref=args.base_ref, source_ref=args.source_ref)
    target_issues = audit_coco(target, label="target", file_reader=ref_reader(args.target_ref))
    fatal_target_codes = {"duplicate_id", "invalid_id", "section_not_list", "item_not_object"}
    fatal_target = [x for x in target_issues if x.severity == "error" and x.code in fatal_target_codes]
    if fatal_target:
        raise RuntimeError("Target is too ambiguous to merge safely:\n" + format_audit(fatal_target, title="fatal target errors"))
    ensure_source_only_additions(base, source)

    base_images = by_id_checked(base.get("images", []), "base images")
    base_annotations = by_id_checked(base.get("annotations", []), "base annotations")
    base_categories = by_id_checked(base.get("categories", []), "base categories")
    source_images = by_id_checked(source.get("images", []), "source images")
    source_annotations = by_id_checked(source.get("annotations", []), "source annotations")
    source_categories = by_id_checked(source.get("categories", []), "source categories")
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
    result_categories_by_norm = index_by_normalized_name(result_categories, "name", norm_text)
    result_images_by_norm = index_by_normalized_name(result_images, "file_name", norm_path)
    target_paths_by_norm = index_paths_by_norm(git_paths(args.target_ref))
    occupied_paths_by_norm = copy.deepcopy(target_paths_by_norm)
    category_id_map: dict[int, int] = {}
    image_id_map: dict[int, int] = {}
    report: list[str] = []

    for category in added_categories:
        old_id = int(category["id"])
        key = norm_text(str(category["name"]))
        existing_matches = result_categories_by_norm.get(key, [])
        if len(existing_matches) > 1:
            raise RuntimeError(f"Target has multiple categories colliding with source name {category['name']!r} after normalization: {[(x.get('id'), x.get('name')) for x in existing_matches]}")
        if existing_matches:
            existing = existing_matches[0]
            if category_payload_without_identity(category) != category_payload_without_identity(existing):
                raise RuntimeError(
                    f"Category name collision is semantically ambiguous for {category['name']!r}: "
                    f"source={category_payload_without_identity(category)}, target={category_payload_without_identity(existing)}"
                )
            category_id_map[old_id] = int(existing["id"])
            report.append(f"category {category['name']}: reused target {existing['name']} id {existing['id']}")
            continue
        new_item = copy.deepcopy(category)
        new_id = next_free(used_category_ids) if old_id in used_category_ids else old_id
        used_category_ids.add(new_id)
        new_item["id"] = new_id
        result_categories.append(new_item)
        result_categories_by_norm.setdefault(key, []).append(new_item)
        category_id_map[old_id] = new_id
        report.append(f"category {category['name']}: source {old_id} -> result {new_id}")

    for image in added_images:
        old_id = int(image["id"])
        old_name = str(image["file_name"])
        new_name, reuse_id, action = resolve_source_image_path(
            image,
            source_ref=args.source_ref,
            target_ref=args.target_ref,
            result_images_by_norm=result_images_by_norm,
            target_paths_by_norm=target_paths_by_norm,
            occupied_paths_by_norm=occupied_paths_by_norm,
        )
        if reuse_id is not None:
            image_id_map[old_id] = reuse_id
            report.append(f"image {old_name}: {action}, source id {old_id} -> target id {reuse_id}")
            continue
        new_item = copy.deepcopy(image)
        new_id = next_free(used_image_ids) if old_id in used_image_ids else old_id
        used_image_ids.add(new_id)
        new_item["id"] = new_id
        new_item["file_name"] = new_name
        result_images.append(new_item)
        result_images_by_norm.setdefault(norm_path(new_name), []).append(new_item)
        image_id_map[old_id] = new_id
        report.append(f"image {old_name}: {action} as {new_name}, source id {old_id} -> result {new_id}")

    existing_annotation_payloads = {
        json.dumps(annotation_payload_without_id(x), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        for x in result_annotations
    }
    for ann in added_annotations:
        new_item = copy.deepcopy(ann)
        old_id = int(ann["id"])
        old_image_id = int(new_item["image_id"])
        old_category_id = int(new_item["category_id"])
        new_item["image_id"] = image_id_map.get(old_image_id, old_image_id)
        new_item["category_id"] = category_id_map.get(old_category_id, old_category_id)
        payload = json.dumps(annotation_payload_without_id(new_item), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if payload in existing_annotation_payloads:
            report.append(f"annotation {old_id}: skipped exact duplicate after remap (image {new_item['image_id']}, category {new_item['category_id']})")
            continue
        new_id = next_free(used_annotation_ids) if old_id in used_annotation_ids else old_id
        used_annotation_ids.add(new_id)
        new_item["id"] = new_id
        result_annotations.append(new_item)
        existing_annotation_payloads.add(payload)
        report.append(f"annotation {old_id} -> {new_id}, image {old_image_id} -> {new_item['image_id']}, category {old_category_id} -> {new_item['category_id']}")

    Path(args.coco).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    git("add", "--", args.coco)
    result_issues = audit_coco(result, label="result", file_reader=worktree_reader)
    new_result_errors = issue_fingerprints(result_issues) - issue_fingerprints(target_issues)
    if new_result_errors:
        details = [x for x in result_issues if x.severity == "error" and x.fingerprint in new_result_errors]
        raise RuntimeError("Merge result introduces new integrity errors compared with target:\n" + format_audit(details, title="new result errors"))
    unresolved = [x for x in git("diff", "--name-only", "--diff-filter=U").splitlines() if x]
    if unresolved:
        raise RuntimeError(f"Unsupported unresolved merge conflicts remain: {unresolved}")

    print(format_audit(target_issues, title="Target pre-merge audit"))
    print(format_audit(source_issues, title="Source pre-merge audit"))
    print(format_audit(result_issues, title="Result post-merge audit"))
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
