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
    proc = subprocess.run(
        ["git", *args],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=not binary,
    )
    if check and proc.returncode:
        error = proc.stderr.decode(errors="replace") if binary else proc.stderr
        raise RuntimeError(f"git {' '.join(args)} failed: {error.strip()}")
    return proc.stdout


def git_bytes(ref: str, path: str) -> bytes:
    return git("show", f"{ref}:{path}", binary=True)


def git_path_exists(ref: str, path: str) -> bool:
    return (
        subprocess.run(
            ["git", "cat-file", "-e", f"{ref}:{path}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        ).returncode
        == 0
    )


def git_paths(ref: str) -> list[str]:
    return [path for path in git("ls-tree", "-r", "--name-only", ref).splitlines() if path]


def load_json_at(ref: str, path: str) -> dict[str, Any]:
    return json.loads(git("show", f"{ref}:{path}"))


def ref_reader(ref: str):
    def read(path: str) -> bytes | None:
        return git_bytes(ref, path) if git_path_exists(ref, path) else None

    return read


def worktree_reader(path: str) -> bytes | None:
    file_path = Path(path)
    return file_path.read_bytes() if file_path.is_file() else None


def path_key(value: str) -> str:
    return unicodedata.normalize("NFC", value.replace("\\", "/")).casefold()


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
        changed = sorted(
            item_id
            for item_id in set(base_items) & set(source_items)
            if base_items[item_id] != source_items[item_id]
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


def image_dimensions_payload(image: dict[str, Any]) -> tuple[Any, Any]:
    return image.get("width"), image.get("height")


def index_categories_by_name(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        name = item.get("name")
        if isinstance(name, str):
            result.setdefault(name, []).append(item)
    return result


def index_images_by_path(items: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        file_name = item.get("file_name")
        if isinstance(file_name, str):
            result.setdefault(path_key(file_name), []).append(item)
    return result


def index_paths_by_key(paths: list[str]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in paths:
        result.setdefault(path_key(path), []).append(path)
    return result


def unique_hashed_name(
    original: str,
    data: bytes,
    occupied_paths_by_key: dict[str, list[str]],
) -> str:
    original_path = Path(original)
    digest = sha256(data)
    for length in (8, 12, 16, 24, 32, 64):
        candidate = original_path.with_name(
            f"{original_path.stem}_{digest[:length]}{original_path.suffix}"
        ).as_posix()
        if path_key(candidate) not in occupied_paths_by_key:
            return candidate
    raise RuntimeError(f"Unable to derive a unique deterministic name for {original}")


def restore_target_file(target_ref: str, target_path: str) -> None:
    data = git_bytes(target_ref, target_path)
    path = Path(target_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    git("add", "--", target_path)


def target_referenced_image_ids(target: dict[str, Any]) -> set[int]:
    return {
        int(annotation["image_id"])
        for annotation in target.get("annotations", [])
        if isinstance(annotation, dict) and isinstance(annotation.get("image_id"), (int, float))
    }


def choose_reusable_image_registration(
    matches: list[dict[str, Any]],
    source_image: dict[str, Any],
    *,
    preferred_id: int,
    referenced_ids: set[int],
) -> dict[str, Any] | None:
    compatible = [
        item
        for item in matches
        if image_dimensions_payload(item) == image_dimensions_payload(source_image)
    ]
    if not compatible:
        return None
    for item in compatible:
        if int(item["id"]) == preferred_id:
            return item
    referenced = [item for item in compatible if int(item["id"]) in referenced_ids]
    if referenced:
        return min(referenced, key=lambda item: int(item["id"]))
    return min(compatible, key=lambda item: int(item["id"]))


def register_occupied_metadata_paths(
    occupied: dict[str, list[str]],
    images: list[dict[str, Any]],
) -> None:
    for image in images:
        file_name = image.get("file_name")
        if isinstance(file_name, str):
            occupied.setdefault(path_key(file_name), [])
            if file_name not in occupied[path_key(file_name)]:
                occupied[path_key(file_name)].append(file_name)


def resolve_added_source_image(
    image: dict[str, Any],
    *,
    source_ref: str,
    target_ref: str,
    result_images_by_path: dict[str, list[dict[str, Any]]],
    target_paths_by_key: dict[str, list[str]],
    occupied_paths_by_key: dict[str, list[str]],
    target_referenced_ids: set[int],
) -> tuple[str, int | None, str]:
    source_id = int(image["id"])
    source_name = str(image["file_name"])
    key = path_key(source_name)
    metadata_matches = result_images_by_path.get(key, [])
    physical_matches = target_paths_by_key.get(key, [])

    if len(physical_matches) > 1:
        raise RuntimeError(
            f"Target Git tree has multiple paths colliding with {source_name!r} after "
            f"slash/case/Unicode normalization: {physical_matches}"
        )

    # An unreferenced source image record may legitimately point at a missing
    # source file. Preserve the metadata record; the source audit guarantees
    # that effective annotations never depend on such a missing file.
    if not git_path_exists(source_ref, source_name):
        return source_name, None, "kept-metadata-only-source-image"

    source_data = git_bytes(source_ref, source_name)
    target_path = physical_matches[0] if physical_matches else None

    if target_path is None:
        # A target registration already used by a target annotation owns its
        # missing-file state. Never fill that registration with source bytes,
        # or the existing target bbox would silently start referring to source
        # image content. Only redundant/unreferenced target registrations may
        # be reused when their physical file is missing.
        unreferenced_matches = [
            item
            for item in metadata_matches
            if int(item["id"]) not in target_referenced_ids
        ]
        reusable = choose_reusable_image_registration(
            unreferenced_matches,
            image,
            preferred_id=source_id,
            referenced_ids=target_referenced_ids,
        )
        if reusable is not None:
            target_name = str(reusable["file_name"])
            path = Path(target_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source_data)
            git("add", "--", target_name)
            occupied_paths_by_key.setdefault(path_key(target_name), []).append(target_name)
            return target_name, int(reusable["id"]), "restored-target-metadata-file"

        if metadata_matches:
            # Target metadata owns this source name. If no unreferenced target
            # registration can safely absorb the source image, keep every
            # target registration and its missing-file state untouched. The
            # merge may already have materialized source_name, so remove that
            # source path before placing the source bytes under a deterministic
            # new name.
            git("rm", "-f", "--ignore-unmatch", "--", source_name, check=False)
            new_name = unique_hashed_name(source_name, source_data, occupied_paths_by_key)
            path = Path(new_name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source_data)
            git("add", "--", new_name)
            occupied_paths_by_key.setdefault(path_key(new_name), []).append(new_name)
            return new_name, None, "renamed-source-around-target-metadata"

        path = Path(source_name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source_data)
        git("add", "--", source_name)
        occupied_paths_by_key.setdefault(key, []).append(source_name)
        return source_name, None, "kept-source-name"

    target_data = git_bytes(target_ref, target_path)
    restore_target_file(target_ref, target_path)

    if source_name != target_path:
        git("rm", "-f", "--ignore-unmatch", "--", source_name, check=False)
        restore_target_file(target_ref, target_path)

    if source_data == target_data:
        reusable = choose_reusable_image_registration(
            metadata_matches,
            image,
            preferred_id=source_id,
            referenced_ids=target_referenced_ids,
        )
        if reusable is not None:
            return target_path, int(reusable["id"]), "reused-identical-target-registration"
        return target_path, None, "reused-identical-target-file"

    new_name = unique_hashed_name(source_name, source_data, occupied_paths_by_key)
    path = Path(new_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(source_data)
    git("add", "--", new_name)
    occupied_paths_by_key.setdefault(path_key(new_name), []).append(new_name)
    return new_name, None, "renamed-source-different-bytes"


def audit_or_raise_source_delta(
    base: dict[str, Any],
    source: dict[str, Any],
    *,
    base_ref: str,
    source_ref: str,
):
    base_issues = audit_coco(base, label="base", file_reader=ref_reader(base_ref))
    source_issues = audit_coco(source, label="source", file_reader=ref_reader(source_ref))
    new_source_errors = issue_fingerprints(source_issues) - issue_fingerprints(base_issues)
    if new_source_errors:
        details = [
            issue
            for issue in source_issues
            if issue.severity == "error" and issue.fingerprint in new_source_errors
        ]
        raise RuntimeError(
            "Source introduces labeling-source integrity errors relative to merge base:\n"
            + format_audit(details, title="source delta errors")
        )
    return base_issues, source_issues


def resolve_existing_category_id(
    source_category_id: int,
    *,
    source_categories: dict[int, dict[str, Any]],
    result_categories: list[dict[str, Any]],
    category_id_map: dict[int, int],
) -> int:
    if source_category_id in category_id_map:
        return category_id_map[source_category_id]

    source_category = source_categories.get(source_category_id)
    if source_category is None:
        raise RuntimeError(f"Source annotation references unknown category id {source_category_id}")
    source_name = str(source_category.get("name", ""))

    result_by_id = {int(item["id"]): item for item in result_categories}
    same_id = result_by_id.get(source_category_id)
    if same_id is not None and same_id.get("name") == source_name:
        return source_category_id

    same_name = [item for item in result_categories if item.get("name") == source_name]
    if len(same_name) == 1:
        return int(same_name[0]["id"])
    if len(same_name) > 1:
        raise RuntimeError(
            f"Target has multiple category registrations for label {source_name!r}; "
            f"cannot resolve source category id {source_category_id}"
        )

    raise RuntimeError(
        f"Target changed or removed base label {source_name!r} (source category id "
        f"{source_category_id}); source annotations depending on it cannot be imported safely"
    )


def resolve_existing_image_id(
    source_image_id: int,
    *,
    source_ref: str,
    target_ref: str,
    source_images: dict[int, dict[str, Any]],
    target_images: dict[int, dict[str, Any]],
    target_images_by_path: dict[str, list[dict[str, Any]]],
    target_referenced_ids: set[int],
) -> int:
    source_image = source_images.get(source_image_id)
    if source_image is None:
        raise RuntimeError(f"Source annotation references unknown image id {source_image_id}")

    source_name = str(source_image.get("file_name", ""))
    if not git_path_exists(source_ref, source_name):
        raise RuntimeError(
            f"Source annotation depends on missing source image {source_name!r} "
            f"(image id {source_image_id})"
        )
    source_data = git_bytes(source_ref, source_name)
    source_dims = image_dimensions_payload(source_image)

    candidates: list[dict[str, Any]] = []
    candidate_pool: list[dict[str, Any]] = []

    same_id = target_images.get(source_image_id)
    if same_id is not None:
        candidate_pool.append(same_id)
    for item in target_images_by_path.get(path_key(source_name), []):
        if item not in candidate_pool:
            candidate_pool.append(item)

    for item in candidate_pool:
        if image_dimensions_payload(item) != source_dims:
            continue
        target_name = str(item.get("file_name", ""))
        if not git_path_exists(target_ref, target_name):
            continue
        if git_bytes(target_ref, target_name) == source_data:
            candidates.append(item)

    if not candidates:
        raise RuntimeError(
            f"Target changed or removed base source image {source_name!r} "
            f"(source image id {source_image_id}); source annotation coordinates "
            "cannot be attached safely"
        )

    for item in candidates:
        if int(item["id"]) == source_image_id:
            return source_image_id
    referenced = [item for item in candidates if int(item["id"]) in target_referenced_ids]
    if referenced:
        return int(min(referenced, key=lambda item: int(item["id"]))["id"])
    return int(min(candidates, key=lambda item: int(item["id"]))["id"])


def packed_bbox_key(annotation: dict[str, Any]) -> tuple[int, int, int, int]:
    bbox = annotation.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise RuntimeError(f"Invalid bbox while resolving annotation: {bbox!r}")
    return tuple(round(float(value)) for value in bbox)  # matches compress_copy_coco


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Apply source labeling additions onto a target labeling repository while "
            "preserving target IDs, label definitions, image files, and registrations."
        )
    )
    parser.add_argument("--base-ref", required=True)
    parser.add_argument("--target-ref", required=True)
    parser.add_argument("--source-ref", required=True)
    parser.add_argument("--coco", default="coco_annotations.json")
    args = parser.parse_args()

    base = load_json_at(args.base_ref, args.coco)
    target = load_json_at(args.target_ref, args.coco)
    source = load_json_at(args.source_ref, args.coco)

    _, source_issues = audit_or_raise_source_delta(
        base,
        source,
        base_ref=args.base_ref,
        source_ref=args.source_ref,
    )
    target_issues = audit_coco(
        target,
        label="target",
        file_reader=ref_reader(args.target_ref),
    )

    fatal_target_codes = {
        "duplicate_id",
        "invalid_id",
        "section_not_list",
        "item_not_object",
        "duplicate_category_name",
        "multiple_annotations_for_label",
    }
    fatal_target = [
        issue
        for issue in target_issues
        if issue.severity == "error" and issue.code in fatal_target_codes
    ]
    if fatal_target:
        raise RuntimeError(
            "Target labeling source is too ambiguous to merge safely:\n"
            + format_audit(fatal_target, title="fatal target errors")
        )

    ensure_source_only_additions(base, source)

    base_images = by_id_checked(base.get("images", []), "base images")
    base_annotations = by_id_checked(base.get("annotations", []), "base annotations")
    base_categories = by_id_checked(base.get("categories", []), "base categories")
    source_images = by_id_checked(source.get("images", []), "source images")
    source_annotations = by_id_checked(source.get("annotations", []), "source annotations")
    source_categories = by_id_checked(source.get("categories", []), "source categories")
    target_images_by_id = by_id_checked(target.get("images", []), "target images")

    added_images = [item for item_id, item in source_images.items() if item_id not in base_images]
    added_annotations = [
        item for item_id, item in source_annotations.items() if item_id not in base_annotations
    ]
    added_categories = [
        item for item_id, item in source_categories.items() if item_id not in base_categories
    ]

    result = copy.deepcopy(target)
    result_images = result.setdefault("images", [])
    result_annotations = result.setdefault("annotations", [])
    result_categories = result.setdefault("categories", [])

    used_image_ids = {int(item["id"]) for item in result_images}
    used_annotation_ids = {int(item["id"]) for item in result_annotations}
    used_category_ids = {int(item["id"]) for item in result_categories}

    result_categories_by_name = index_categories_by_name(result_categories)
    result_images_by_path = index_images_by_path(result_images)
    target_images_by_path = index_images_by_path(target.get("images", []))
    target_paths_by_key = index_paths_by_key(git_paths(args.target_ref))
    occupied_paths_by_key = copy.deepcopy(target_paths_by_key)
    register_occupied_metadata_paths(occupied_paths_by_key, result_images)

    target_referenced_ids = target_referenced_image_ids(target)
    category_id_map: dict[int, int] = {}
    image_id_map: dict[int, int] = {}
    report: list[str] = []

    # Label identity is the exact category name. Case/Unicode-equivalent but
    # Non-identical source label names stay distinct; the merge must not guess
    # that case/Unicode-similar spellings represent the same editable label.
    for category in added_categories:
        old_id = int(category["id"])
        name = str(category["name"])
        existing_matches = result_categories_by_name.get(name, [])
        if len(existing_matches) > 1:
            raise RuntimeError(
                f"Target has multiple category registrations for label {name!r}: "
                f"{[(item.get('id'), item.get('name')) for item in existing_matches]}"
            )
        if existing_matches:
            existing = existing_matches[0]
            category_id_map[old_id] = int(existing["id"])
            report.append(
                f"label {name}: reused target category id {existing['id']} "
                "(target metadata wins)"
            )
            continue

        new_item = copy.deepcopy(category)
        new_id = old_id
        if new_id in used_category_ids:
            new_id = next_free(used_category_ids)
        else:
            used_category_ids.add(new_id)
        new_item["id"] = new_id
        result_categories.append(new_item)
        result_categories_by_name.setdefault(name, []).append(new_item)
        category_id_map[old_id] = new_id
        report.append(f"label {name}: source category {old_id} -> result {new_id}")

    for image in added_images:
        old_id = int(image["id"])
        old_name = str(image["file_name"])
        new_name, reuse_id, action = resolve_added_source_image(
            image,
            source_ref=args.source_ref,
            target_ref=args.target_ref,
            result_images_by_path=result_images_by_path,
            target_paths_by_key=target_paths_by_key,
            occupied_paths_by_key=occupied_paths_by_key,
            target_referenced_ids=target_referenced_ids,
        )

        if reuse_id is not None:
            image_id_map[old_id] = reuse_id
            report.append(
                f"image {old_name}: {action}, source id {old_id} -> target id {reuse_id}"
            )
            continue

        new_item = copy.deepcopy(image)
        new_id = old_id
        if new_id in used_image_ids:
            new_id = next_free(used_image_ids)
        else:
            used_image_ids.add(new_id)
        new_item["id"] = new_id
        new_item["file_name"] = new_name
        result_images.append(new_item)
        result_images_by_path.setdefault(path_key(new_name), []).append(new_item)
        image_id_map[old_id] = new_id
        report.append(
            f"image {old_name}: {action} as {new_name}, source id {old_id} -> result {new_id}"
        )

    result_category_by_id = {int(item["id"]): item for item in result_categories}
    target_images_by_path = index_images_by_path(target.get("images", []))

    # Existing target label definitions. Target audit guarantees at most one
    # annotation per exact label name.
    result_annotation_by_label: dict[str, dict[str, Any]] = {}
    for annotation in result_annotations:
        category = result_category_by_id.get(int(annotation["category_id"]))
        if category is None:
            continue
        name = category.get("name")
        if isinstance(name, str) and name not in result_annotation_by_label:
            result_annotation_by_label[name] = annotation

    for annotation in added_annotations:
        new_item = copy.deepcopy(annotation)
        old_annotation_id = int(annotation["id"])
        old_image_id = int(new_item["image_id"])
        old_category_id = int(new_item["category_id"])

        if old_image_id in image_id_map:
            mapped_image_id = image_id_map[old_image_id]
        else:
            mapped_image_id = resolve_existing_image_id(
                old_image_id,
                source_ref=args.source_ref,
                target_ref=args.target_ref,
                source_images=source_images,
                target_images=target_images_by_id,
                target_images_by_path=target_images_by_path,
                target_referenced_ids=target_referenced_ids,
            )

        mapped_category_id = resolve_existing_category_id(
            old_category_id,
            source_categories=source_categories,
            result_categories=result_categories,
            category_id_map=category_id_map,
        )

        new_item["image_id"] = mapped_image_id
        new_item["category_id"] = mapped_category_id
        label_name = str(result_category_by_id[mapped_category_id]["name"])

        existing = result_annotation_by_label.get(label_name)
        if existing is not None:
            if (
                int(existing["image_id"]) == mapped_image_id
                and packed_bbox_key(existing) == packed_bbox_key(new_item)
            ):
                report.append(
                    f"annotation {old_annotation_id}: skipped equivalent duplicate label "
                    f"{label_name!r} after target remap"
                )
                continue
            raise RuntimeError(
                f"Label collision for {label_name!r}: target already defines annotation "
                f"{existing.get('id')} on image {existing.get('image_id')} bbox "
                f"{existing.get('bbox')}; source annotation {old_annotation_id} would "
                f"define image {mapped_image_id} bbox {new_item.get('bbox')}. "
                "A labeling-source label may have only one effective annotation; "
                "target is preserved and the source change requires manual resolution."
            )

        new_id = old_annotation_id
        if new_id in used_annotation_ids:
            new_id = next_free(used_annotation_ids)
        else:
            used_annotation_ids.add(new_id)
        new_item["id"] = new_id
        result_annotations.append(new_item)
        result_annotation_by_label[label_name] = new_item
        report.append(
            f"annotation {old_annotation_id} -> {new_id}, image {old_image_id} -> "
            f"{mapped_image_id}, category {old_category_id} -> {mapped_category_id} "
            f"(label {label_name})"
        )

    Path(args.coco).write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    git("add", "--", args.coco)

    result_issues = audit_coco(
        result,
        label="result",
        file_reader=worktree_reader,
    )
    new_result_errors = issue_fingerprints(result_issues) - issue_fingerprints(target_issues)
    if new_result_errors:
        details = [
            issue
            for issue in result_issues
            if issue.severity == "error" and issue.fingerprint in new_result_errors
        ]
        raise RuntimeError(
            "Merge result introduces new labeling-source integrity errors compared with target:\n"
            + format_audit(details, title="new result errors")
        )

    unresolved = [
        path
        for path in git("diff", "--name-only", "--diff-filter=U").splitlines()
        if path
    ]
    if unresolved:
        raise RuntimeError(f"Unsupported unresolved merge conflicts remain: {unresolved}")

    print(format_audit(target_issues, title="Target labeling-source audit"))
    print(format_audit(source_issues, title="Source labeling-source audit"))
    print(format_audit(result_issues, title="Result labeling-source audit"))
    print("Semantic labeling-source merge completed (target priority)")
    for line in report:
        print(f"- {line}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
