#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from pathlib import PurePosixPath
import struct
import unicodedata
from typing import Any, Callable, Iterable


@dataclass(frozen=True, order=True)
class AuditIssue:
    severity: str
    code: str
    subject: str
    message: str

    @property
    def fingerprint(self) -> tuple[str, str]:
        return (self.code, self.subject)


def _norm_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _norm_path(value: str) -> str:
    return _norm_text(value.replace("\\", "/"))


def _safe_relative_path(value: str) -> bool:
    if not value or value.startswith(("/", "\\")):
        return False
    path = PurePosixPath(value.replace("\\", "/"))
    return not path.is_absolute() and ".." not in path.parts


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _as_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    return None


def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    offset = 2
    sof = {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}
    while offset + 4 <= len(data):
        if data[offset] != 0xFF:
            offset += 1
            continue
        while offset < len(data) and data[offset] == 0xFF:
            offset += 1
        if offset >= len(data):
            break
        marker = data[offset]
        offset += 1
        if marker in {0xD8, 0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if offset + 2 > len(data):
            break
        segment_length = int.from_bytes(data[offset:offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(data):
            break
        if marker in sof and segment_length >= 7:
            height = int.from_bytes(data[offset + 3:offset + 5], "big")
            width = int.from_bytes(data[offset + 5:offset + 7], "big")
            return width, height
        offset += segment_length
    return None


def _webp_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None
    kind = data[12:16]
    if kind == b"VP8X" and len(data) >= 30:
        return 1 + int.from_bytes(data[24:27], "little"), 1 + int.from_bytes(data[27:30], "little")
    if kind == b"VP8 " and len(data) >= 30 and data[23:26] == b"\x9d\x01\x2a":
        return int.from_bytes(data[26:28], "little") & 0x3FFF, int.from_bytes(data[28:30], "little") & 0x3FFF
    if kind == b"VP8L" and len(data) >= 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    return None


def image_dimensions(data: bytes) -> tuple[int, int] | None:
    return _png_dimensions(data) or _jpeg_dimensions(data) or _webp_dimensions(data)


def audit_coco(
    data: dict[str, Any],
    *,
    label: str,
    file_reader: Callable[[str], bytes | None] | None = None,
) -> list[AuditIssue]:
    """Audit the editable labeling-source COCO, not the packed runtime COCO.

    The source repository is intentionally allowed to contain redundant image
    registrations. Only images that are actually referenced by annotations are
    required to exist and have correct dimensions. A label/category name is the
    source-level feature identity and must resolve to exactly one annotation.
    """

    issues: list[AuditIssue] = []

    def add(severity: str, code: str, subject: str, message: str) -> None:
        issues.append(AuditIssue(severity, code, subject, message))

    if not isinstance(data, dict):
        return [AuditIssue("error", "root_not_object", label, "COCO root must be a JSON object")]

    sections: dict[str, list[dict[str, Any]]] = {}
    for section in ("images", "annotations", "categories"):
        value = data.get(section)
        if not isinstance(value, list):
            add("error", "section_not_list", section, f"{section} must be a list")
            sections[section] = []
            continue
        valid_items: list[dict[str, Any]] = []
        for index, item in enumerate(value):
            if not isinstance(item, dict):
                add("error", "item_not_object", f"{section}[{index}]", f"{section}[{index}] must be an object")
            else:
                valid_items.append(item)
        sections[section] = valid_items

    id_maps: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for section, items in sections.items():
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for index, item in enumerate(items):
            item_id = _as_int(item.get("id"))
            if item_id is None:
                add("error", "invalid_id", f"{section}[{index}]", f"{section}[{index}].id must be an integer")
                continue
            groups[item_id].append(item)
        for item_id, members in groups.items():
            if len(members) > 1:
                add(
                    "error",
                    "duplicate_id",
                    f"{section}:{item_id}",
                    f"duplicate {section} id {item_id} appears {len(members)} times",
                )
        id_maps[section] = groups

    valid_images = set(id_maps["images"])
    valid_categories = set(id_maps["categories"])
    unique_image_by_id = {item_id: members[0] for item_id, members in id_maps["images"].items() if len(members) == 1}
    unique_category_by_id = {
        item_id: members[0] for item_id, members in id_maps["categories"].items() if len(members) == 1
    }

    category_names: dict[str, list[dict[str, Any]]] = defaultdict(list)
    similar_category_names: dict[str, set[str]] = defaultdict(set)
    for index, category in enumerate(sections["categories"]):
        name = category.get("name")
        if not isinstance(name, str) or not name.strip():
            add("error", "invalid_category_name", f"categories[{index}]", "category name must be a non-empty string")
            continue
        category_names[name].append(category)
        similar_category_names[_norm_text(name)].add(name)

    for name, members in category_names.items():
        ids = sorted(
            item_id
            for item_id in (_as_int(item.get("id")) for item in members)
            if item_id is not None
        )
        if len(ids) > 1:
            add(
                "error",
                "duplicate_category_name",
                f"label:{name}",
                f"label/category name {name!r} is registered by multiple category ids {ids}",
            )

    # Case/Unicode-equivalent names are distinct at runtime, so do not merge or
    # reject them automatically. Surface them only as a typo-risk warning.
    for normalized, names in similar_category_names.items():
        if len(names) > 1:
            add(
                "warning",
                "similar_category_names",
                f"label-normalized:{normalized}",
                f"distinct label names {sorted(names)} differ only by Unicode normalization/case",
            )

    referenced_image_ids: set[int] = set()
    label_annotations: dict[str, list[tuple[int | None, int | None]]] = defaultdict(list)
    image_metadata_dims: dict[int, tuple[int, int]] = {}

    for image_id, image in unique_image_by_id.items():
        width = _as_int(image.get("width"))
        height = _as_int(image.get("height"))
        if width is not None and width > 0 and height is not None and height > 0:
            image_metadata_dims[image_id] = (width, height)

    for index, annotation in enumerate(sections["annotations"]):
        annotation_id = _as_int(annotation.get("id"))
        subject = f"annotation:{annotation_id if annotation_id is not None else index}"
        image_id = _as_int(annotation.get("image_id"))
        category_id = _as_int(annotation.get("category_id"))

        if image_id is None:
            add("error", "invalid_image_reference", subject, f"annotation image_id must be an integer, got {annotation.get('image_id')!r}")
        elif image_id not in valid_images:
            add("error", "missing_image_reference", subject, f"annotation references missing image_id {image_id}")
        else:
            referenced_image_ids.add(image_id)

        if category_id is None:
            add(
                "error",
                "invalid_category_reference",
                subject,
                f"annotation category_id must be an integer, got {annotation.get('category_id')!r}",
            )
        elif category_id not in valid_categories:
            add("error", "missing_category_reference", subject, f"annotation references missing category_id {category_id}")
        else:
            category = unique_category_by_id.get(category_id)
            if category is not None:
                name = category.get("name")
                if isinstance(name, str) and name.strip():
                    label_annotations[name].append((annotation_id, image_id))

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            add("error", "invalid_bbox", subject, f"bbox must be a four-number list, got {bbox!r}")
            continue
        values = [_as_number(value) for value in bbox]
        if any(value is None for value in values):
            add("error", "invalid_bbox", subject, f"bbox contains non-finite/non-numeric values: {bbox!r}")
            continue
        x, y, width, height = (float(value) for value in values if value is not None)
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            add("error", "invalid_bbox_geometry", subject, f"bbox must have x/y >= 0 and w/h > 0, got {bbox}")
            continue
        if image_id in image_metadata_dims:
            image_width, image_height = image_metadata_dims[image_id]
            if x + width > image_width + 1e-6 or y + height > image_height + 1e-6:
                add(
                    "error",
                    "bbox_out_of_bounds",
                    subject,
                    f"bbox {bbox} exceeds image {image_id} bounds {image_width}x{image_height}",
                )

    # Source-labeling invariant confirmed by the repository: one feature label
    # maps to one effective source annotation. The packer may rewrite images,
    # but it does not make two definitions of the same label meaningful.
    for name, entries in label_annotations.items():
        if len(entries) > 1:
            add(
                "error",
                "multiple_annotations_for_label",
                f"label:{name}",
                f"label {name!r} has multiple annotations {entries}; source labels must be unique",
            )

    # File name duplication itself is normal source noise (X-AnyLabeling /
    # historical exports can leave extra image registrations). Only the
    # registrations actually used by annotations are hard dependencies.
    image_paths_by_normalized_name: dict[str, set[str]] = defaultdict(set)
    referenced_paths_by_normalized_name: dict[str, set[str]] = defaultdict(set)

    for index, image in enumerate(sections["images"]):
        image_id = _as_int(image.get("id"))
        file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            add("error", "invalid_file_name", f"images[{index}]", "image file_name must be a non-empty string")
            continue
        if not _safe_relative_path(file_name):
            add("error", "unsafe_file_name", f"image:{file_name}", f"image path {file_name!r} must be a safe relative path")
            continue

        normalized_path = _norm_path(file_name)
        image_paths_by_normalized_name[normalized_path].add(file_name)
        if image_id in referenced_image_ids:
            referenced_paths_by_normalized_name[normalized_path].add(file_name)

            width = _as_int(image.get("width"))
            height = _as_int(image.get("height"))
            if width is None or width <= 0 or height is None or height <= 0:
                add(
                    "error",
                    "invalid_image_dimensions",
                    f"image-id:{image_id}",
                    f"referenced image {file_name!r} must have positive integer dimensions, got {image.get('width')}x{image.get('height')}",
                )

            if file_reader is not None:
                blob = file_reader(file_name)
                if blob is None:
                    add(
                        "error",
                        "missing_referenced_image_file",
                        f"image-id:{image_id}",
                        f"referenced source image file {file_name!r} is missing",
                    )
                else:
                    actual = image_dimensions(blob)
                    if (
                        actual is not None
                        and width is not None
                        and height is not None
                        and width > 0
                        and height > 0
                        and actual != (width, height)
                    ):
                        add(
                            "error",
                            "image_dimension_mismatch",
                            f"image-id:{image_id}",
                            f"COCO says {width}x{height}, actual source file {file_name!r} is {actual[0]}x{actual[1]}",
                        )

    # Different physical path spellings that collapse on case-insensitive
    # filesystems are dangerous only when they are both effective inputs.
    for normalized_path, names in referenced_paths_by_normalized_name.items():
        if len(names) > 1:
            add(
                "error",
                "referenced_path_collision",
                f"file-name-normalized:{normalized_path}",
                f"referenced source image paths {sorted(names)} collide after slash/case/Unicode normalization",
            )

    return sorted(issues)


def issue_fingerprints(
    issues: Iterable[AuditIssue],
    *,
    severity: str = "error",
) -> set[tuple[str, str]]:
    return {issue.fingerprint for issue in issues if issue.severity == severity}


def format_audit(issues: Iterable[AuditIssue], *, title: str) -> str:
    issues = list(issues)
    errors = sum(issue.severity == "error" for issue in issues)
    warnings = sum(issue.severity == "warning" for issue in issues)
    lines = [f"{title}: {errors} error(s), {warnings} warning(s)"]
    for issue in issues:
        lines.append(f"- {issue.severity.upper()} {issue.code}: {issue.message}")
    return "\n".join(lines)
