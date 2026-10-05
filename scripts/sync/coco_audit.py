#!/usr/bin/env python3
from __future__ import annotations

from collections import Counter, defaultdict
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
    p = PurePosixPath(value.replace("\\", "/"))
    return not p.is_absolute() and ".." not in p.parts

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
    value = float(value)
    return value if math.isfinite(value) else None

def _png_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) >= 24 and data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        return struct.unpack(">II", data[16:24])
    return None

def _jpeg_dimensions(data: bytes) -> tuple[int, int] | None:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        return None
    i = 2
    sof = {0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF}
    while i + 4 <= len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        while i < len(data) and data[i] == 0xFF:
            i += 1
        if i >= len(data):
            break
        marker = data[i]
        i += 1
        if marker in {0xD8,0xD9} or 0xD0 <= marker <= 0xD7:
            continue
        if i + 2 > len(data):
            break
        seglen = int.from_bytes(data[i:i+2], "big")
        if seglen < 2 or i + seglen > len(data):
            break
        if marker in sof and seglen >= 7:
            height = int.from_bytes(data[i+3:i+5], "big")
            width = int.from_bytes(data[i+5:i+7], "big")
            return width, height
        i += seglen
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

def _semantic_annotation_key(ann: dict[str, Any]) -> tuple[Any, ...] | None:
    bbox = ann.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    nums = tuple(_as_number(x) for x in bbox)
    if any(x is None for x in nums):
        return None
    return (_as_int(ann.get("image_id")), _as_int(ann.get("category_id")), nums, ann.get("iscrowd"))

def audit_coco(data: dict[str, Any], *, label: str, file_reader: Callable[[str], bytes | None] | None = None) -> list[AuditIssue]:
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
        for i, item in enumerate(value):
            if not isinstance(item, dict):
                add("error", "item_not_object", f"{section}[{i}]", f"{section}[{i}] must be an object")
        sections[section] = [item for item in value if isinstance(item, dict)]
    id_maps: dict[str, dict[int, list[dict[str, Any]]]] = {}
    for section, items in sections.items():
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for idx, item in enumerate(items):
            item_id = _as_int(item.get("id"))
            if item_id is None:
                add("error", "invalid_id", f"{section}[{idx}]", f"{section}[{idx}].id must be an integer")
                continue
            groups[item_id].append(item)
        for item_id, members in groups.items():
            if len(members) > 1:
                add("error", "duplicate_id", f"{section}:{item_id}", f"duplicate {section} id {item_id} appears {len(members)} times")
        id_maps[section] = groups
    cats_exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    cats_norm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for idx, cat in enumerate(sections["categories"]):
        name = cat.get("name")
        if not isinstance(name, str) or not name.strip():
            add("error", "invalid_category_name", f"categories[{idx}]", "category name must be a non-empty string")
            continue
        cats_exact[name].append(cat); cats_norm[_norm_text(name)].append(cat)
    for name, members in cats_exact.items():
        ids = sorted(_as_int(x.get("id")) for x in members if _as_int(x.get("id")) is not None)
        if len(ids) > 1:
            add("error", "duplicate_category_name", f"category-name:{name}", f"category name {name!r} is used by ids {ids}")
    for norm, members in cats_norm.items():
        names = sorted({str(x.get("name")) for x in members})
        if len(names) > 1:
            ids = sorted(_as_int(x.get("id")) for x in members if _as_int(x.get("id")) is not None)
            add("error", "normalized_category_name_collision", f"category-name-norm:{norm}", f"category names {names} collide after Unicode NFC + casefold; ids={ids}")
    images_exact: dict[str, list[dict[str, Any]]] = defaultdict(list)
    images_norm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    content_hashes: dict[str, list[str]] = defaultdict(list)
    image_dims: dict[int, tuple[int,int]] = {}
    import hashlib
    for idx, image in enumerate(sections["images"]):
        image_id = _as_int(image.get("id")); file_name = image.get("file_name")
        if not isinstance(file_name, str) or not file_name:
            add("error", "invalid_file_name", f"images[{idx}]", "image file_name must be a non-empty string"); continue
        if not _safe_relative_path(file_name):
            add("error", "unsafe_file_name", f"image:{file_name}", f"image path {file_name!r} must be a safe relative path")
        images_exact[file_name].append(image); images_norm[_norm_path(file_name)].append(image)
        width = _as_int(image.get("width")); height = _as_int(image.get("height"))
        if width is None or width <= 0 or height is None or height <= 0:
            add("error", "invalid_image_dimensions", f"image:{file_name}", f"image width/height must be positive integers, got {image.get('width')}x{image.get('height')}")
        elif image_id is not None:
            image_dims[image_id] = (width,height)
        if file_reader is not None:
            blob = file_reader(file_name)
            if blob is None:
                add("error", "missing_image_file", f"image:{file_name}", f"COCO references missing image file {file_name}")
            else:
                content_hashes[hashlib.sha256(blob).hexdigest()].append(file_name)
                actual = image_dimensions(blob)
                if actual is not None and width is not None and height is not None and actual != (width,height):
                    add("error", "image_dimension_mismatch", f"image:{file_name}", f"COCO says {width}x{height}, actual file is {actual[0]}x{actual[1]}")
    for name, members in images_exact.items():
        ids = sorted(_as_int(x.get("id")) for x in members if _as_int(x.get("id")) is not None)
        if len(ids) > 1:
            add("error", "duplicate_file_name", f"file-name:{name}", f"file_name {name!r} is used by image ids {ids}")
    for norm, members in images_norm.items():
        names = sorted({str(x.get("file_name")) for x in members})
        if len(names) > 1:
            ids = sorted(_as_int(x.get("id")) for x in members if _as_int(x.get("id")) is not None)
            add("error", "normalized_file_name_collision", f"file-name-norm:{norm}", f"image paths {names} collide after slash normalization + Unicode NFC + casefold; ids={ids}")
    for digest, names in content_hashes.items():
        uniq = sorted(set(names))
        if len(uniq) > 1:
            add("warning", "duplicate_image_content", f"sha256:{digest}", f"identical image bytes are stored under multiple names: {uniq}")
    valid_images = set(id_maps["images"]); valid_cats = set(id_maps["categories"])
    sem_anns: dict[tuple[Any,...], list[int]] = defaultdict(list); ref_images: Counter[int] = Counter(); ref_cats: Counter[int] = Counter()
    for idx, ann in enumerate(sections["annotations"]):
        ann_id = _as_int(ann.get("id")); image_id = _as_int(ann.get("image_id")); cat_id = _as_int(ann.get("category_id")); subject = f"annotation:{ann_id if ann_id is not None else idx}"
        if image_id is None: add("error", "invalid_image_reference", subject, f"annotation image_id must be an integer, got {ann.get('image_id')!r}")
        elif image_id not in valid_images: add("error", "missing_image_reference", subject, f"annotation references missing image_id {image_id}")
        else: ref_images[image_id] += 1
        if cat_id is None: add("error", "invalid_category_reference", subject, f"annotation category_id must be an integer, got {ann.get('category_id')!r}")
        elif cat_id not in valid_cats: add("error", "missing_category_reference", subject, f"annotation references missing category_id {cat_id}")
        else: ref_cats[cat_id] += 1
        bbox = ann.get("bbox"); bbox_nums = None
        if not isinstance(bbox, list) or len(bbox) != 4:
            add("error", "invalid_bbox", subject, f"bbox must be a four-number list, got {bbox!r}")
        else:
            vals = [_as_number(x) for x in bbox]
            if any(v is None for v in vals): add("error", "invalid_bbox", subject, f"bbox contains non-finite/non-numeric values: {bbox!r}")
            else:
                bbox_nums = [float(v) for v in vals if v is not None]; x,y,w,h = bbox_nums
                if x < 0 or y < 0 or w <= 0 or h <= 0: add("error", "invalid_bbox_geometry", subject, f"bbox must have x/y >= 0 and w/h > 0, got {bbox}")
                if image_id in image_dims:
                    iw, ih = image_dims[image_id]
                    if x + w > iw + 1e-6 or y + h > ih + 1e-6: add("error", "bbox_out_of_bounds", subject, f"bbox {bbox} exceeds image {image_id} bounds {iw}x{ih}")
        area = _as_number(ann.get("area"))
        if area is None or area <= 0: add("error", "invalid_area", subject, f"annotation area must be a positive finite number, got {ann.get('area')!r}")
        elif bbox_nums is not None and not ann.get("segmentation"):
            expected = bbox_nums[2] * bbox_nums[3]
            if abs(area - expected) > max(1e-6, abs(expected)*1e-6): add("warning", "area_bbox_mismatch", subject, f"area {area} differs from bbox area {expected}")
        if ann.get("iscrowd") not in (0,1,False,True): add("warning", "unexpected_iscrowd", subject, f"iscrowd should normally be 0 or 1, got {ann.get('iscrowd')!r}")
        sem = _semantic_annotation_key(ann)
        if sem is not None and ann_id is not None: sem_anns[sem].append(ann_id)
    for key, ids in sem_anns.items():
        if len(ids) > 1: add("warning", "duplicate_annotation", f"annotation-semantic:{repr(key)}", f"annotations {sorted(ids)} have the same image/category/bbox payload")
    for image_id in sorted(valid_images):
        if ref_images[image_id] == 0: add("warning", "unreferenced_image", f"image-id:{image_id}", f"image id {image_id} has no annotations")
    for cat_id in sorted(valid_cats):
        if ref_cats[cat_id] == 0: add("warning", "unreferenced_category", f"category-id:{cat_id}", f"category id {cat_id} has no annotations")
    return sorted(issues)

def issue_fingerprints(issues: Iterable[AuditIssue], *, severity: str = "error") -> set[tuple[str,str]]:
    return {x.fingerprint for x in issues if x.severity == severity}

def format_audit(issues: Iterable[AuditIssue], *, title: str) -> str:
    issues = list(issues); errors = sum(x.severity == "error" for x in issues); warnings = sum(x.severity == "warning" for x in issues)
    lines = [f"{title}: {errors} error(s), {warnings} warning(s)"]
    lines += [f"- {x.severity.upper()} {x.code}: {x.message}" for x in issues]
    return "\n".join(lines)
