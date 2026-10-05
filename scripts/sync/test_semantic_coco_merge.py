from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


RESOLVER = Path(__file__).with_name("semantic_coco_merge.py").resolve()


def run(cwd: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(
        list(args),
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if check and proc.returncode:
        raise AssertionError(
            f"command failed ({proc.returncode}): {' '.join(args)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def write_coco(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def commit_all(repo: Path, message: str) -> str:
    run(repo, "git", "add", "-A")
    run(repo, "git", "commit", "-m", message)
    return run(repo, "git", "rev-parse", "HEAD").stdout.strip()


class SemanticCocoMergeIntegrationTest(unittest.TestCase):
    def test_id_remap_identical_reuse_and_different_image_rename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run(repo, "git", "init", "-b", "main")
            run(repo, "git", "config", "user.name", "test")
            run(repo, "git", "config", "user.email", "test@example.com")

            (repo / "base.png").write_bytes(b"base")
            base = {
                "images": [{"id": 1, "file_name": "base.png", "width": 1, "height": 1}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [0, 0, 1, 1],
                        "area": 1,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "base", "supercategory": ""}],
            }
            write_coco(repo / "coco_annotations.json", base)
            base_sha = commit_all(repo, "base")

            run(repo, "git", "checkout", "-b", "ours")
            (repo / "same.png").write_bytes(b"identical")
            (repo / "different.png").write_bytes(b"branch-different")
            (repo / "unregistered.png").write_bytes(b"branch-unregistered")
            ours = {
                "images": base["images"]
                + [
                    {"id": 5, "file_name": "same.png", "width": 10, "height": 10},
                    {"id": 6, "file_name": "different.png", "width": 20, "height": 20},
                    {"id": 8, "file_name": "unregistered.png", "width": 30, "height": 30},
                ],
                "annotations": base["annotations"]
                + [
                    {"id": 5, "image_id": 5, "category_id": 5, "bbox": [1, 2, 3, 4], "area": 12, "iscrowd": 0},
                    {"id": 6, "image_id": 6, "category_id": 5, "bbox": [5, 6, 7, 8], "area": 56, "iscrowd": 0},
                    {"id": 8, "image_id": 8, "category_id": 5, "bbox": [9, 10, 11, 12], "area": 132, "iscrowd": 0},
                ],
                "categories": base["categories"]
                + [{"id": 5, "name": "branch_category", "supercategory": ""}],
            }
            write_coco(repo / "coco_annotations.json", ours)
            ours_sha = commit_all(repo, "ours")

            run(repo, "git", "checkout", "-b", "theirs", base_sha)
            (repo / "same.png").write_bytes(b"identical")
            (repo / "different.png").write_bytes(b"upstream-different")
            # This file deliberately exists in Git but is not registered in upstream COCO.
            (repo / "unregistered.png").write_bytes(b"upstream-unregistered")
            theirs = {
                "images": base["images"]
                + [
                    {"id": 2, "file_name": "same.png", "width": 10, "height": 10},
                    {"id": 6, "file_name": "different.png", "width": 21, "height": 21},
                ],
                "annotations": base["annotations"]
                + [
                    {"id": 5, "image_id": 2, "category_id": 5, "bbox": [2, 2, 2, 2], "area": 4, "iscrowd": 0},
                    {"id": 6, "image_id": 6, "category_id": 5, "bbox": [3, 3, 3, 3], "area": 9, "iscrowd": 0},
                ],
                "categories": base["categories"]
                + [{"id": 5, "name": "upstream_category", "supercategory": ""}],
            }
            write_coco(repo / "coco_annotations.json", theirs)
            theirs_sha = commit_all(repo, "theirs")

            run(repo, "git", "checkout", "ours")
            merge = run(repo, "git", "merge", "--no-ff", "--no-commit", theirs_sha, check=False)
            self.assertNotEqual(merge.returncode, 0, "fixture should create real merge conflicts")

            resolved = run(
                repo,
                sys.executable,
                str(RESOLVER),
                "--base-ref",
                base_sha,
                "--ours-ref",
                ours_sha,
                "--theirs-ref",
                theirs_sha,
                "--coco",
                "coco_annotations.json",
            )
            self.assertIn("Semantic COCO merge completed", resolved.stdout)
            self.assertEqual(
                run(repo, "git", "diff", "--name-only", "--diff-filter=U").stdout.strip(),
                "",
            )

            result = json.loads((repo / "coco_annotations.json").read_text(encoding="utf-8"))
            images_by_name = {item["file_name"]: item for item in result["images"]}
            categories_by_name = {item["name"]: item for item in result["categories"]}

            # Identical same-name content is deduplicated onto upstream's image record.
            self.assertEqual(images_by_name["same.png"]["id"], 2)
            self.assertEqual(sum(x["file_name"] == "same.png" for x in result["images"]), 1)

            branch_category_id = categories_by_name["branch_category"]["id"]
            self.assertNotEqual(branch_category_id, 5)

            different_hash = hashlib.sha256(b"branch-different").hexdigest()[:8]
            different_name = f"different_{different_hash}.png"
            self.assertIn(different_name, images_by_name)
            self.assertEqual((repo / different_name).read_bytes(), b"branch-different")
            self.assertEqual((repo / "different.png").read_bytes(), b"upstream-different")

            unregistered_hash = hashlib.sha256(b"branch-unregistered").hexdigest()[:8]
            unregistered_name = f"unregistered_{unregistered_hash}.png"
            self.assertIn(unregistered_name, images_by_name)
            self.assertEqual((repo / unregistered_name).read_bytes(), b"branch-unregistered")
            self.assertEqual((repo / "unregistered.png").read_bytes(), b"upstream-unregistered")

            branch_anns = [
                ann for ann in result["annotations"] if ann["category_id"] == branch_category_id
            ]
            self.assertEqual(len(branch_anns), 3)
            image_ids = {images_by_name["same.png"]["id"], images_by_name[different_name]["id"], images_by_name[unregistered_name]["id"]}
            self.assertEqual({ann["image_id"] for ann in branch_anns}, image_ids)
            self.assertEqual(len({ann["id"] for ann in result["annotations"]}), len(result["annotations"]))


if __name__ == "__main__":
    unittest.main()
