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
        list(args), cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
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
    def test_target_wins_and_source_is_remapped(self) -> None:
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

            run(repo, "git", "checkout", "-b", "target")
            (repo / "same.png").write_bytes(b"identical")
            (repo / "different.png").write_bytes(b"target-bytes")
            target = {
                "images": base["images"]
                + [
                    {"id": 5, "file_name": "same.png", "width": 10, "height": 10},
                    {"id": 6, "file_name": "different.png", "width": 20, "height": 20},
                ],
                "annotations": base["annotations"]
                + [
                    {"id": 5, "image_id": 5, "category_id": 5, "bbox": [1, 1, 2, 2], "area": 4, "iscrowd": 0},
                    {"id": 6, "image_id": 6, "category_id": 6, "bbox": [2, 2, 2, 2], "area": 4, "iscrowd": 0},
                ],
                "categories": base["categories"]
                + [
                    {"id": 5, "name": "target_same", "supercategory": ""},
                    {"id": 6, "name": "target_different", "supercategory": ""},
                ],
            }
            write_coco(repo / "coco_annotations.json", target)
            target_sha = commit_all(repo, "target")

            run(repo, "git", "checkout", "-b", "source", base_sha)
            (repo / "same.png").write_bytes(b"identical")
            (repo / "different.png").write_bytes(b"source-bytes")
            source = {
                "images": base["images"]
                + [
                    {"id": 5, "file_name": "same.png", "width": 10, "height": 10},
                    {"id": 6, "file_name": "different.png", "width": 21, "height": 21},
                ],
                "annotations": base["annotations"]
                + [
                    {"id": 5, "image_id": 5, "category_id": 5, "bbox": [3, 3, 3, 3], "area": 9, "iscrowd": 0},
                    {"id": 6, "image_id": 6, "category_id": 6, "bbox": [4, 4, 4, 4], "area": 16, "iscrowd": 0},
                ],
                "categories": base["categories"]
                + [
                    {"id": 5, "name": "source_same", "supercategory": ""},
                    {"id": 6, "name": "source_different", "supercategory": ""},
                ],
            }
            write_coco(repo / "coco_annotations.json", source)
            source_sha = commit_all(repo, "source")

            run(repo, "git", "checkout", "target")
            merge = run(repo, "git", "merge", "--no-ff", "--no-commit", source_sha, check=False)
            self.assertNotEqual(merge.returncode, 0)

            resolved = run(
                repo,
                sys.executable,
                str(RESOLVER),
                "--base-ref",
                base_sha,
                "--target-ref",
                target_sha,
                "--source-ref",
                source_sha,
                "--coco",
                "coco_annotations.json",
            )
            self.assertIn("target priority", resolved.stdout)
            self.assertEqual(
                run(repo, "git", "diff", "--name-only", "--diff-filter=U").stdout.strip(),
                "",
            )

            result = json.loads((repo / "coco_annotations.json").read_text(encoding="utf-8"))
            cats = {x["name"]: x["id"] for x in result["categories"]}
            images = {x["file_name"]: x["id"] for x in result["images"]}

            self.assertEqual(cats["target_same"], 5)
            self.assertEqual(cats["target_different"], 6)
            self.assertGreater(cats["source_same"], 6)
            self.assertGreater(cats["source_different"], 6)
            self.assertEqual(images["same.png"], 5)

            hashed = f"different_{hashlib.sha256(b'source-bytes').hexdigest()[:8]}.png"
            self.assertIn(hashed, images)
            self.assertEqual((repo / "different.png").read_bytes(), b"target-bytes")
            self.assertEqual((repo / hashed).read_bytes(), b"source-bytes")

            target_ann5 = next(x for x in result["annotations"] if x["id"] == 5)
            target_ann6 = next(x for x in result["annotations"] if x["id"] == 6)
            self.assertEqual(target_ann5["category_id"], 5)
            self.assertEqual(target_ann6["category_id"], 6)

            source_ann_ids = [
                x["id"] for x in result["annotations"] if x["id"] not in {1, 5, 6}
            ]
            self.assertEqual(len(source_ann_ids), 2)
            self.assertTrue(all(x > 6 for x in source_ann_ids))

    def test_target_may_modify_base_while_source_additions_are_imported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run(repo, "git", "init", "-b", "main")
            run(repo, "git", "config", "user.name", "test")
            run(repo, "git", "config", "user.email", "test@example.com")
            (repo / "base.png").write_bytes(b"base")
            base = {
                "images": [{"id": 1, "file_name": "base.png", "width": 1, "height": 1}],
                "annotations": [],
                "categories": [{"id": 1, "name": "base", "supercategory": ""}],
            }
            write_coco(repo / "coco_annotations.json", base)
            base_sha = commit_all(repo, "base")

            run(repo, "git", "checkout", "-b", "target")
            target = json.loads(json.dumps(base))
            target["categories"][0]["supercategory"] = "target-owned-change"
            write_coco(repo / "coco_annotations.json", target)
            target_sha = commit_all(repo, "target changes existing record")

            run(repo, "git", "checkout", "-b", "source", base_sha)
            source = json.loads(json.dumps(base))
            source["categories"].append({"id": 2, "name": "source_new", "supercategory": ""})
            write_coco(repo / "coco_annotations.json", source)
            source_sha = commit_all(repo, "source adds")

            run(repo, "git", "checkout", "target")
            run(repo, "git", "merge", "--no-ff", "--no-commit", source_sha, check=False)
            run(
                repo,
                sys.executable,
                str(RESOLVER),
                "--base-ref",
                base_sha,
                "--target-ref",
                target_sha,
                "--source-ref",
                source_sha,
            )
            result = json.loads((repo / "coco_annotations.json").read_text(encoding="utf-8"))
            self.assertEqual(result["categories"][0]["supercategory"], "target-owned-change")
            self.assertIn("source_new", {x["name"] for x in result["categories"]})


if __name__ == "__main__":
    unittest.main()
