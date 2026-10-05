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
    proc = subprocess.run(list(args), cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode:
        raise AssertionError(f"command failed ({proc.returncode}): {' '.join(args)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")
    return proc

def write_coco(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")

def commit_all(repo: Path, message: str) -> str:
    run(repo, "git", "add", "-A"); run(repo, "git", "commit", "-m", message)
    return run(repo, "git", "rev-parse", "HEAD").stdout.strip()

def setup_repo(tmp: str) -> Path:
    repo = Path(tmp)
    run(repo, "git", "init", "-b", "main"); run(repo, "git", "config", "user.name", "test"); run(repo, "git", "config", "user.email", "test@example.com")
    return repo

def resolve(repo: Path, base: str, target: str, source: str, *, check: bool = True):
    run(repo, "git", "checkout", "target"); run(repo, "git", "merge", "--no-ff", "--no-commit", source, check=False)
    return run(repo, sys.executable, str(RESOLVER), "--base-ref", base, "--target-ref", target, "--source-ref", source, check=check)

class SemanticCocoMergeIntegrationTest(unittest.TestCase):
    def test_target_wins_and_source_is_remapped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = setup_repo(tmp); (repo / "base.png").write_bytes(b"base")
            base = {"images":[{"id":1,"file_name":"base.png","width":100,"height":100}],"annotations":[{"id":1,"image_id":1,"category_id":1,"bbox":[0,0,1,1],"area":1,"iscrowd":0}],"categories":[{"id":1,"name":"base","supercategory":""}]}
            write_coco(repo / "coco_annotations.json", base); base_sha = commit_all(repo, "base")
            run(repo, "git", "checkout", "-b", "target"); (repo / "same.png").write_bytes(b"identical"); (repo / "different.png").write_bytes(b"target-bytes")
            target = {"images":base["images"]+[{"id":5,"file_name":"same.png","width":10,"height":10},{"id":6,"file_name":"different.png","width":20,"height":20}],"annotations":base["annotations"]+[{"id":5,"image_id":5,"category_id":5,"bbox":[1,1,2,2],"area":4,"iscrowd":0},{"id":6,"image_id":6,"category_id":6,"bbox":[2,2,2,2],"area":4,"iscrowd":0}],"categories":base["categories"]+[{"id":5,"name":"target_same","supercategory":""},{"id":6,"name":"target_different","supercategory":""}]}
            write_coco(repo / "coco_annotations.json", target); target_sha = commit_all(repo, "target")
            run(repo, "git", "checkout", "-b", "source", base_sha); (repo / "same.png").write_bytes(b"identical"); (repo / "different.png").write_bytes(b"source-bytes")
            source = {"images":base["images"]+[{"id":5,"file_name":"same.png","width":10,"height":10},{"id":6,"file_name":"different.png","width":21,"height":21}],"annotations":base["annotations"]+[{"id":5,"image_id":5,"category_id":5,"bbox":[3,3,3,3],"area":9,"iscrowd":0},{"id":6,"image_id":6,"category_id":6,"bbox":[4,4,4,4],"area":16,"iscrowd":0}],"categories":base["categories"]+[{"id":5,"name":"source_same","supercategory":""},{"id":6,"name":"source_different","supercategory":""}]}
            write_coco(repo / "coco_annotations.json", source); source_sha = commit_all(repo, "source")
            output = resolve(repo, base_sha, target_sha, source_sha); self.assertIn("target priority", output.stdout)
            data = json.loads((repo / "coco_annotations.json").read_text()); cats = {x["name"]:x["id"] for x in data["categories"]}; images = {x["file_name"]:x["id"] for x in data["images"]}
            self.assertEqual(cats["target_same"],5); self.assertEqual(cats["target_different"],6); self.assertGreater(cats["source_same"],6); self.assertGreater(cats["source_different"],6); self.assertEqual(images["same.png"],5)
            hashed = f"different_{hashlib.sha256(b'source-bytes').hexdigest()[:8]}.png"; self.assertIn(hashed, images); self.assertEqual((repo / "different.png").read_bytes(), b"target-bytes"); self.assertEqual((repo / hashed).read_bytes(), b"source-bytes")

    def test_normalized_category_name_conflict_with_different_semantics_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = setup_repo(tmp); (repo / "base.png").write_bytes(b"base")
            base={"images":[{"id":1,"file_name":"base.png","width":10,"height":10}],"annotations":[],"categories":[]}; write_coco(repo/"coco_annotations.json",base); base_sha=commit_all(repo,"base")
            run(repo,"git","checkout","-b","target"); target=json.loads(json.dumps(base)); target["categories"].append({"id":10,"name":"Foo","supercategory":"target"}); write_coco(repo/"coco_annotations.json",target); target_sha=commit_all(repo,"target")
            run(repo,"git","checkout","-b","source",base_sha); source=json.loads(json.dumps(base)); source["categories"].append({"id":20,"name":"foo","supercategory":"source"}); write_coco(repo/"coco_annotations.json",source); source_sha=commit_all(repo,"source")
            result=resolve(repo,base_sha,target_sha,source_sha,check=False); self.assertNotEqual(result.returncode,0); self.assertIn("semantically ambiguous",result.stderr)

    def test_normalized_category_name_same_semantics_reuses_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo=setup_repo(tmp); (repo/"img.png").write_bytes(b"img"); base={"images":[{"id":1,"file_name":"img.png","width":10,"height":10}],"annotations":[],"categories":[]}; write_coco(repo/"coco_annotations.json",base); base_sha=commit_all(repo,"base")
            run(repo,"git","checkout","-b","target"); target=json.loads(json.dumps(base)); target["categories"].append({"id":10,"name":"Foo","supercategory":"same"}); write_coco(repo/"coco_annotations.json",target); target_sha=commit_all(repo,"target")
            run(repo,"git","checkout","-b","source",base_sha); source=json.loads(json.dumps(base)); source["categories"].append({"id":20,"name":"foo","supercategory":"same"}); source["annotations"].append({"id":20,"image_id":1,"category_id":20,"bbox":[1,1,2,2],"area":4,"iscrowd":0}); write_coco(repo/"coco_annotations.json",source); source_sha=commit_all(repo,"source")
            resolve(repo,base_sha,target_sha,source_sha); data=json.loads((repo/"coco_annotations.json").read_text()); self.assertEqual([(x["id"],x["name"]) for x in data["categories"]],[(10,"Foo")]); self.assertEqual(data["annotations"][0]["category_id"],10)

    def test_casefold_image_collision_renames_only_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo=setup_repo(tmp); base={"images":[],"annotations":[],"categories":[]}; write_coco(repo/"coco_annotations.json",base); base_sha=commit_all(repo,"base")
            run(repo,"git","checkout","-b","target"); (repo/"Foo.PNG").write_bytes(b"target"); target={"images":[{"id":10,"file_name":"Foo.PNG","width":10,"height":10}],"annotations":[],"categories":[]}; write_coco(repo/"coco_annotations.json",target); target_sha=commit_all(repo,"target")
            run(repo,"git","checkout","-b","source",base_sha); (repo/"foo.png").write_bytes(b"source"); source={"images":[{"id":10,"file_name":"foo.png","width":11,"height":11}],"annotations":[],"categories":[]}; write_coco(repo/"coco_annotations.json",source); source_sha=commit_all(repo,"source")
            resolve(repo,base_sha,target_sha,source_sha); data=json.loads((repo/"coco_annotations.json").read_text()); names={x["file_name"] for x in data["images"]}; expected=f"foo_{hashlib.sha256(b'source').hexdigest()[:8]}.png"; self.assertIn("Foo.PNG",names); self.assertIn(expected,names); self.assertEqual((repo/"Foo.PNG").read_bytes(),b"target"); self.assertEqual((repo/expected).read_bytes(),b"source")

    def test_identical_image_bytes_but_metadata_conflict_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo=setup_repo(tmp); base={"images":[],"annotations":[],"categories":[]}; write_coco(repo/"coco_annotations.json",base); base_sha=commit_all(repo,"base")
            run(repo,"git","checkout","-b","target"); (repo/"same.png").write_bytes(b"same"); write_coco(repo/"coco_annotations.json",{"images":[{"id":1,"file_name":"same.png","width":10,"height":10}],"annotations":[],"categories":[]}); target_sha=commit_all(repo,"target")
            run(repo,"git","checkout","-b","source",base_sha); (repo/"same.png").write_bytes(b"same"); write_coco(repo/"coco_annotations.json",{"images":[{"id":2,"file_name":"same.png","width":20,"height":20}],"annotations":[],"categories":[]}); source_sha=commit_all(repo,"source")
            result=resolve(repo,base_sha,target_sha,source_sha,check=False); self.assertNotEqual(result.returncode,0); self.assertIn("conflicting COCO metadata",result.stderr)

    def test_source_new_invalid_reference_is_rejected_before_merge_result(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo=setup_repo(tmp); base={"images":[],"annotations":[],"categories":[]}; write_coco(repo/"coco_annotations.json",base); base_sha=commit_all(repo,"base")
            run(repo,"git","checkout","-b","target"); target_sha=base_sha
            run(repo,"git","checkout","-b","source",base_sha); source={"images":[],"annotations":[{"id":1,"image_id":999,"category_id":888,"bbox":[0,0,1,1],"area":1,"iscrowd":0}],"categories":[]}; write_coco(repo/"coco_annotations.json",source); source_sha=commit_all(repo,"source")
            result=resolve(repo,base_sha,target_sha,source_sha,check=False); self.assertNotEqual(result.returncode,0); self.assertIn("Source introduces integrity errors",result.stderr)

    def test_exact_duplicate_annotation_after_remap_is_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo=setup_repo(tmp); (repo/"img.png").write_bytes(b"same"); base={"images":[{"id":1,"file_name":"img.png","width":10,"height":10}],"annotations":[],"categories":[]}; write_coco(repo/"coco_annotations.json",base); base_sha=commit_all(repo,"base")
            run(repo,"git","checkout","-b","target"); target=json.loads(json.dumps(base)); target["categories"].append({"id":10,"name":"foo","supercategory":""}); target["annotations"].append({"id":10,"image_id":1,"category_id":10,"bbox":[1,1,2,2],"area":4,"iscrowd":0}); write_coco(repo/"coco_annotations.json",target); target_sha=commit_all(repo,"target")
            run(repo,"git","checkout","-b","source",base_sha); source=json.loads(json.dumps(base)); source["categories"].append({"id":20,"name":"foo","supercategory":""}); source["annotations"].append({"id":20,"image_id":1,"category_id":20,"bbox":[1,1,2,2],"area":4,"iscrowd":0}); write_coco(repo/"coco_annotations.json",source); source_sha=commit_all(repo,"source")
            output=resolve(repo,base_sha,target_sha,source_sha); data=json.loads((repo/"coco_annotations.json").read_text()); self.assertEqual(len(data["annotations"]),1); self.assertIn("skipped exact duplicate",output.stdout)

if __name__ == "__main__":
    unittest.main()
