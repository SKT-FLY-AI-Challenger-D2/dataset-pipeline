#!/usr/bin/env python3
import os
import json
import argparse
from pathlib import Path

DATA_ROOT = Path("playground/data")
FAKE_DIR = DATA_ROOT / "fake"
REAL_DIR = DATA_ROOT / "real"
TRAIN_JSONL = DATA_ROOT / "train.jsonl"
TEST_JSONL  = DATA_ROOT / "test.jsonl"

def load_referenced_images(*jsonl_paths: Path) -> set[str]:
    """
    JSONL들의 각 라인에서 obj["image"] 값을 모아 set으로 반환.
    예: "fake/abf..._005.png"
    """
    referenced = set()
    for p in jsonl_paths:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                img = obj.get("image")
                if isinstance(img, str) and img:
                    referenced.add(img.replace("\\", "/"))
    return referenced

def sync_jsonl_with_disk(jsonl_path: Path, dry_run: bool = True, remove_errors: bool = False):
    """
    JSONL 파일을 읽어서 이미지가 실제로 디렉토리에 없으면 해당 라인 삭제.
    API 에러가 포함된 라인도 선택적으로 삭제 가능.
    """
    if not jsonl_path.exists():
        print(f"[WARN] JSONL not found: {jsonl_path}")
        return

    print(f"\n[SYNC] Checking {jsonl_path}...")
    
    kept_lines = []
    removed_count = 0
    error_removed_count = 0

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                print(f"[WARN] Skipping corrupted line in {jsonl_path}")
                continue

            img_rel_path = obj.get("image")
            if not img_rel_path:
                print(f"[REMOVE] No image field in line: {line[:100]}...")
                removed_count += 1
                continue

            full_img_path = DATA_ROOT / img_rel_path
            
            # API 에러 체크
            is_api_error = "Analysis failed due to API error" in line
            if remove_errors and is_api_error:
                error_removed_count += 1
                continue
            
            # 이미지 존재 여부 체크
            if not full_img_path.exists():
                print(f"[REMOVE] Missing image: {img_rel_path}")
                removed_count += 1
                continue

            kept_lines.append(line)

    print(f"Summary for {jsonl_path.name}:")
    print(f"  - Kept: {len(kept_lines)}")
    print(f"  - Removed (Missing Image): {removed_count}")
    if remove_errors:
        print(f"  - Removed (API Error): {error_removed_count}")

    if not dry_run:
        if removed_count > 0 or error_removed_count > 0:
            with jsonl_path.open("w", encoding="utf-8") as f:
                for kl in kept_lines:
                    f.write(kl + "\n")
            print(f"[DONE] Updated {jsonl_path}")
        else:
            print("[INFO] No changes needed.")
    else:
        print(f"[DRY-RUN] No files modified.")

def cleanup_orphan_images(dry_run: bool = True) -> None:
    """
    fake/ 및 real/ 폴더에 있는 이미지 파일 중
    train/test jsonl 어느 곳에도 등장하지 않으면 삭제.
    """
    referenced = load_referenced_images(TRAIN_JSONL, TEST_JSONL)

    for target_dir in [FAKE_DIR, REAL_DIR]:
        if not target_dir.exists():
            continue
        
        print(f"\n[CLEANUP] Checking orphan images in {target_dir}...")
        deleted = 0
        kept = 0
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        
        for fp in sorted(target_dir.iterdir()):
            if not fp.is_file() or fp.suffix.lower() not in exts:
                continue

            rel = fp.relative_to(DATA_ROOT).as_posix()  # "fake/xxx.png"
            if rel in referenced:
                kept += 1
                continue

            if dry_run:
                if deleted < 10: # 너무 많으면 일부만 출력
                    print(f"[DRY-RUN] Would delete: {rel}")
                deleted += 1
            else:
                try:
                    fp.unlink()
                    deleted += 1
                except Exception as e:
                    print(f"[ERROR] Failed to delete {fp}: {e}")

        print(f"Summary for {target_dir.name}: kept={kept}, deleted={deleted} (dry_run={dry_run})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", default=False, help="삭제하지 않고 대상만 출력")
    parser.add_argument("--execute", action="store_true", help="실제로 삭제 및 파일 수정을 진행 (dry-run 해제)")
    parser.add_argument("--remove-errors", action="store_true", help="API 에러 메시지가 포함된 항목도 삭제")
    args = parser.parse_args()

    # 안전을 위해 명시적으로 --execute를 주지 않으면 dry_run=True
    dry_run = not args.execute
    
    print(f"Options: dry_run={dry_run}, remove_errors={args.remove_errors}")
    
    # 1. JSONL 싱크 (이미지가 없는 라인 제거 + 옵션으로 에러 라인 제거)
    sync_jsonl_with_disk(TRAIN_JSONL, dry_run=dry_run, remove_errors=args.remove_errors)
    sync_jsonl_with_disk(TEST_JSONL, dry_run=dry_run, remove_errors=args.remove_errors)
    
    # 2. 고아 이미지 삭제 (JSONL에 없는 파일 삭제)
    cleanup_orphan_images(dry_run=dry_run)
