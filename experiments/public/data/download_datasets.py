#!/usr/bin/env python3
"""
Download ShareGPTVideo/train_video_and_instruction and TIGER-Lab/MMEB-train
datasets and extract their archives in parallel.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import os
import tarfile
import zipfile
from pathlib import Path
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from tqdm import tqdm


CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming extraction


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    relative_dir: str
    allow_patterns: Sequence[str] | None = None
    extract_subdirs: Sequence[str] | None = None


DATASETS: Sequence[DatasetSpec] = (
    DatasetSpec(repo_id="TIGER-Lab/MMEB-train", relative_dir="MMEB-train"),
    DatasetSpec(
        repo_id="ShareGPTVideo/train_video_and_instruction",
        relative_dir="video",
        allow_patterns=("train_300k", "train_300k/*"),
        extract_subdirs=("train_300k",),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download VLM2Vec training datasets and extract archives concurrently."
    )
    parser.add_argument(
        "--basedir",
        type=Path,
        default=Path("data/vlm2vec_train"),
        help="Directory used to store the downloaded datasets.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=max(os.cpu_count() or 1, 4),
        help="Maximum number of parallel extraction workers.",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only extract archives; assume the datasets exist locally already.",
    )
    return parser.parse_args()


def download_dataset(
    repo_id: str, local_dir: Path, allow_patterns: Sequence[str] | None
) -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading {repo_id} to {local_dir} ...")
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
        allow_patterns=list(allow_patterns) if allow_patterns else None,
    )


def ensure_within_directory(base: Path, candidate: Path, archive: Path, member_name: str) -> None:
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise RuntimeError(
            f"Blocked path traversal attempt from {archive}: {member_name}"
        ) from exc


def is_archive(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith((".tar.gz", ".tgz", ".tar", ".zip"))


def find_archives(root: Path) -> Iterable[Path]:
    for path in root.rglob("*"):
        if path.is_file() and is_archive(path):
            yield path


def safe_extract_tar(archive: Path, dest: Path, position: int | None) -> None:
    with tarfile.open(archive, "r:*") as tar:
        dest = dest.resolve()
        members = tar.getmembers()
        total_bytes = sum(max(member.size, 0) for member in members if member.size)
        use_bytes = total_bytes > 0
        with tqdm(
            total=total_bytes if use_bytes else len(members),
            desc=f"{archive.name}",
            unit="B" if use_bytes else "file",
            unit_scale=use_bytes,
            position=position,
            leave=False,
        ) as member_bar:
            for member in members:
                member_path = (dest / member.name).resolve()
                ensure_within_directory(dest, member_path, archive, member.name)
                if member.isdir():
                    member_path.mkdir(parents=True, exist_ok=True)
                    if not use_bytes:
                        member_bar.update(1)
                    continue

                if member.isfile():
                    member_path.parent.mkdir(parents=True, exist_ok=True)
                    source = tar.extractfile(member)
                    if source is None:
                        raise RuntimeError(f"Failed to extract {member.name} in {archive}")
                    with source, open(member_path, "wb") as out_file:
                        while True:
                            chunk = source.read(CHUNK_SIZE)
                            if not chunk:
                                break
                            out_file.write(chunk)
                            if use_bytes:
                                member_bar.update(len(chunk))
                    os.chmod(member_path, member.mode)
                    if not use_bytes:
                        member_bar.update(1)
                    continue

                # For other member types (symlinks, etc.) fall back to default extraction.
                tar.extract(member, dest)
                if not use_bytes:
                    member_bar.update(1)


def safe_extract_zip(archive: Path, dest: Path, position: int | None) -> None:
    with zipfile.ZipFile(archive) as zf:
        dest = dest.resolve()
        infos = zf.infolist()
        total_bytes = sum(info.file_size for info in infos if info.file_size)
        use_bytes = total_bytes > 0
        with tqdm(
            total=total_bytes if use_bytes else len(infos),
            desc=f"{archive.name}",
            unit="B" if use_bytes else "file",
            unit_scale=use_bytes,
            position=position,
            leave=False,
        ) as member_bar:
            for info in infos:
                member_path = (dest / info.filename).resolve()
                ensure_within_directory(dest, member_path, archive, info.filename)
                if info.is_dir():
                    member_path.mkdir(parents=True, exist_ok=True)
                    if not use_bytes:
                        member_bar.update(1)
                    continue

                member_path.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as source, open(member_path, "wb") as out_file:
                    while True:
                        chunk = source.read(CHUNK_SIZE)
                        if not chunk:
                            break
                        out_file.write(chunk)
                        if use_bytes:
                            member_bar.update(len(chunk))
                if not use_bytes:
                    member_bar.update(1)


def extract_archive(archive: Path, position: int | None = None) -> Path:
    target_dir = archive.parent
    if archive.suffix == ".zip":
        safe_extract_zip(archive, target_dir, position)
    else:
        safe_extract_tar(archive, target_dir, position)
    return archive


def parallel_extract(archives: List[Path], max_workers: int) -> None:
    if not archives:
        print("No archives found for extraction.")
        return

    workers = min(max_workers, len(archives))
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_archive = {
            executor.submit(extract_archive, archive, idx + 1): archive
            for idx, archive in enumerate(archives)
        }
        with tqdm(total=len(archives), desc="Archives", unit="archive", position=0) as pbar:
            for future in concurrent.futures.as_completed(future_to_archive):
                archive = future_to_archive[future]
                try:
                    future.result()
                except Exception as exc:  # pylint: disable=broad-exception-caught
                    print(f"Extraction failed for {archive}: {exc}")
                    raise
                finally:
                    pbar.update(1)


def main() -> None:
    args = parse_args()
    base_dir: Path = args.basedir.expanduser().resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    extract_dirs = []
    for spec in DATASETS:
        local_dir = base_dir / spec.relative_dir
        local_dir.mkdir(parents=True, exist_ok=True)
        if not args.skip_download:
            download_dataset(spec.repo_id, local_dir, spec.allow_patterns)

        if spec.extract_subdirs:
            for subdir in spec.extract_subdirs:
                extract_dirs.append((local_dir / subdir).resolve())
        else:
            extract_dirs.append(local_dir)

    archives: List[Path] = []
    for directory in extract_dirs:
        if directory.exists():
            archives.extend(list(find_archives(directory)))

    parallel_extract(archives, args.max_workers)


if __name__ == "__main__":
    main()
