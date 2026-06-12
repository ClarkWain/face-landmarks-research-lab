"""Extract 80k more CelebA images (020001..100000) from the full zip into celeba_ssl_20k folder."""
from __future__ import annotations

import sys
import zipfile
from pathlib import Path

from tqdm import tqdm


def main() -> None:
    src = Path("C:/Users/Administrator/Downloads/img_align_celeba.zip")
    dst_dir = Path("data/celeba_ssl_20k/img_align_celeba")
    dst_dir.mkdir(parents=True, exist_ok=True)

    start = 20001
    end = 100000

    with zipfile.ZipFile(src) as archive:
        for idx in tqdm(range(start, end + 1)):
            name = f"img_align_celeba/{idx:06d}.jpg"
            target = dst_dir / f"{idx:06d}.jpg"
            if target.exists():
                continue
            try:
                with archive.open(name) as f, open(target, "wb") as out:
                    out.write(f.read())
            except KeyError:
                print(f"missing {name}")
    total = len(list(dst_dir.glob("*.jpg")))
    print(f"total jpgs in {dst_dir}: {total}")


if __name__ == "__main__":
    main()
