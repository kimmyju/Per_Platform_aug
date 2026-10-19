#!/usr/bin/env python3
"""이미지 한 장을 오프라인·온라인 증강에 넣고 결과를 저장한다.

원본 증강 코드는 수정하지 않고 `offline_augment.py`, `online_augment.py`를 호출한다.
두 경로의 샘플링 키를 파일 이름으로 맞춰, 같은 시드·플랫폼·화질이면 같은 레시피가 나온다.

예:
    python run_one_image.py
    python run_one_image.py --input celeb_1.png --output . --platform facebook --quality low

온라인 CLI는 증강된 픽셀을 디코딩한 뒤 출력 경로에 JPEG로 한 번 더 저장한다.
그래서 해상도와 레시피가 같아도 오프라인 파일과 바이트 크기는 다를 수 있다.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser(description='이미지 한 장을 오프라인·온라인으로 증강하고 결과를 저장')
    ap.add_argument('--input', type=Path, default=ROOT / 'celeb_1.png', help='입력 이미지 (기본: ./celeb_1.png)')
    ap.add_argument('--output', type=Path, default=Path('.'), help='결과 저장 디렉터리 (기본: 현재 디렉터리)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--epoch', type=int, default=0)
    ap.add_argument('--platform', type=str, default='facebook', help='강제 플랫폼. 자연 샘플링은 --no-force')
    ap.add_argument('--quality', type=str, default='low', help='강제 화질. 자연 샘플링은 --no-force')
    ap.add_argument('--no-force', action='store_true', help='플랫폼·화질을 섞어서 샘플링')
    return ap.parse_args()


def run(cmd, *, cwd: Path) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, cwd=cwd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout)
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    if proc.stderr:
        sys.stderr.write(proc.stderr)
    return proc


def main():
    args = parse_args()
    src = args.input.expanduser().resolve()
    if not src.is_file():
        raise SystemExit(f'입력 이미지가 없습니다: {src}')
    out_dir = args.output.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    online_cmd = [
        sys.executable, str(ROOT / 'online_augment.py'), 'demo-image',
        '--input', src.name,
        '--output', str(out_dir / f'{src.stem}_online.jpg'),
        '--seed', str(args.seed),
        '--epoch', str(args.epoch),
    ]
    if not args.no_force:
        online_cmd += ['--platform', args.platform, '--quality', args.quality]

    with tempfile.TemporaryDirectory() as tmp:
        staged = Path(tmp) / src.name
        shutil.copy2(src, staged)
        offline_cmd = [
            sys.executable, str(ROOT / 'offline_augment.py'),
            '--input', str(Path(tmp)),
            '--output', str(out_dir),
            '--copies', '1',
            '--seed', str(args.seed),
            '--epoch', str(args.epoch),
        ]
        if not args.no_force:
            offline_cmd += ['--force-platform', args.platform, '--force-quality', args.quality]
        print('===== OFFLINE =====')
        off = run(offline_cmd, cwd=ROOT)
        print(off.stdout, end='')

    print('===== ONLINE =====')
    on = run(online_cmd, cwd=src.parent)
    online_info = json.loads(on.stdout)
    online_json = out_dir / f'{src.stem}_online.json'
    online_json.write_text(json.dumps(online_info, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(on.stdout, end='')

    manifest_path = out_dir / 'augmentation_manifest.jsonl'
    manifest = json.loads(manifest_path.read_text(encoding='utf8').splitlines()[-1])
    offline_image = Path(manifest['output'])
    online_image = out_dir / f'{src.stem}_online.jpg'
    print('===== SAVED =====')
    print(f'offline image: {offline_image}')
    print(f'offline manifest: {manifest_path}')
    print(f'online image: {online_image}')
    print(f'online info: {online_json}')
    print(
        f"recipe offline={manifest.get('recipe_id')} online={online_info.get('recipe_id')} "
        f"platform offline={manifest.get('platform')} online={online_info.get('platform')}"
    )


if __name__ == '__main__':
    main()
