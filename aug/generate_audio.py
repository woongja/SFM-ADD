#!/usr/bin/env python3
import os, re, argparse, shutil, glob, json
from pathlib import Path
import yaml, pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import random
from collections import defaultdict


# --- 너의 augmentation 모듈들 ---
from augmentation import (
    BackgroundNoiseAugmentorDeepen as BackgroundNoiseAugmentor,
    BackgroundMusicAugmentorDeepen as BackgroundMusicAugmentor,
    GaussianAugmentorV1,
    HighPassFilterAugmentor,
    LowPassFilterAugmentor,
    PitchAugmentor,
    TimeStretchAugmentor,
    EchoAugmentorDeepen as EchoAugmentor,
    ReverbAugmentor,
)

AUGMENTATION_CLASSES = {
    "background_noise": BackgroundNoiseAugmentor,
    "background_music": BackgroundMusicAugmentor,
    "gaussian_noise": GaussianAugmentorV1,
    "high_pass_filter": LowPassFilterAugmentor.__class__,  # placeholder (아래에서 덮어씀)
    "low_pass_filter": LowPassFilterAugmentor,
    "pitch_shift": PitchAugmentor,
    "time_stretch": TimeStretchAugmentor,
    "echo": EchoAugmentor,
    "reverberation": ReverbAugmentor,
}
AUGMENTATION_CLASSES["high_pass_filter"] = HighPassFilterAugmentor  # 명시 덮어쓰기
AUG_LIST_ALL = list(AUGMENTATION_CLASSES.keys())

SPLIT_PAT = re.compile(r"ASVspoof2019_LA_(train|dev|eval)", re.IGNORECASE)

def smart_tokenize(s: str):
    return [t for t in s.strip().split() if t]

def detect_path_and_subset(tokens):
    """
    다양한 프로토콜 라인을 견고하게 처리:
    - ASV19 통합형:  speaker path label2
    - DF21 eval형:  subset path label2
    - 기타: 토큰 중 경로처럼 보이는 걸 path로, subset은 토큰에 있으면 사용, 없으면 경로에서 추론
    """
    # path 토큰 탐색 ( / 포함 + 오디오 확장자 )
    path_idx = None
    for i, t in enumerate(tokens):
        tl = t.lower()
        if "/" in t and (tl.endswith(".flac") or tl.endswith(".wav")):
            path_idx = i
            break
    if path_idx is None:
        return None, None  # 못 찾으면 스킵

    # subset 후보 (명시)
    subset = None
    for t in tokens:
        tl = t.lower()
        if tl in ("train","dev","eval"):
            subset = tl
            break
    # 경로에서 추론
    if subset is None:
        lp = tokens[path_idx].lower()
        m = re.search(r"asvspoof2019_la_(train|dev|eval)", lp)
        if m: subset = m.group(1)
        else:
            # 폴더명에 train/dev/eval이 있으면 사용
            mm = SPLIT_PAT.search(lp)
            subset = (mm.group(1).lower() if mm else "train")

    path = tokens[path_idx]
    return path, subset

def assign_lines_balanced(lines, aug_list, seed=42, cap_per_aug=None):
    rng = random.Random(seed)
    shuffled = lines[:]
    rng.shuffle(shuffled)
    N, K = len(shuffled), len(aug_list)
    if K == 0 or N == 0: return {a: [] for a in aug_list}
    base, rem = N // K, N % K
    targets = [base + (1 if i < rem else 0) for i in range(K)]
    if cap_per_aug is not None:
        targets = [min(t, cap_per_aug) for t in targets]
    out, idx = {}, 0
    for i, a in enumerate(aug_list):
        take = max(0, min(targets[i], N - idx))
        out[a] = shuffled[idx:idx+take]
        idx += take
    return out

def infer_subset_from_path(p: str) -> str:
    m = SPLIT_PAT.search(p)
    if m: return m.group(1).lower()
    lp = p.lower()
    for s in ("train", "dev", "eval"):
        if f"/{s}/" in lp: return s
    return "train"

def parse_protocol_line(line: str):
    s = line.strip()
    if not s or s.startswith("#"): return None
    parts = s.split()
    if len(parts) < 3: return None
    spk, fullpath, lab = parts[0], parts[1], parts[2]
    return spk, fullpath, lab

def ensure_dirs(root: Path):
    for subset in ("train","dev","eval"):
        (root/subset/"clean").mkdir(parents=True, exist_ok=True)
        (root/subset/"augmented").mkdir(parents=True, exist_ok=True)

def collect_protocol_files(protocols, protocol_dir, pattern):
    files = []
    if protocols:
        for p in protocols:
            files.append(Path(p))
    if protocol_dir:
        pat = pattern or "*.txt"
        files.extend([Path(x) for x in glob.glob(str(Path(protocol_dir)/pat))])
    if not files:
        raise ValueError("No protocol files found. Use --protocol or --protocol-dir.")
    return files

def process_single_line(args_tuple):
    """
    Process a single protocol line - designed for multiprocessing

    args_tuple =
      (
        line,                   # str: one protocol line
        aug_list,               # list[str]: enabled augmentations for this run (after --exclude)
        AUG_CONFIG,             # dict: yaml-loaded config
        AUGMENTATION_CLASSES,   # dict[str, class]
        sr,                     # int: target sample rate
        out_root,               # Path: output root
        copy_clean,             # bool: copy/convert clean (if True)
        force_subset,           # Optional[str]: 'train'/'dev'/'eval' or None
        overwrite,              # bool: overwrite outputs if exist
        skip_existing,          # bool: skip if output exists
        skip_clean_meta,        # bool: do NOT write clean meta/copy in this run
        clean_only,             # bool: only do clean meta/copy; skip all augmentations
      )
    """
    (line, aug_list, AUG_CONFIG, AUGMENTATION_CLASSES, sr, out_root,
     copy_clean, force_subset, overwrite, skip_existing,
     skip_clean_meta, clean_only) = args_tuple

    parsed = parse_protocol_line(line)
    if not parsed:
        return []

    spk, clean_flac, lab = parsed
    subset = (force_subset or infer_subset_from_path(clean_flac))

    clean_stem = Path(clean_flac).stem
    out_clean = out_root / subset / "clean" / f"{clean_stem}.wav"

    records = []

    # ---------- CLEAN META / COPY ----------
    if not skip_clean_meta:
        # meta row (clean)
        records.append({
            "subset": subset,
            "speaker": spk,
            "src_path": clean_flac,
            "file_path": str(out_clean),
            "label2": lab,
            "label1": "clean",
            "mode": "-",
            "snr_db": "-",
            "params_json": "-",
            "samplerate": sr,
            "format": "wav",
        })

        if copy_clean:
            # 기본 구현: 원본을 그대로 카피 (확장자만 .wav로 바뀌는게 싫다면 변환 함수로 교체)
            try:
                if out_clean.exists():
                    if overwrite:
                        out_clean.unlink()
                    elif skip_existing:
                        pass
                if (not out_clean.exists()) or overwrite:
                    # 필요 시 실제 디코드/리샘플을 하려면 export_clean_wav로 바꿔도 됨
                    shutil.copy2(clean_flac, out_clean)
            except Exception as e:
                print(f"[WARN] clean copy failed for {clean_flac} -> {out_clean}: {e}")

    # clean 전용 모드면 여기서 종료
    if clean_only:
        return records

    # ---------- AUGMENTATIONS ----------
    for aug_name in aug_list:
        # 방어: 혹시 'clean'이 aug_list에 들어와도 무시
        if aug_name == "clean":
            continue

        # 대상 파일 경로
        out_aug = out_root / subset / "augmented" / f"{clean_stem}__{aug_name}.wav"
        out_aug.parent.mkdir(parents=True, exist_ok=True)

        # 존재 시 처리 정책
        if out_aug.exists():
            if overwrite:
                try:
                    out_aug.unlink()
                except Exception as e:
                    print(f"[WARN] overwrite remove failed: {out_aug} ({e})")
            elif skip_existing:
                # 메타만 추가하고 스킵
                records.append({
                    "subset": subset,
                    "speaker": spk,
                    "src_path": clean_flac,
                    "file_path": str(out_aug),
                    "label2": lab,
                    "label1": aug_name,
                    "mode": "-",
                    "snr_db": "-",
                    "params_json": "-",
                    "samplerate": sr,
                    "format": "wav",
                })
                continue

        # 증강 실행
        try:
            aug_class = AUGMENTATION_CLASSES[aug_name]
        except KeyError:
            print(f"[WARN] Unknown augmentation '{aug_name}', skipping.")
            continue

        cfg = dict(AUG_CONFIG.get(aug_name, {}))
        cfg.setdefault("target_sr", sr)
        cfg["output_path"] = str(out_aug)
        cfg["out_format"] = "wav"

        try:
            augmentor = aug_class(cfg)
            # 입력 소스: 기본은 원본 경로 사용
            # (clean WAV로 증강하고 싶으면: src_for_aug = str(out_clean) if out_clean.exists() else clean_flac)
            src_for_aug = clean_flac

            augmentor.load(src_for_aug)
            augmentor.transform()
            augmentor.augmented_audio.export(str(out_aug), format="wav")

            params = getattr(augmentor, "params", None)
            records.append({
                "subset": subset,
                "speaker": spk,
                "src_path": clean_flac,
                "file_path": str(out_aug),
                "label2": lab,
                "label1": aug_name,
                "mode": getattr(augmentor, "mode", "-"),
                "snr_db": getattr(augmentor, "snr_db", "-"),
                "params_json": "-" if params is None else json.dumps(params, ensure_ascii=False),
                "samplerate": sr,
                "format": "wav",
            })
        except Exception as e:
            print(f"[ERROR] processing failed: {clean_flac} with {aug_name}: {e}")

    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", nargs="*", help="one or more protocol.txt files")
    ap.add_argument("--protocol-dir", help="directory containing protocol files")
    ap.add_argument("--protocol-pattern", help="glob pattern inside --protocol-dir (default: *.txt)")
    ap.add_argument("--out-root", required=True, help="output root dir")
    ap.add_argument("--aug-config", required=True, help="YAML config path")
    ap.add_argument("--sr", type=int, default=16000, help="export sample rate (default 16000)")
    ap.add_argument("--copy-clean", action="store_true", help="also copy clean into out-root/subset/clean")
    ap.add_argument("--exclude", nargs="*", default=[], help="augmentation names to exclude")
    ap.add_argument("--force-subset",dest="force_subset",choices=["train","dev","eval"],help="force subset for all lines from these protocol(s)")
    ap.add_argument("--meta-out", default="meta_noise.csv", help="metadata CSV filename (created under out-root)")
    ap.add_argument("--num-workers", type=int, default=None, help="Number of parallel workers (default: CPU count)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present")
    ap.add_argument("--skip-existing", action="store_true", help="Skip if output already exists")
    # --- C 단계 추가 ---
    ap.add_argument("--dump-assignments-dir", help="If set, dump per-augmentation line lists (*.txt) here and exit")
    ap.add_argument("--include-auto-tune", action="store_true", help="Include auto_tune in augmentation list")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for augmentation assignment")
    ap.add_argument("--cap-per-aug", type=int, default=None, help="Optional hard cap per augmentation")
    ap.add_argument("--skip-clean-meta", action="store_true",help="Do not record/copy clean for this run (useful when running non-clean shards)")
    ap.add_argument("--clean-only", action="store_true",help="Only write/copy clean outputs; skip all augmentations")
    ap.add_argument("--include-clean", action="store_true",help="Include 'clean' as a pseudo-augmentation when dumping assignments")


    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dirs(out_root)

    with open(args.aug_config, "r") as f:
        AUG_CONFIG = yaml.safe_load(f)

    prot_files = collect_protocol_files(args.protocol, args.protocol_dir, args.protocol_pattern)

    all_lines = []
    for pfile in prot_files:
        lines = [ln for ln in open(pfile, "r", encoding="utf-8")
                 if ln.strip() and not ln.startswith("#")]
        all_lines.extend(lines)

    # -------------------------
    # C단계: dump 모드 처리
    # -------------------------
    if args.dump_assignments_dir:
        # 1) 증강 리스트 구성
        aug_list = [a for a in AUG_LIST_ALL if a not in set(args.exclude)]
        if args.include_auto_tune and "auto_tune" not in aug_list:
            aug_list.append("auto_tune")
        if args.include_clean and "clean" not in aug_list:
            aug_list.append("clean")

        print(f"[DUMP] total raw lines: {len(all_lines)}")
        print(f"[DUMP] aug_list: {aug_list}")

        # 2) subset별 라인 그룹핑
        subset_buckets = {"train": [], "dev": [], "eval": []}
        unresolved = 0
        for raw in all_lines:
            toks = smart_tokenize(raw)
            path, subset = detect_path_and_subset(toks)
            if path is None or subset not in subset_buckets:
                unresolved += 1
                continue
            # 덤프에는 '원문 라인' 그대로 저장해야 나중에 파서와 동일하게 동작
            subset_buckets[subset].append(raw)

        print(f"[DUMP] grouped: train={len(subset_buckets['train'])}, "
              f"dev={len(subset_buckets['dev'])}, eval={len(subset_buckets['eval'])}, "
              f"unresolved={unresolved}")

        # 3) subset별로 균등 분배 후 파일로 저장
        dump_dir = Path(args.dump_assignments_dir)
        dump_dir.mkdir(parents=True, exist_ok=True)
        total_dumped = 0

        summary_lines = []
        for subset, lines_bucket in subset_buckets.items():
            subdir = dump_dir / subset
            subdir.mkdir(parents=True, exist_ok=True)

            assign_map = assign_lines_balanced(
                lines=lines_bucket, aug_list=aug_list,
                seed=args.seed, cap_per_aug=args.cap_per_aug
            )

            for aug, lst in assign_map.items():
                outp = subdir / f"{aug}.txt"
                with open(outp, "w", encoding="utf-8") as f:
                    for ln in lst:
                        f.write(ln.rstrip() + "\n")
                print(f"[DUMP] {subset}/{aug}: {len(lst)} -> {outp}")
                summary_lines.append(f"{subset}/{aug}: {len(lst)}")
                total_dumped += len(lst)

        with open(dump_dir / "_summary.txt", "w", encoding="utf-8") as f:
            for s in summary_lines:
                f.write(s + "\n")
            f.write(f"TOTAL_DUMPED: {total_dumped}\n")
            f.write(f"TOTAL_LINES: {len(all_lines)}\n")

        print(f"[DUMP][DONE] dumped to: {dump_dir}")
        return
    # -------------------------

    aug_list = [a for a in AUG_LIST_ALL if a not in set(args.exclude)]

    meta_path = out_root / args.meta_out
    meta_exists = meta_path.exists()
    all_records = []

    num_workers = args.num_workers or cpu_count()
    print(f"Total lines to process: {len(all_lines)}")
    print(f"Using {num_workers} workers for parallel processing")

    worker_args = [
    (
        line,
        aug_list,
        AUG_CONFIG,
        AUGMENTATION_CLASSES,
        args.sr,
        out_root,
        args.copy_clean,
        args.force_subset,
        args.overwrite,
        args.skip_existing,
        args.skip_clean_meta,
        args.clean_only,
    )
    for line in all_lines
]

    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_single_line, worker_args),
            total=len(all_lines),
            desc="Processing audio files"
        ))

    for record_list in results:
        all_records.extend(record_list)

    df_new = pd.DataFrame(all_records)
    if meta_exists:
        df_old = pd.read_csv(meta_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(meta_path, index=False)
    else:
        df_new.to_csv(meta_path, index=False)

    print(f"[DONE] saved metadata: {meta_path}  (+{len(df_new)} rows)")
    print(f"out-root: {out_root}")


if __name__ == "__main__":
    main()
