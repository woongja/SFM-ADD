#!/usr/bin/env python3
import os, re, argparse, shutil, glob, json
from pathlib import Path
import yaml, pandas as pd
from tqdm import tqdm

from augmentation import AutoTuneAugmentor  # 전용

SPLIT_PAT = re.compile(r"ASVspoof2019_LA_(train|dev|eval)", re.IGNORECASE)

def infer_subset_from_path(p: str) -> str:
    m = SPLIT_PAT.search(p)
    if m: return m.group(1).lower()
    lp = p.lower()
    for s in ("train", "dev", "eval"):
        if f"/{s}/" in lp: return s
    return "train"

SUBSET_SET = {"train","dev","eval"}

def parse_protocol_line(line: str):
    """
    지원 포맷:
      1) ASV19:  <speaker> <fullpath> <label2>
      2) DF21 :  <subset>  <fullpath> <label2>   (subset ∈ {train,dev,eval})
    return: (speaker, fullpath, label2, subset_hint or None)
    """
    s = line.strip()
    if not s or s.startswith("#"): return None
    parts = s.split()
    if len(parts) < 3: return None

    a, b, c = parts[0], parts[1], parts[2]
    if a.lower() in SUBSET_SET:
        subset_hint = a.lower()
        speaker = "_"              # DF21은 speaker 미지정 → placeholder
        fullpath = b
        lab = c
    else:
        subset_hint = None
        speaker = a
        fullpath = b
        lab = c
    return speaker, fullpath, lab, subset_hint

def ensure_dirs(root: Path):
    for subset in ("train","dev","eval"):
        (root/subset/"clean").mkdir(parents=True, exist_ok=True)
        (root/subset/"augmented").mkdir(parents=True, exist_ok=True)

def collect_protocol_files(protocols, protocol_dir, pattern):
    files = []
    if protocols:
        for p in protocols: files.append(Path(p))
    if protocol_dir:
        pat = pattern or "*.txt"
        files.extend([Path(x) for x in glob.glob(str(Path(protocol_dir)/pat))])
    if not files:
        raise ValueError("No protocol files found. Use --protocol or --protocol-dir.")
    return files

def export_clean_wav(src_path: str, dst_wav: Path, sr: int, overwrite: bool, skip_existing: bool):
    """Decode any input (e.g., FLAC) and export as mono WAV with target sr."""
    try:
        if dst_wav.exists():
            if overwrite:
                dst_wav.unlink()
            elif skip_existing:
                return  # keep existing
        if (not dst_wav.exists()) or overwrite:
            from pydub import AudioSegment
            seg = AudioSegment.from_file(src_path)
            if seg.frame_rate != sr:
                seg = seg.set_frame_rate(sr)
            # 필요 시 스테레오 유지하려면 아래 줄 주석
            seg = seg.set_channels(1)
            seg.export(str(dst_wav), format="wav")
    except Exception as e:
        print(f"[WARN] Clean export failed for {src_path}: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", nargs="*", help="one or more protocol.txt files")
    ap.add_argument("--protocol-dir", help="directory containing protocol files")
    ap.add_argument("--protocol-pattern", help="glob pattern inside --protocol-dir (default: *.txt)")
    ap.add_argument("--out-root", required=True, help="output root dir")
    ap.add_argument("--aug-config", required=True, help="YAML config path (must include 'auto_tune')")
    ap.add_argument("--sr", type=int, default=16000, help="export sample rate (default 16000)")
    ap.add_argument("--copy-clean", action="store_true", help="also copy clean into out-root/subset/clean")
    ap.add_argument("--force-subset", choices=["train","dev","eval"], help="force subset for all lines")
    ap.add_argument("--meta-out", default="meta_noise_autotune.csv", help="metadata CSV filename (under out-root)")
    # --- 추가 옵션 ---
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs if present")
    ap.add_argument("--skip-existing", action="store_true", help="Skip processing if target file already exists")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    ensure_dirs(out_root)

    with open(args.aug_config, "r") as f:
        cfg_all = yaml.safe_load(f)
    if "auto_tune" not in cfg_all:
        raise ValueError("YAML must contain 'auto_tune' section.")
    base_cfg = dict(cfg_all["auto_tune"])
    base_cfg.setdefault("target_sr", args.sr)

    meta_path = out_root / args.meta_out
    meta_exists = meta_path.exists()
    all_records = []

    prot_files = collect_protocol_files(args.protocol, args.protocol_dir, args.protocol_pattern)

    for pfile in prot_files:
        lines = [ln for ln in open(pfile, "r", encoding="utf-8") if ln.strip() and not ln.startswith("#")]
        for ln in tqdm(lines, desc=f"AutoTune {pfile.name}", total=len(lines)):
            parsed = parse_protocol_line(ln)
            if not parsed:
                continue
            spk, clean_flac, lab, subset_hint = parsed

            # subset 결정 우선순위: --force-subset > subset_hint(DF21) > 경로추론
            subset = args.force_subset or subset_hint or infer_subset_from_path(clean_flac)

            clean_stem = Path(clean_flac).stem
            out_clean = out_root / subset / "clean" / f"{clean_stem}.wav"
            out_aug  = out_root / subset / "augmented" / f"{clean_stem}__auto_tune.wav"
            out_aug.parent.mkdir(parents=True, exist_ok=True)

            # 메타(클린)
            all_records.append({
                "subset": subset,
                "speaker": spk,
                "src_path": clean_flac,
                "file_path": str(out_clean),
                "label2": lab,
                "label1": "clean",
                "mode": "-",
                "snr_db": "-",
                "params_json": "-",
                "samplerate": args.sr,
                "format": "wav",
            })

            # 필요 시 clean WAV 생성
            if args.copy_clean:
                export_clean_wav(clean_flac, out_clean, args.sr, args.overwrite, args.skip_existing)

            # ---- AutoTune 증강 ----
            try:
                if out_aug.exists():
                    if args.overwrite:
                        out_aug.unlink()
                    elif args.skip_existing:
                        # 메타만 추가하고 스킵
                        all_records.append({
                            "subset": subset,
                            "speaker": spk,
                            "src_path": clean_flac,
                            "file_path": str(out_aug),
                            "label2": lab,
                            "label1": "auto_tune",
                            "mode": "-",
                            "snr_db": "-",
                            "params_json": "-",
                            "samplerate": args.sr,
                            "format": "wav",
                        })
                        continue

                # ✅ clean WAV가 있으면 그걸 입력으로 사용(디코딩/속도 이득)
                src_for_aug = str(out_clean) if out_clean.exists() else clean_flac

                cfg = dict(base_cfg)
                cfg["output_path"] = str(out_aug)
                cfg["out_format"] = "wav"

                augmentor = AutoTuneAugmentor(cfg)
                augmentor.load(src_for_aug)
                augmentor.transform()
                augmentor.augmented_audio.export(str(out_aug), format="wav")

                params = getattr(augmentor, "params", None)
                all_records.append({
                    "subset": subset,
                    "speaker": spk,
                    "src_path": clean_flac,   # 원본 경로를 src_path로 유지
                    "file_path": str(out_aug),
                    "label2": lab,
                    "label1": "auto_tune",
                    "mode": getattr(augmentor, "mode", "-"),
                    "snr_db": getattr(augmentor, "snr_db", "-"),
                    "params_json": "-" if params is None else json.dumps(params, ensure_ascii=False),
                    "samplerate": args.sr,
                    "format": "wav",
                })
            except Exception as e:
                print(f"[ERROR] AutoTune failed for {clean_flac}: {e}")

    df_new = pd.DataFrame(all_records)
    if meta_exists:
        df_old = pd.read_csv(meta_path)
        df_all = pd.concat([df_old, df_new], ignore_index=True)
        df_all.to_csv(meta_path, index=False)
    else:
        df_new.to_csv(meta_path, index=False)

    print(f"[DONE] saved metadata: {meta_path} (+{len(df_new)} rows)")
    print(f"out-root: {out_root}")

if __name__ == "__main__":
    main()
