import argparse
import os
import torch
import torchaudio
from pathlib import Path
from src.metrics import SISNRi, SDRi, PESQ, STOI
from src.metrics.tracker import MetricTracker


formats = ["wav", "flac", "mp3"]

def fetch_correct_path(s_dir: Path, filestem: str):
    for format in formats:
        if (s_dir / f"{filestem}.{format}").exists():
            return s_dir / f"{filestem}.{format}"
    assert False, f"File not found: {s_dir / filestem}"


def load_audio(path, squeeze_channel=True):
    audio_tensor, _ = torchaudio.load(str(path))
    if squeeze_channel:
        audio_tensor = audio_tensor[0, :]
    else:
        audio_tensor = audio_tensor[0:1, :]
    return audio_tensor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--predictions_dir', required=True, help="Path to dir with saved predictions from inference.py")
    parser.add_argument('--gt_dir', required=True, help='Path to dir with GT speaker waveforms')
    parser.add_argument('--mix_dir', required=True, help="Path to dir with mixes which were separated")
    parser.add_argument('--pesq', required=False, action="store_true", help="calculate PESQ metric")
    parser.add_argument('--stoi', required=False, action="store_true", help="Calculate STOI metric")
    parser.add_argument('--batch_size', type=int, default=50, help="Batch size for evalulation")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    s1_est_dir = Path(args.predictions_dir) / "s1"
    s2_est_dir = Path(args.predictions_dir) / "s2"

    s1_gt_dir = Path(args.gt_dir) / "s1"
    s2_gt_dir = Path(args.gt_dir) / "s2"

    mix_dir = Path(args.mix_dir)

    metrics = [
        SISNRi("SI-SNRi"),
        SDRi("SDRi")
    ]

    if args.pesq:
        metrics.append(
            PESQ("PESQ")
        )
    if args.stoi:
        metrics.append(
            STOI("STOI")
        )

    metrics_tracker = MetricTracker(*[met.name for met in metrics])
    data_dicts = []

    for format in formats:
        for s1_est_path in s1_est_dir.glob("*." + format):
            s2_est_path = fetch_correct_path(s2_est_dir, s1_est_path.stem)
            s1_gt_path = fetch_correct_path(s1_gt_dir, s1_est_path.stem)
            s2_gt_path = fetch_correct_path(s2_gt_dir, s1_est_path.stem)
            mix_path = fetch_correct_path(mix_dir, s1_est_path.stem)

            logits = torch.stack([
                load_audio(path)
                for path in [s1_est_path, s2_est_path]
            ],dim=0)
            audio_concat = torch.stack([
                load_audio(path)
                for path in [s1_gt_path, s2_gt_path]
            ],dim=0)
            audio_mix = load_audio(mix_path, squeeze_channel=False)
            data_dicts.append(
                {
                    "logits": logits,
                    "audio_concat": audio_concat,
                    "audio_mix": audio_mix
                }
            )

            if len(data_dicts) == args.batch_size:
                batch = {
                    name: torch.stack([elem[name] for elem in data_dicts], dim=0).to(device)
                    for name in ["logits", "audio_concat", "audio_mix"]
                }
                for met in metrics:
                    metrics_tracker.update(met.name, met(**batch))
                data_dicts = []
    
    if len(data_dicts) > 0:
        batch = {
            name: torch.stack([elem[name] for elem in data_dicts], dim=0)
            for name in ["logits", "audio_concat", "audio_mix"]
        }
        for met in metrics:
            metrics_tracker.update(met.name, met(**batch))
    
    for name, val in metrics_tracker.result().items():
        print(f"    {name:15s}: {val}")


if __name__ == "__main__":
    main()