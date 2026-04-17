import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _load_epoch_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in ("arial.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _parse_epoch_from_path(path: Path) -> int | None:
    match = re.search(r"epoch_(\d+)", path.stem)
    if match is None:
        return None
    return int(match.group(1))


def _annotate_frame(path: Path, show_epoch: bool) -> Image.Image:
    with Image.open(path) as im:
        annotated = im.convert("RGB")

    if not show_epoch:
        return annotated

    epoch = _parse_epoch_from_path(path)
    if epoch is None:
        label = path.stem
    else:
        label = f"Epoch {epoch:04d}"

    draw = ImageDraw.Draw(annotated)
    font = _load_epoch_font(size=28)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]

    x = 12
    y = 12
    pad = 8
    draw.rectangle(
        (x - pad, y - pad, x + text_w + pad, y + text_h + pad),
        fill=(0, 0, 0),
    )
    draw.text((x, y), label, fill=(255, 255, 255), font=font)
    return annotated


def build_animation(
    frames_dir: str,
    out_path: str,
    duration_ms: int,
    show_epoch: bool,
):
    frame_paths = sorted(Path(frames_dir).glob("epoch_*.png"))
    if len(frame_paths) == 0:
        raise FileNotFoundError(f"No frame PNGs matching 'epoch_*.png' found in {frames_dir}")

    frames = [_annotate_frame(path, show_epoch).convert("P", palette=Image.ADAPTIVE) for path in frame_paths]
    first, rest = frames[0], frames[1:]
    first.save(
        out_path,
        save_all=True,
        append_images=rest,
        duration=duration_ms,
        loop=0,
    )
    for frame in frames:
        frame.close()


def parse_args():
    p = argparse.ArgumentParser(description="Create a GIF animation from saved GAN progression frames.")
    p.add_argument("--frames_dir", type=str, required=True, help="Directory containing progression frames named epoch_XXXX.png")
    p.add_argument("--out", type=str, default="", help="Output GIF path. Defaults to <frames_dir>/../generator_progression.gif")
    p.add_argument("--duration_ms", type=int, default=800, help="Frame duration in milliseconds (smaller = faster)")
    p.add_argument("--no_epoch_label", action="store_true", default=False, help="Do not draw epoch text on the frames in the GIF")
    return p.parse_args()


def main():
    args = parse_args()
    if args.duration_ms < 1:
        raise ValueError("--duration_ms must be >= 1")

    frames_dir = Path(args.frames_dir)
    if not frames_dir.is_dir():
        raise FileNotFoundError(frames_dir)

    if args.out.strip():
        out_path = Path(args.out)
    else:
        out_path = frames_dir.parent / "generator_progression.gif"

    build_animation(
        frames_dir=str(frames_dir),
        out_path=str(out_path),
        duration_ms=args.duration_ms,
        show_epoch=(not args.no_epoch_label),
    )
    print(f"Saved animation to: {out_path}")


if __name__ == "__main__":
    main()
