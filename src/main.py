# main.py
import argparse
from scripts import run_detection, run_occlusion, run_full_insertion

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", choices=["all", "detect", "occlusion"], default="all")
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="outputs/")
    args = parser.parse_args()

    if args.step == "detect" or args.step == "all":
        run_detection.run(video_path=args.video_path, output_dir=args.output_dir)

    if args.step == "occlusion" or args.step == "all":
        run_occlusion.run(output_dir=args.output_dir)

    if args.step == "all":
        run_full_insertion.run(output_dir=args.output_dir)

if __name__ == "__main__":
    main()
