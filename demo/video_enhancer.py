import os
import argparse
import subprocess
import pathlib

def run_cmd(cmd):
    print(">>> Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)

def main():
    parser = argparse.ArgumentParser(description="Low-light video enhancer")
    parser.add_argument("--video_path", type=str, required=True, help="Path to input low-light video")
    parser.add_argument("--weights", type=str, default="weight/Epoch99.pth",
                        help="Path to pretrained model weights (.pth file)")
    args = parser.parse_args()

    input_video = args.video_path
    input_stem = pathlib.Path(input_video).stem
    frames_input_dir = "data/test_data/Frames/"
    frames_output_dir = "data/test_output/Frames/"
    enhanced_video = f"enhanced_{input_stem}.mp4"

    # Make sure frame directories exist
    os.makedirs(frames_input_dir, exist_ok=True)
    os.makedirs(frames_output_dir, exist_ok=True)

    # 1. Extract frames from video
    run_cmd([
        "python", "demo/make_video.py",
        "--choice", "V2I",
        "--video_path", input_video,
        "--image_lowlight_folder", f"{frames_input_dir}%d.jpg"
    ])

    # 2. Enhance frames with test.py
    run_cmd([
        "python", "test.py",
        "--input_dir", frames_input_dir,
        "--test_dir", frames_output_dir,
        "--weight_dir", args.weights
    ])

    # 3. Reassemble enhanced frames into video
    run_cmd([
        "python", "demo/make_video.py",
        "--choice", "I2V",
        "--image_folder", frames_output_dir,
        "--save_path", enhanced_video,
        "--video_path", input_video
    ])

    print(f"\n✅ Done! Enhanced video saved at: {enhanced_video}")

if __name__ == "__main__":
    main()
