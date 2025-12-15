# Copyright (c) 2025 Matt Hanson
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
make_scholar_video_bulletproof.py
Author: Matt Hanson + ChatGPT (GPT-5)
Description:
    Fully automatic pipeline to create a talking-head video using SadTalker,
    with expressive British scholar narration and optional ambient audio.
    SADTALKER path example: C:\Users\mhans\anaconda3\envs\sadtalker\SadTalker
"""

import os
import sys
import subprocess
from pathlib import Path
from gtts import gTTS

# ====================================================
# CONFIGURATION
# ====================================================
TEXT_LINE = (
    "The Broncos had better learn to score in the first half of the game, honestly mate!"
)
IMAGE_NAME = "matty_pimp_daddy.png"      # your uploaded photo
AUDIO_NAME = "scholar_line.mp3"
AMBIENT_TRACK = "ambient_light.mp3"      # optional background file
RESULT_VIDEO_NAME = "matty_scholar.mp4"
EXPRESSION_SCALE = 1.2
POSE_SCALE = 1.1
# ====================================================


def run_command(cmd, cwd=None):
    """Run a shell command safely with visible output."""
    print(f"\n>>> Running in {cwd or os.getcwd()}:\n{cmd}\n")
    result = subprocess.run(cmd, shell=True, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"❌ Command failed: {cmd}")


def find_sadtalker():
    """Try to find the SadTalker folder automatically."""
    candidates = [
        Path.cwd() / "SadTalker",
        Path.home() / "SadTalker",
        Path("C:/Users/mhans/anaconda3/envs/sadtalker/SadTalker"),
    ]
    for c in candidates:
        if (c / "inference.py").exists():
            print(f"✅ Found SadTalker at: {c}")
            return c
    sys.exit("❌ Could not locate SadTalker folder — please clone it and try again.")


def ensure_dirs(base_dir):
    """Ensure all input/output folders exist."""
    (base_dir / "inputs").mkdir(exist_ok=True)
    (base_dir / "results").mkdir(exist_ok=True)


def create_audio(inputs_dir):
    """Generate the scholar voice audio file."""
    audio_path = inputs_dir / AUDIO_NAME
    if not audio_path.exists():
        print("🎙 Generating British scholar voice...")
        tts = gTTS(TEXT_LINE, lang="en", tld="co.uk")
        tts.save(audio_path)
        print(f"✅ Saved voice to {audio_path}")
    else:
        print(f"ℹ️ Using existing audio file: {audio_path}")
    return audio_path


def patch_basicsr_imports(env_site_dir):
    """Patch basicsr if it references deprecated torchvision functions."""
    degradations_py = env_site_dir / "basicsr" / "data" / "degradations.py"
    if degradations_py.exists():
        content = degradations_py.read_text()
        if "functional_tensor" in content:
            print("🩹 Patching basicsr to use updated torchvision import...")
            new_content = content.replace(
                "from torchvision.transforms.functional_tensor import rgb_to_grayscale",
                "try:\n    from torchvision.transforms.functional import rgb_to_grayscale\nexcept ImportError:\n    from torchvision.transforms.functional_tensor import rgb_to_grayscale",
            )
            degradations_py.write_text(new_content)
            print("✅ basicsr patched successfully.")
    else:
        print("ℹ️ basicsr degradation file not found; skipping patch.")


def generate_video(sadtalker_dir, inputs_dir, results_dir):
    """Run SadTalker inference, auto-detecting supported args."""
    img_path = inputs_dir / IMAGE_NAME
    audio_path = inputs_dir / AUDIO_NAME
    if not img_path.exists():
        sys.exit(f"❌ Source image not found at {img_path}")
    if not audio_path.exists():
        sys.exit(f"❌ Audio not found at {audio_path}")

    # --- Auto-detect whether --pose_scale is supported ---
    help_txt = subprocess.run(
        "python inference.py -h",
        shell=True,
        cwd=sadtalker_dir,
        capture_output=True,
        text=True,
    )
    pose_flag = ""
    if "--pose_scale" in help_txt.stdout:
        pose_flag = f"--pose_scale {POSE_SCALE} "
        print("✅ SadTalker supports --pose_scale.")
    else:
        print("ℹ️ SadTalker version does not support --pose_scale; skipping it.")

    print("🎞 Generating expressive lip-synced video...")
    cmd = (
        f"python inference.py "
        f"--driven_audio \"{audio_path}\" "
        f"--source_image \"{img_path}\" "
        f"--enhancer gfpgan "
        f"--still "
        f"--preprocess full "
        f"--expression_scale {EXPRESSION_SCALE} "
        f"{pose_flag}"
        f"--result_dir \"{results_dir}\""
    )
    run_command(cmd, cwd=sadtalker_dir)


def add_ambient(results_dir, inputs_dir):
    """Optionally mix ambient background using ffmpeg."""
    generated_videos = list(results_dir.glob("*.mp4"))
    if not generated_videos:
        print("⚠️ No output video found to mix audio with.")
        return

    base_video = generated_videos[0]
    ambient = inputs_dir / AMBIENT_TRACK
    final_path = results_dir / f"final_{RESULT_VIDEO_NAME}"

    if ambient.exists():
        print("🔊 Adding ambient background...")
        cmd = (
            f'ffmpeg -y -i "{base_video}" -i "{ambient}" '
            f'-filter_complex "amix=inputs=2:duration=first:dropout_transition=3" '
            f'"{final_path}"'
        )
        run_command(cmd)
        print(f"🎧 Final output: {final_path}")
    else:
        print("ℹ️ No ambient track found, skipping mix.")


def main():
    base_dir = Path.cwd()
    ensure_dirs(base_dir)
    sadtalker_dir = find_sadtalker()
    inputs_dir = base_dir / "inputs"
    results_dir = base_dir / "results"

    # Auto-patch basicsr if needed
    site_dir = Path(sys.executable).parent / "Lib" / "site-packages"
    patch_basicsr_imports(site_dir)

    create_audio(inputs_dir)
    generate_video(sadtalker_dir, inputs_dir, results_dir)
    add_ambient(results_dir, inputs_dir)

    print("\n✅ All done! Check the 'results' folder for your new video.")


if __name__ == "__main__":
    main()

    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
