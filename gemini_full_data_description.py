import json
import os
from pathlib import Path
from google import genai


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = Path("/home/student01/Downloads/data.json")

OUTPUT_FILE = Path(
    "/home/student01/Downloads/gemini_full_video_description.json"
)

MODEL_NAME = "gemini-2.5-flash"


# ============================================================
# GEMINI PROMPT
# ============================================================

SYSTEM_PROMPT = """
You are an expert in temporal video understanding and cooking-process
analysis.

You are given structured frame-by-frame analysis of a cooking video.

The frames are presented in chronological order.

For every frame, the data may contain:

1. frame number
2. timestamp
3. cooking action
4. visible food ingredients
5. actively used utensils
6. independent visual description of the current frame
7. rolling context summarizing previous events
8. context-aware description of the current frame

Your task is to reconstruct a highly detailed, coherent description of
the ENTIRE cooking video.

IMPORTANT:

- Treat the frame order as chronological.
- Do NOT simply concatenate the frame descriptions.
- Synthesize the information across frames.
- Preserve the temporal sequence of events.
- Describe what happens from the beginning of the video to the end.
- Track ingredients as they appear and are incorporated into the dish.
- Track important changes in the state of the food.
- Track cooking actions such as mixing, stirring, chopping, pouring,
  heating, frying, baking, kneading, etc.
- Track utensils when they are relevant to understanding the process.
- Identify transitions between different stages of preparation.
- Use the independent frame descriptions for visual details.
- Use the context-aware descriptions to understand continuity between
  frames.
- Use the structured action, ingredient, and utensil information as
  additional evidence.
- Avoid repeating the same event merely because it appears in many
  consecutive frames.
- However, preserve meaningful small changes and intermediate steps.
- Do not invent ingredients, actions, or events.
- Do not infer an entire recipe merely from what you think the dish is.
- If the evidence is uncertain or contradictory, acknowledge the
  uncertainty rather than inventing information.
- Distinguish clearly between things that are directly visible and things
  that are reasonably inferred.
- Do not focus on irrelevant details such as camera composition,
  countertops, clothing, or background objects unless they are important
  to understanding the cooking process.

The final answer should be a highly detailed chronological narrative of
the cooking process.

It should read as though someone carefully watched the entire cooking
video and is explaining exactly what happened, step by step.

Include fine-grained temporal progression when supported by the data.

Do not mention that the information came from frames, a VLM, Qwen,
JSON, or another model.

Return ONLY the final cooking-video description.
"""


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Could not find data file: {DATA_FILE}"
        )

    with open(DATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================
# FORMAT ONE FRAME
# ============================================================

def format_frame(frame, fps):
    """
    Convert one frame from data.json into a compact textual representation.
    """

    frame_number = frame["frame_number"]
    timestamp = frame_number / fps

    questions = frame.get("questions", [])

    action = ""
    ingredients = ""
    utensils = ""
    independent_description = ""

    # --------------------------------------------------------
    # Current data.json structure:
    #
    # questions[0] = action
    # questions[1] = ingredients
    # questions[2] = utensils
    # questions[3] = independent description
    # --------------------------------------------------------

    if len(questions) > 0:
        action = questions[0].get("answer", "").strip()

    if len(questions) > 1:
        ingredients = questions[1].get("answer", "").strip()

    if len(questions) > 2:
        utensils = questions[2].get("answer", "").strip()

    if len(questions) > 3:
        independent_description = questions[3].get(
            "answer", ""
        ).strip()

    rolling_context = frame.get(
        "rolling_context", ""
    ).strip()

    context_description = frame.get(
        "context_description", ""
    ).strip()

    return f"""
============================================================
FRAME {frame_number} | {timestamp:.2f} seconds
============================================================

COOKING ACTION:
{action}

VISIBLE INGREDIENTS:
{ingredients}

ACTIVELY USED UTENSILS:
{utensils}

INDEPENDENT FRAME DESCRIPTION:
{independent_description}

ROLLING CONTEXT:
{rolling_context}

CONTEXT-AWARE FRAME DESCRIPTION:
{context_description}
""".strip()


# ============================================================
# BUILD COMPLETE VIDEO INPUT
# ============================================================

def build_video_input(video):
    """
    Convert the complete video entry from data.json into a
    chronological textual representation.
    """

    fps = video["fps"]

    video_name = video["video_name"]

    frames = video["frames"]

    print(f"Video: {video_name}")
    print(f"FPS: {fps}")
    print(f"Frames: {len(frames)}")

    formatted_frames = []

    for frame in frames:
        formatted_frames.append(
            format_frame(frame, fps)
        )

    return f"""
VIDEO: {video_name}
FPS: {fps}
NUMBER OF SAMPLED FRAMES: {len(frames)}

The following frame analyses are ordered chronologically.

{chr(10).join(formatted_frames)}
""".strip()


# ============================================================
# GENERATE GEMINI DESCRIPTION
# ============================================================

def generate_video_description(video, client):
    video_name = video["video_name"]

    print("\n")
    print("=" * 80)
    print(f"PROCESSING VIDEO: {video_name}")
    print("=" * 80)

    video_input = build_video_input(video)

    prompt = SYSTEM_PROMPT + "\n\n" + video_input

    print("\nSending complete frame-level data to Gemini...")
    print(f"Input characters: {len(prompt):,}")

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=prompt,
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    description = response.text.strip()

    print("\n")
    print("=" * 80)
    print("GEMINI VIDEO DESCRIPTION")
    print("=" * 80)
    print(description)
    print("=" * 80)

    return {
        "video_name": video_name,
        "model": MODEL_NAME,
        "description": description,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # Check API key
    # --------------------------------------------------------

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable is not set.\n"
            "Run:\n"
            'export GEMINI_API_KEY="YOUR_API_KEY"'
        )

    # --------------------------------------------------------
    # Create Gemini client
    # --------------------------------------------------------

    client = genai.Client(api_key=api_key)

    # --------------------------------------------------------
    # Load data.json
    # --------------------------------------------------------

    data = load_data()

    print(f"Found {len(data)} video(s) in data.json.")

    results = []

    # --------------------------------------------------------
    # Process every video
    # --------------------------------------------------------

    for video in data:

        result = generate_video_description(
            video,
            client
        )

        results.append(result)

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
            ensure_ascii=False
        )

    print("\n")
    print("=" * 80)
    print("DONE")
    print("=" * 80)
    print(f"Results saved to:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
