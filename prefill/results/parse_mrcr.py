import glob
import json
import os


def aggregate_scores(model_name, level, root_dir="./results/mrcr"):
    # 1. Define the pattern to find the specific files in the specific folders
    # Matches: [Any Prefix]_qwen2.5... / output-adakv-layer.json
    folder_pattern = f"*_{model_name}"
    filename = f"output-{level}.json"
    search_path = os.path.join(root_dir, folder_pattern, filename)

    # Find all matching files
    files = glob.glob(search_path)

    if not files:
        print(f"No files found matching pattern: {search_path}")
        return

    print(f"Found {len(files)} files. Processing...")

    # Dictionary to store scores: {'1.0': [0.0376, 0.04...], '0.75': [...]}
    score_aggregator = {}

    # 2. Iterate through files and collect scores
    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

                for ratio_key, content in data.items():
                    # Ensure the key contains a score before trying to access it
                    if isinstance(content, dict) and "score" in content:
                        score = content["score"]

                        if ratio_key not in score_aggregator:
                            score_aggregator[ratio_key] = []

                        score_aggregator[ratio_key].append(score)

        except (json.JSONDecodeError, IOError) as e:
            print(f"Error reading {file_path}: {e}")

    # 3. Calculate Averages
    averaged_results = {}

    for ratio, scores in score_aggregator.items():
        if scores:
            avg_score = sum(scores) / len(scores) * 100
            averaged_results[ratio] = round(avg_score, 2)  # Rounding for readability

    return averaged_results


if __name__ == "__main__":
    import argparse

    from results.parse import get_eviction_level

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="qwen2.5-7b-instruct-1m_fastkvzip_chunk16k_w4096",
    )
    parser.add_argument("-l", "--level", type=str, default="")
    args = parser.parse_args()

    if args.level == "":
        args.level = get_eviction_level(args.model)

    results = aggregate_scores(args.model, args.level)

    # Print results nicely formatted
    print("\n--- Averaged Scores per Ratio ---")
    print(json.dumps(results, indent=4))
