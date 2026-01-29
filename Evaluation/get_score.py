import json
import argparse
import os


def get_score(obj):
    """Get score from either auto_judge.score or score field"""
    if "auto_judge" in obj:
        return obj["auto_judge"].get("score", 0)
    return obj.get("score", 0)


def calc_pass1(path, verbose=True):
    """Calculate pass@1 for a single JSONL file"""
    total = 0
    correct = 0

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            score = get_score(obj)

            total += 1
            if score == 1:
                correct += 1

    pass1 = correct / total if total > 0 else 0

    if verbose:
        print(f"\nFile: {path}")
        print(f"  Total: {total}")
        print(f"  Correct: {correct}")
        print(f"  pass@1: {pass1:.4f}")

    return pass1


def load_scores_from_file(path):
    """Load scores from a single JSONL file, return {question: score} dictionary"""
    scores = {}
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            question = obj.get("question", obj.get("prompt", ""))
            if not question:
                raise ValueError(f"Missing 'question' or 'prompt' field in {path}")
            score = get_score(obj)
            scores[question] = score
    return scores


def calc_passk(folder):
    """Calculate pass@k for all JSONL files in a folder, and print pass@1 for each file"""
    # Find all jsonl files
    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".jsonl")
    ])

    if len(files) == 0:
        print("No jsonl files found in folder.")
        return

    print("Found files:")
    for f in files:
        print(" ", f)

    # Print pass@1 for each file
    print("\n=== pass@1 for each file ===")
    for f in files:
        calc_pass1(f, verbose=True)

    # Load scores from all files, each returns {question: score} dictionary
    all_scores = [load_scores_from_file(f) for f in files]
    k = len(all_scores)

    # Get all questions
    all_questions = set(all_scores[0].keys())
    
    # Ensure all files contain the same set of questions
    for i, scores in enumerate(all_scores):
        if set(scores.keys()) != all_questions:
            missing = all_questions - set(scores.keys())
            extra = set(scores.keys()) - all_questions
            error_msg = f"File {files[i]} has different questions!\n"
            if missing:
                error_msg += f"  Missing: {len(missing)} questions\n"
                # print(missing)
            if extra:
                error_msg += f"  Extra: {len(extra)} questions\n"
                print(extra)
            raise ValueError(error_msg)

    n_samples = len(all_questions)

    # Calculate pass@k: for each question, check if at least one file got it correct
    passed = 0
    for question in all_questions:
        if any(all_scores[j][question] == 1 for j in range(k)):
            passed += 1

    passk = passed / n_samples if n_samples else 0

    print("\n=== pass@k summary ===")
    print(f"Samples: {n_samples}")
    print(f"Correct (at least one success): {passed}")
    print(f"pass@{k}: {passk:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to a JSONL file")
    parser.add_argument("--folder", help="Path to a folder containing JSONL files")
    args = parser.parse_args()

    if args.file and args.folder:
        print("Error: Please provide either --file or --folder, not both.")
        exit(1)

    if args.file:
        calc_pass1(args.file)

    elif args.folder:
        calc_passk(args.folder)

    else:
        print("Error: Please specify --file or --folder")
