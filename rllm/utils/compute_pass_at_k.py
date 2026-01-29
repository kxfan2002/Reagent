import os
from collections import defaultdict

import torch


def compute_pass_at_k(results, k_values: list[int]):
    """
    计算 pass@1 和 pass@k 指标
    
    Args:
        results: 轨迹结果列表，每个轨迹需要有 task 和 reward 属性
        k_values: 要计算的 k 值列表，例如 [1, 3]
    
    Pass@1: 每一遍生成的正确率（例如30道题生成3遍，分别计算每遍的正确率）
    Pass@k: 每道题 k 次尝试中至少对 1 次就算对，这样的题占总题数的比例
    
    Returns:
        包含 pass@1 每轮结果、pass@k 结果、每个问题详情的字典
    """
    import hashlib
    import json

    # 按问题分组，记录每次尝试的结果
    problem_attempts: defaultdict[str, list[bool]] = defaultdict(list)
    problem_task_map: dict[str, str] = {}  # 问题哈希到原始任务的映射

    for trajectory in results:
        task = trajectory.task

        # Generate hash of problem dict/string
        if isinstance(task, dict):
            problem_str = json.dumps(task, sort_keys=True)
        else:
            problem_str = str(task)
        problem_hash = hashlib.md5(problem_str.encode()).hexdigest()

        is_correct = trajectory.reward > 0
        problem_attempts[problem_hash].append(is_correct)
        problem_task_map[problem_hash] = problem_str[:100]  # 保存前100字符用于标识

    total_problems = len(problem_attempts)
    n_rounds = max(len(attempts) for attempts in problem_attempts.values())

    # ========== 验证：确保每道题的尝试次数一致 ==========
    attempts_counts = [len(attempts) for attempts in problem_attempts.values()]
    if len(set(attempts_counts)) > 1:
        print(f"[警告] 各题目的尝试次数不一致: {set(attempts_counts)}")
    
    # ========== 计算 Pass@1: 每一遍的正确率 ==========
    pass_at_1_per_round = []
    for round_idx in range(n_rounds):
        correct_count = 0
        valid_count = 0
        for problem_hash, attempts in problem_attempts.items():
            if round_idx < len(attempts):
                valid_count += 1
                if attempts[round_idx]:
                    correct_count += 1
        if valid_count > 0:
            accuracy = correct_count / valid_count
            pass_at_1_per_round.append({
                "round": round_idx + 1,
                "correct": correct_count,
                "total": valid_count,
                "accuracy": accuracy,
            })
        # 验证每轮题目数
        if valid_count != total_problems:
            print(f"[警告] Round {round_idx + 1}: 只有 {valid_count}/{total_problems} 道题有结果")

    # 计算 Pass@1 平均值
    avg_pass_at_1 = sum(r["accuracy"] for r in pass_at_1_per_round) / len(pass_at_1_per_round) if pass_at_1_per_round else 0

    # ========== 计算 Pass@k: k 次中至少对 1 次的比例 ==========
    pass_at_k_results = {}
    for k in k_values:
        correct_count = 0
        for problem_hash, attempts in problem_attempts.items():
            # 取前 k 次尝试，只要有一次正确就算对
            first_k = attempts[:k]
            if any(first_k):
                correct_count += 1
        pass_at_k_results[f"pass@{k}"] = {
            "correct": correct_count,
            "total": total_problems,
            "accuracy": correct_count / total_problems if total_problems > 0 else 0,
        }

    # ========== 每个问题的详细结果 ==========
    per_problem_details = {}
    for problem_hash, attempts in problem_attempts.items():
        per_problem_details[problem_hash] = {
            "task_preview": problem_task_map[problem_hash],
            "n_attempts": len(attempts),
            "n_correct": sum(attempts),
            "attempts": attempts,  # [True, False, True, ...]
        }

    # ========== 打印结果 ==========
    print("=" * 60)
    print(f"Total unique problems: {total_problems}")
    print(f"Total trajectories: {sum(len(a) for a in problem_attempts.values())}")
    print(f"Number of rounds: {n_rounds}")
    print("=" * 60)

    print("\n--- Pass@1 (每轮正确率) ---")
    for r in pass_at_1_per_round:
        print(f"  Round {r['round']}: {r['correct']}/{r['total']} = {r['accuracy']:.4f}")
    print(f"  Average Pass@1: {avg_pass_at_1:.4f}")

    print("\n--- Pass@k (k次中至少对1次) ---")
    for k_name, info in pass_at_k_results.items():
        print(f"  {k_name}: {info['correct']}/{info['total']} = {info['accuracy']:.4f}")

    return {
        "pass_at_1_per_round": pass_at_1_per_round,
        "avg_pass_at_1": avg_pass_at_1,
        "pass_at_k": pass_at_k_results,
        "per_problem_details": per_problem_details,
        "total_problems": total_problems,
        "n_rounds": n_rounds,
    }


def save_trajectories(results, save_dir="./trajectories", filename="trajectories.pt"):
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    torch.save(results, save_path)
    print(f"Trajectories saved to {save_path}")
    return save_path
