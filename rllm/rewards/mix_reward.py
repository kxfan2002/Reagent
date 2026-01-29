"""
混合 Reward 函数：自动根据答案类型选择数学判分或文本判分

基于原始实现：
- 数学判分：使用 MathRuler 的 grade_answer
- 文本判分：使用 search_reward 的 F1/EM 评估
"""

import re
from typing import Any

from rllm.rewards.math_reward import RewardMathFn
from rllm.rewards.reward_types import RewardConfig, RewardInput, RewardOutput
from rllm.rewards.search_reward import RewardSearchFn


class RewardMixFn:
    """
    混合 Reward 函数类，能够根据答案类型自动选择合适的评估方法
    
    - 数学答案（整数、小数、分数、LaTeX表达式）→ 使用 RewardMathFn
    - 文本答案 → 使用 RewardSearchFn (F1/EM)
    """
    
    def __init__(self, config: RewardConfig):
        self.config = config
        self.math_reward_fn = RewardMathFn(config)
        self.search_reward_fn = RewardSearchFn(config)
    
    def extract_boxed_content(self, text: str) -> str | None:
        """
        从 \\boxed{} 中提取内容，支持嵌套括号
        
        Args:
            text: 包含 \\boxed{} 的文本
            
        Returns:
            提取的内容，如果没有找到则返回 None
        """
        if not text or not isinstance(text, str):
            return None
        
        # 寻找最后一个 \boxed{ 的位置
        box_match = re.search(r'\\boxed?\{', text)
        if box_match:
            start_pos = box_match.end() - 1  # 指向开始的 {
            brace_count = 0
            content_start = box_match.end()
            
            for i, char in enumerate(text[start_pos:], start_pos):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        content = text[content_start:i].strip()
                        return content if content else None
        return None
    
    def is_math_answer(self, answer: str) -> bool:
        """
        判断答案是否为数学类型
        
        识别规则：
        - 纯数字（整数、小数、分数）
        - LaTeX 数学表达式
        - 科学计数法
        
        Args:
            answer: 待判断的答案字符串
            
        Returns:
            True 表示数学答案，False 表示文本答案
        """
        if not answer or not isinstance(answer, str):
            return False
        
        answer = answer.strip()
        
        # 1. 检查是否为纯数字（整数）
        if re.match(r'^[+-]?\d+$', answer):
            return True
        
        # 2. 检查是否为纯小数
        if re.match(r'^[+-]?\d*\.\d+$', answer):
            return True
        
        # 3. 检查是否为纯分数
        if re.match(r'^[+-]?\d+/\d+$', answer):
            return True
        
        # 4. 检查 LaTeX 数学表达式
        latex_patterns = [
            r'\\boxed\{.*\}',                    # LaTeX boxed
            r'\\[dtf]?frac\{.*\}\{.*\}',         # LaTeX 分数 (\frac, \dfrac, \tfrac)
            r'^\$.*\$$',                          # LaTeX 数学模式
            r'[+\-*/=<>≤≥≠∫∑∏]',                # 数学运算符
        ]
        
        for pattern in latex_patterns:
            if re.search(pattern, answer):
                return True
        
        # 5. 科学计数法
        if re.match(r'^[+-]?\d*\.?\d+[eE][+-]?\d+$', answer):
            return True
        
        # 6. 包含数学变量和运算的表达式（如 2x+3, x^2）
        if re.search(r'[a-zA-Z]\s*[\^+\-*/]|[\^+\-*/]\s*[a-zA-Z]', answer):
            return True
        
        return False
    
    def __call__(self, task_info: dict, action: str) -> RewardOutput:
        """
        计算混合 reward，自动根据答案类型选择评估方法
        
        Args:
            task_info: 任务信息字典，包含 ground_truth 等
            action: 模型的回答
            
        Returns:
            RewardOutput: 包含 reward、is_correct 和 metadata
        """
        if not action or action == "":
            return RewardOutput(
                reward=self.config.format_error_reward,
                is_correct=False,
                metadata={"error": "Empty action", "evaluation_method": "none"}
            )
        
        # 获取 ground_truth
        ground_truth = task_info.get("ground_truth") or task_info.get("answer")
        if ground_truth is None:
            return RewardOutput(
                reward=self.config.unk_error_reward,
                is_correct=False,
                metadata={"error": "No ground truth provided", "evaluation_method": "none"}
            )
        
        # 提取答案内容（从 \boxed{} 中）
        extracted_answer = self.extract_boxed_content(action)
        if extracted_answer is None:
            # 如果没有 \boxed{}，使用 search_reward 的提取方法
            extracted_answer = self.search_reward_fn.extract_answer_from_response(action)
        
        # 🔥 Debug 日志：打印答案和标准答案
        # print(f"[RewardMixFn] 提取的答案: {extracted_answer}", flush=True)
        # print(f"[RewardMixFn] 标准答案: {ground_truth}", flush=True)
        # print(f"[RewardMixFn] 完整回答: {action[:200]}...", flush=True)
        
        # 🔥 特殊处理：yes/no 答案（大小写不敏感）
        # 如果 ground_truth 是 yes 或 no，只要提取的答案中包含对应的词就算正确
        gt_normalized = str(ground_truth).strip().lower()
        if gt_normalized in ["yes", "no"]:
            extracted_normalized = str(extracted_answer).strip().lower() if extracted_answer else ""
            # 检查提取的答案中是否包含 ground_truth
            if gt_normalized in extracted_normalized:
                return RewardOutput(
                    reward=self.config.correct_reward,
                    is_correct=True,
                    metadata={
                        "evaluation_method": "yes_no_match",
                        "extracted_answer": extracted_answer,
                        "ground_truth": ground_truth,
                        "answer_type": "yes_no"
                    }
                )
            else:
                return RewardOutput(
                    reward=self.config.incorrect_reward,
                    is_correct=False,
                    metadata={
                        "evaluation_method": "yes_no_mismatch",
                        "extracted_answer": extracted_answer,
                        "ground_truth": ground_truth,
                        "answer_type": "yes_no"
                    }
                )
        
        # 🔥 特殊处理：true/false 答案（大小写不敏感）
        # 如果 ground_truth 是 true 或 false，只要提取的答案中包含对应的词就算正确
        if gt_normalized in ["true", "false"]:
            extracted_normalized = str(extracted_answer).strip().lower() if extracted_answer else ""
            # 检查提取的答案中是否包含 ground_truth
            if gt_normalized in extracted_normalized:
                return RewardOutput(
                    reward=self.config.correct_reward,
                    is_correct=True,
                    metadata={
                        "evaluation_method": "true_false_match",
                        "extracted_answer": extracted_answer,
                        "ground_truth": ground_truth,
                        "answer_type": "true_false"
                    }
                )
            else:
                return RewardOutput(
                    reward=self.config.incorrect_reward,
                    is_correct=False,
                    metadata={
                        "evaluation_method": "true_false_mismatch",
                        "extracted_answer": extracted_answer,
                        "ground_truth": ground_truth,
                        "answer_type": "true_false"
                    }
                )
        
        # 判断答案类型
        # 转换 ground_truth 为字符串进行判断
        if isinstance(ground_truth, list):
            # 如果 ground_truth 是列表，检查第一个元素
            gt_sample = str(ground_truth[0]) if ground_truth else ""
        else:
            gt_sample = str(ground_truth)
        
        is_math = self.is_math_answer(extracted_answer) or self.is_math_answer(gt_sample)
        
        # 🔥 Debug 日志：打印答案类型判断
        # print(f"[RewardMixFn] 答案类型判断: is_math={is_math}, extracted_answer={extracted_answer}, gt_sample={gt_sample}", flush=True)
        
        # 根据答案类型选择评估方法
        if is_math:
            # 使用数学 reward 函数
            try:
                result = self.math_reward_fn(task_info, action)
                result.metadata["evaluation_method"] = "math"
                result.metadata["extracted_answer"] = extracted_answer
                result.metadata["answer_type"] = "math"
                
                # 🔥 Debug 日志：打印数学判分结果
                print(f"[RewardMixFn] 使用数学评估 -> reward={result.reward}, is_correct={result.is_correct}", flush=True)
                
                return result
            except Exception as e:
                # 如果数学评估失败，降级到文本评估
                print(f"⚠️  Math evaluation failed: {e}, falling back to text evaluation", flush=True)
                pass
        
        # 使用文本 reward 函数（search reward）
        reward_input = RewardInput(task_info=task_info, action=action)
        result = self.search_reward_fn(reward_input)
        result.metadata["evaluation_method"] = "text"
        result.metadata["answer_type"] = "text"
        
        # 🔥 Debug 日志：打印文本判分结果
        print(f"[RewardMixFn] 使用文本评估 -> reward={result.reward}, is_correct={result.is_correct}", flush=True)
        
        return result


def mixed_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    便捷的混合 reward 函数接口
    
    自动根据答案类型选择合适的判分函数：
    - 数学答案 → 使用 math_acc_reward (基于 MathRuler)
    - 文本答案 → 使用文本相似度 (F1/EM)
    
    Args:
        task_info: 任务信息字典，包含 ground_truth 等
        action: 模型的回答
        
    Returns:
        RewardOutput: 包含 reward、is_correct 和 metadata
        
    Example:
        >>> task_info = {"ground_truth": "42"}
        >>> action = "The answer is \\boxed{42}"
        >>> result = mixed_reward_fn(task_info, action)
        >>> print(result.reward)  # 1.0
        >>> print(result.metadata["evaluation_method"])  # "math"
    """
    reward_config = RewardConfig()
    reward_fn = RewardMixFn(reward_config)
    return reward_fn(task_info, action)


def adaptive_reward_fn(task_info: dict, action: str) -> RewardOutput:
    """
    自适应 reward 函数，根据 data_source 字段自动选择评估方法
    
    如果 task_info 中有 data_source 字段：
    - data_source == "math" → 强制使用数学评估
    - data_source == "search" → 强制使用文本评估
    - 其他 → 自动判断答案类型
    
    Args:
        task_info: 任务信息字典，可包含 data_source 字段
        action: 模型的回答
        
    Returns:
        RewardOutput: 包含 reward、is_correct 和 metadata
    """
    reward_config = RewardConfig()
    data_source = task_info.get("data_source", "auto")
    meta_info_source = task_info.get("meta_info_source", "auto")
    category = task_info.get("category", "auto")
    # import pdb; pdb.set_trace()
    ground_truth = task_info.get("ground_truth") or task_info.get("answer")
    extracted_answer = None
    mix_fn = RewardMixFn(reward_config)
    if ground_truth is not None:
        # 需要实例化一个 RewardMixFn 来使用其 extract_boxed_content 方法
        # mix_fn = RewardMixFn(reward_config)
        extracted_answer = mix_fn.extract_boxed_content(action)
        if extracted_answer is None:
            # 如果没有 \boxed{}，使用 search_reward 的提取方法
            search_fn = RewardSearchFn(reward_config)
            extracted_answer = search_fn.extract_answer_from_response(action)
        
        gt_normalized = str(ground_truth).strip().lower()
        
        # 🔥 特殊处理：yes/no 答案
        if gt_normalized in ["yes", "no"]:
            extracted_normalized = str(extracted_answer).strip().lower() if extracted_answer else ""
            if gt_normalized in extracted_normalized:
                print(f"[adaptive_reward_fn] 使用 yes/no 评估 -> ground_truth={ground_truth} extracted_answer={extracted_answer} reward={reward_config.correct_reward}", flush=True)
                return RewardOutput(
                    reward=reward_config.correct_reward,
                    is_correct=True,
                    metadata={
                        "evaluation_method": "yes_no_match",
                        "extracted_answer": extracted_answer,
                        "ground_truth": ground_truth,
                        "answer_type": "yes_no",
                        "data_source": data_source
                    }
                )
            else:
                print(f"[adaptive_reward_fn] 使用 yes/no 评估 -> ground_truth={ground_truth} extracted_answer={extracted_answer} reward={reward_config.incorrect_reward}", flush=True)
                return RewardOutput(
                    reward=reward_config.incorrect_reward,
                    is_correct=False,
                    metadata={
                        "evaluation_method": "yes_no_mismatch",
                        "extracted_answer": extracted_answer,
                        "ground_truth": ground_truth,
                        "answer_type": "yes_no",
                        "data_source": data_source
                    }
                )
        
        # 🔥 特殊处理：true/false 答案
        if gt_normalized in ["true", "false"]:
            extracted_normalized = str(extracted_answer).strip().lower() if extracted_answer else ""
            if gt_normalized in extracted_normalized:
                print(f"[adaptive_reward_fn] 使用 true/false 评估 -> ground_truth={ground_truth} extracted_answer={extracted_answer} reward={reward_config.correct_reward}", flush=True)
                return RewardOutput(
                    reward=reward_config.correct_reward,
                    is_correct=True,
                    metadata={
                        "evaluation_method": "true_false_match",
                        "extracted_answer": extracted_answer,
                        "ground_truth": ground_truth,
                        "answer_type": "true_false",
                        "data_source": data_source
                    }
                )
            else:
                print(f"[adaptive_reward_fn] 使用 true/false 评估 -> ground_truth={ground_truth} extracted_answer={extracted_answer} reward={reward_config.incorrect_reward}", flush=True)
                return RewardOutput(
                    reward=reward_config.incorrect_reward,
                    is_correct=False,
                    metadata={
                        "evaluation_method": "true_false_mismatch",
                        "extracted_answer": extracted_answer,
                        "ground_truth": ground_truth,
                        "answer_type": "true_false",
                        "data_source": data_source
                    }
                )
    
    if meta_info_source == "math" or category == "math" or data_source == "math":
        # 强制使用数学评估
        # print(f"[adaptive_reward_fn] 使用数学评估（强制）", flush=True)
        math_fn = RewardMathFn(reward_config)
        result = math_fn(task_info, action)
        result.metadata["evaluation_method"] = "math"
        result.metadata["data_source"] = data_source
        print(f"[adaptive_reward_fn] 使用数学评估（强制） -> ground_truth={ground_truth} extracted_answer={extracted_answer} reward={result.reward}", flush=True)
        
        # 🔥 Debug 日志：打印结果
        # print(f"[adaptive_reward_fn] 数学评估结果 -> reward={result.reward}, is_correct={result.is_correct}", flush=True)
        
        return result
    elif data_source == "search" or str(data_source).lower().find("qa") != -1:
        # 强制使用文本评估
        # print(f"[adaptive_reward_fn] 使用文本评估（强制）", flush=True)
        search_fn = RewardSearchFn(reward_config)
        reward_input = RewardInput(task_info=task_info, action=action)
        result = search_fn(reward_input)
        result.metadata["evaluation_method"] = "text"
        result.metadata["data_source"] = data_source
        print(f"[adaptive_reward_fn] 使用文本评估（强制） -> ground_truth={ground_truth} extracted_answer={extracted_answer} reward={result.reward}", flush=True)
        # 🔥 Debug 日志：打印结果
        # print(f"[adaptive_reward_fn] 文本评估结果 -> reward={result.reward}, is_correct={result.is_correct}", flush=True)
        
        return result
    else:
        # 自动判断
        # print(f"[adaptive_reward_fn] 使用自动判断模式", flush=True)
        # mix_fn = RewardMixFn(reward_config)
        result = mix_fn(task_info, action)
        result.metadata["data_source"] = data_source
        print(f"[adaptive_reward_fn] 使用自动判断模式 -> ground_truth={ground_truth} extracted_answer={extracted_answer} reward={result.reward}", flush=True)
        return result


if __name__ == "__main__":
    # 测试数学答案
    print("=" * 80)
    print("测试 1: 数学答案")
    print("=" * 80)
    task_info_math = {
        "ground_truth": "42",
        "data_source": "math"
    }
    action_math = "After calculation, the answer is \\boxed{42}"
    result_math = mixed_reward_fn(task_info_math, action_math)
    print(f"Reward: {result_math.reward}")
    print(f"Is Correct: {result_math.is_correct}")
    print(f"Metadata: {result_math.metadata}")
    print()
    
    # 测试文本答案
    print("=" * 80)
    print("测试 2: 文本答案")
    print("=" * 80)
    task_info_text = {
        "ground_truth": "Paris",
        "data_source": "search"
    }
    action_text = "The capital of France is \\boxed{Paris}"
    result_text = mixed_reward_fn(task_info_text, action_text)
    print(f"Reward: {result_text.reward}")
    print(f"Is Correct: {result_text.is_correct}")
    print(f"Metadata: {result_text.metadata}")
    print()
    
    # 测试自适应模式
    print("=" * 80)
    print("测试 3: 自适应模式（自动判断）")
    print("=" * 80)
    task_info_auto = {
        "ground_truth": "9:45"
    }
    action_auto = "\\boxed{9:45}"
    result_auto = adaptive_reward_fn(task_info_auto, action_auto)
    print(f"Reward: {result_auto.reward}")
    print(f"Is Correct: {result_auto.is_correct}")
    print(f"Metadata: {result_auto.metadata}")
    print()
    
    # 🔥 测试 yes/no 答案（正确情况）
    print("=" * 80)
    print("测试 4: Yes/No 答案 - 正确（包含额外文本）")
    print("=" * 80)
    task_info_yes = {
        "ground_truth": "yes"
    }
    action_yes = "\\boxed{Yes, the reason is that the evidence clearly supports this conclusion.}"
    result_yes = mixed_reward_fn(task_info_yes, action_yes)
    print(f"Reward: {result_yes.reward}")
    print(f"Is Correct: {result_yes.is_correct}")
    print(f"Metadata: {result_yes.metadata}")
    print()
    
    # 🔥 测试 yes/no 答案（大小写不敏感）
    print("=" * 80)
    print("测试 5: Yes/No 答案 - 大小写不敏感")
    print("=" * 80)
    task_info_no = {
        "ground_truth": "NO"
    }
    action_no = "\\boxed{no, because the conditions are not met}"
    result_no = mixed_reward_fn(task_info_no, action_no)
    print(f"Reward: {result_no.reward}")
    print(f"Is Correct: {result_no.is_correct}")
    print(f"Metadata: {result_no.metadata}")
    print()
    
    # 🔥 测试 yes/no 答案（错误情况）
    print("=" * 80)
    print("测试 6: Yes/No 答案 - 错误（不匹配）")
    print("=" * 80)
    task_info_yes_wrong = {
        "ground_truth": "yes"
    }
    action_yes_wrong = "\\boxed{No, this is incorrect}"
    result_yes_wrong = mixed_reward_fn(task_info_yes_wrong, action_yes_wrong)
    print(f"Reward: {result_yes_wrong.reward}")
    print(f"Is Correct: {result_yes_wrong.is_correct}")
    print(f"Metadata: {result_yes_wrong.metadata}")
    print()
    
    # 🔥 测试 true/false 答案（正确情况）
    print("=" * 80)
    print("测试 7: True/False 答案 - 正确（包含额外文本）")
    print("=" * 80)
    task_info_true = {
        "ground_truth": "true"
    }
    action_true = "\\boxed{True, because the statement is logically valid.}"
    result_true = mixed_reward_fn(task_info_true, action_true)
    print(f"Reward: {result_true.reward}")
    print(f"Is Correct: {result_true.is_correct}")
    print(f"Metadata: {result_true.metadata}")
    print()
    
    # 🔥 测试 true/false 答案（大小写不敏感）
    print("=" * 80)
    print("测试 8: True/False 答案 - 大小写不敏感")
    print("=" * 80)
    task_info_false = {
        "ground_truth": "FALSE"
    }
    action_false = "\\boxed{false, the condition does not hold}"
    result_false = mixed_reward_fn(task_info_false, action_false)
    print(f"Reward: {result_false.reward}")
    print(f"Is Correct: {result_false.is_correct}")
    print(f"Metadata: {result_false.metadata}")
    print()
    
    # 🔥 测试 true/false 答案（错误情况）
    print("=" * 80)
    print("测试 9: True/False 答案 - 错误（不匹配）")
    print("=" * 80)
    task_info_true_wrong = {
        "ground_truth": "true"
    }
    action_true_wrong = "\\boxed{False, this is incorrect}"
    result_true_wrong = mixed_reward_fn(task_info_true_wrong, action_true_wrong)
    print(f"Reward: {result_true_wrong.reward}")
    print(f"Is Correct: {result_true_wrong.is_correct}")
    print(f"Metadata: {result_true_wrong.metadata}")
    print()
    
    print("✅ 所有测试完成！")

