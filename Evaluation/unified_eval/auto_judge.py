import re
import time
import random
import string
import requests
import os
from typing import Dict, Any, Optional
from openai import OpenAI


def get_geminiflash_response(query: str, temperature: float = 0.0, max_retry: int = 5) -> str:
    """Get response from Gemini Flash model using standard OpenAI-compatible API."""
    api_key = os.environ.get("GEMINI_API_KEY")
    api_base = os.environ.get("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
    
    if not api_key:
        print("Warning: GEMINI_API_KEY not set, skipping Gemini response", flush=True)
        return None
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        for retry_cnt in range(max_retry):
            try:
                response = client.chat.completions.create(
                    model="gemini-2.0-flash-exp",
                    messages=[{"role": "user", "content": query}],
                    temperature=temperature,
                    max_tokens=32768
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"get_geminiflash_response {retry_cnt} error: {e}", flush=True)
                if retry_cnt == max_retry - 1:
                    return None
                time.sleep(random.uniform(4, 32))
    except Exception as e:
        print(f"Failed to initialize Gemini client: {e}", flush=True)
    
    return None


def get_deepseekchat_response(query: str, temperature: float = 0.0, max_retry: int = 5) -> str:
    """Get response from DeepSeek Chat model using standard OpenAI-compatible API."""
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    api_base = os.environ.get("DEEPSEEK_API_BASE", "https://api.deepseek.com/v1")
    
    if not api_key:
        print("Warning: DEEPSEEK_API_KEY not set, skipping DeepSeek response", flush=True)
        return None
    
    try:
        client = OpenAI(
            api_key=api_key,
            base_url=api_base
        )
        
        for retry_cnt in range(max_retry):
            try:
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[{"role": "user", "content": query}],
                    temperature=temperature,
                    max_tokens=8192
                )
                content = response.choices[0].message.content
                if content:
                    return content
            except Exception as e:
                print(f"get_deepseekchat_response {retry_cnt} error: {e}", flush=True)
                if retry_cnt == max_retry - 1:
                    return None
                time.sleep(random.uniform(4, 32))
    except Exception as e:
        print(f"Failed to initialize DeepSeek client: {e}", flush=True)
    
    return None


def normalize_answer(s):
    """Normalize answer for comparison."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_xml_field(reply, field):
    """Extract XML field from reply."""
    import re
    try:
        result = re.findall(f"<{field}>(.*?)</{field}>", reply, re.DOTALL)
        if len(result) == 1:
            return result[0].strip()
        else:
            raise ValueError(f"There are {len(result)} {field} field in reply")
    except:
        raise ValueError(f"Failed to extract {field} field")


def extract_solution(solution_str):
    """Extract the answer from the solution string."""
    # Handle different message formats
    if "<|im_start|>assistant" in solution_str:
        extract_solution_str = solution_str.split("<|im_start|>assistant")[-1].split("</think>")[-1].replace("<|im_end|>", "").strip()
    else:
        # Handle normal conversation format or direct prediction content
        extract_solution_str = solution_str.strip()
    
    try:
        # Try to extract XML format answer tag
        answer = extract_xml_field(extract_solution_str, "answer")
    except:
        # If no XML format, try to extract simple <answer> tag content
        if "<answer>" in extract_solution_str and "</answer>" in extract_solution_str:
            answer = extract_solution_str.split("<answer>")[1].split("</answer>")[0].strip()
        else:
            # If neither exists, return the entire content as answer
            answer = extract_solution_str.strip()
    
    return answer


def em_check(prediction, golden_answers):
    """Exact match check."""
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    normalized_prediction = normalize_answer(prediction)
    score = 0
    for golden_answer in golden_answers:
        golden_answer = normalize_answer(golden_answer)
        if golden_answer == normalized_prediction:
            score = 1
            break
    return score


JUDGE_PROMPT = """
Judge whether the following [response] to [question] is correct, based on the [ground_truth] below. [ground_truth] is a list of correct answers.

[question]: %s

[response]: %s

[ground_truth]: %s

You need to first extract the final answer from the [response], then give a label "yes" or "no" based on if the extracted_final_answer matches the [ground_truth].
Your judgement must be in the following format:

<reasoning>
<!-- Think briefly about what is the extracted_final_answer, and why the extracted_final_answer is correct or incorrect based on [ground_truth]. Focus only on if there are meaningful differences between [ground_truth] and the extracted_final_answer. -->
</reasoning>
<extracted_final_answer>
<!-- The final exact answer extracted from the [response]. The [response] must be confident, definite and clear. It cannot be a guess or a general prediction. The extracted_final_answer must be unambiguous. Otherwise, return "no answer". -->
</extracted_final_answer>
<label>
<!-- 'yes' if the extracted_final_answer matches the [ground_truth], otherwise 'no'. -->
</label>
"""


def compute_score_genrm(prediction: str, ground_truth: str, question: str, engine: str = "deepseekchat") -> Dict[str, Any]:
    """
    The scoring function with llm as judge.
    
    Args:
        prediction: The final prediction/answer content (not the full conversation)
        ground_truth: The correct answer(s) 
        question: The original question
        engine: LLM engine to use for judging ("deepseekchat" or "geminiflash")
    
    Returns:
        Dict containing score, extracted_answer, and judge_response
    """
    # 从 prediction 中提取答案，处理可能的 <answer> 标签
    answer = extract_solution(prediction)
    
    judge_prompt = JUDGE_PROMPT % (question, answer, ground_truth)
    judge_prompt = judge_prompt.strip()
    
    time.sleep(random.uniform(0, 16))  # Reduce delay
    
    try:
        if engine == "geminiflash":
            judge_response = get_geminiflash_response(judge_prompt, max_retry=30)
        else:  # default to deepseekchat
            judge_response = get_deepseekchat_response(judge_prompt, max_retry=30)
        
        try:
            label = extract_xml_field(judge_response, "label")
            if "yes" in label.lower():
                score = 1
            else:
                score = 0
        except:
            # If XML parsing fails, try simple text matching
            if "yes" in judge_response.lower() and "no" not in judge_response.lower():
                score = 1
            else:
                score = 0
                
        result = {
            "score": score,
            "extracted_answer": answer,
            "judge_response": judge_response,
            "ground_truth": ground_truth,
            "question": question
        }
        
        print(f"Auto Judge Result: Score={score}, Answer='{answer}', Ground Truth='{ground_truth}'")
        
    except Exception as e:
        print(f"Auto judge failed with error: {e}, falling back to exact match")
        # Fall back to exact matching
        score = em_check(answer, ground_truth)
        result = {
            "score": score,
            "extracted_answer": answer,
            "judge_response": f"Auto judge failed: {str(e)}",
            "ground_truth": ground_truth,
            "question": question,
            "fallback": True
        }
    
    return result


def simple_em_score(prediction: str, ground_truth: str) -> Dict[str, Any]:
    """Simple exact match scoring without LLM judge."""
    answer = extract_solution(prediction)
    score = em_check(answer, ground_truth)
    
    return {
        "score": score,
        "extracted_answer": answer,
        "ground_truth": ground_truth,
        "method": "exact_match"
    }

if __name__ == "__main__":
    prediction = """
    <answer>
    The answer is 10.
    </answer>
    """
    ground_truth = "10"
    question = "What is the answer?"
    print(compute_score_genrm(prediction, ground_truth, question, engine="deepseekchat"))