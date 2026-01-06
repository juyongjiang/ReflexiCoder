import re
import math
import asyncio
import traceback
from typing import List, Union, Tuple
from concurrent.futures import ThreadPoolExecutor
from .sandbox.code_reward import code_reward_fn


_TRAJECTORY_CACHE = {} 
_thread_pool = ThreadPoolExecutor(max_workers=96) 

class CompletionParser:
    def __init__(self, sample_language: str = 'python'):
        self.reason_re = re.compile(r'<think>\s*([\s\S]*?)\s*</think>', re.MULTILINE)
        self.answer_re = re.compile(
            fr'<answer>\s*([\s\S]*?```{re.escape(sample_language)}[\s\S]*?```[\s\S]*?)\s*</answer>',
            re.MULTILINE
        )
        self.reflection_re = re.compile(r'<reflection>\s*([\s\S]*?)\s*</reflection>', re.MULTILINE)

    def parse(self, text: str):
        text = text.strip()
        pos = 0
        reflection_count = 0
        answers = []

        # reason
        m_reason = self.reason_re.search(text, pos)
        if not m_reason:
            return 0.0, 0, []
        pos = m_reason.end()

        # answer
        m_answer = self.answer_re.search(text, pos)
        if not m_answer:
            return 0.0, 0, []
        answers.append(m_answer.group(1))
        pos = m_answer.end()

        # reflection-answer pair
        m_ref = self.reflection_re.search(text, pos)
        if not m_ref:
            return 0.0, 0, []  
        
        reflection_count += 1
        pos = m_ref.end()

        m_answer = self.answer_re.search(text, pos)
        if not m_answer:
            return 0.0, 0, []
        answers.append(m_answer.group(1))
        pos = m_answer.end()

        # more reflection-answer pairs (optional)
        while pos < len(text):
            m_ref = self.reflection_re.search(text, pos)
            if not m_ref:
                break
            reflection_count += 1
            pos = m_ref.end()

            m_answer = self.answer_re.search(text, pos)
            if not m_answer:
                return 0.0, 0, []
            answers.append(m_answer.group(1))
            pos = m_answer.end()

        return (1.0, reflection_count, answers) if pos == len(text) else (0.0, 0, [])

_parser_cache = {} # global programming language parser
def get_parser(sample_language: str):
    if sample_language not in _parser_cache:
        _parser_cache[sample_language] = CompletionParser(sample_language)
    return _parser_cache[sample_language]

async def _parse_trajectory_async(completion: str, sample_language: str = 'python'):
    def _parse():
        try:
            parser = get_parser(sample_language)
            return parser.parse(completion)
        except Exception as e:
            try:
                import torch
                rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else "N/A"
            except:
                rank = "N/A"
            traceback.print_exc()
            print(f"[Rank {rank}] Exception: {repr(e)}, Return (0.0, 0, [])")
            return 0.0, 0, []

    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_thread_pool, _parse)


async def _get_cached_trajectory_async(
    completion: str,
    data_source: str = None,
    ground_truth: Union[str, List[str]] = None
) -> Tuple[float, int, List[str], List[float]]:
    key = hash(completion)

    if key in _TRAJECTORY_CACHE:
        format_reward, reflection_count, answers, r_scores = _TRAJECTORY_CACHE[key]
        if r_scores is None and data_source is not None:
            try:
                loop = asyncio.get_event_loop()
                r_scores = await asyncio.gather(*[
                    loop.run_in_executor(_thread_pool, code_reward_fn, data_source, ans, ground_truth)
                    for ans in answers
                ])
            except: # OOM
                r_scores = []
                traceback.print_exc()
                
            _TRAJECTORY_CACHE[key] = (format_reward, reflection_count, answers, r_scores)
        return format_reward, reflection_count, answers, r_scores

    format_reward, reflection_count, answers = await _parse_trajectory_async(completion)
    r_scores = None

    if data_source is not None:
        try:
            loop = asyncio.get_event_loop()
            r_scores = await asyncio.gather(*[
                loop.run_in_executor(_thread_pool, code_reward_fn, data_source, ans, ground_truth)
                for ans in answers
            ])
        except: # OOM
            r_scores = []
            traceback.print_exc()

    _TRAJECTORY_CACHE[key] = (format_reward, reflection_count, answers, r_scores)
    return format_reward, reflection_count, answers, r_scores


async def async_trajectory_format_reward_fn(completion: str) -> float:
    try:
        format_reward, _, _, _ = await _get_cached_trajectory_async(completion)
        return format_reward
    except Exception as e: # OOM
        try:
            import torch
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else "N/A"
        except:
            rank = "N/A"
        traceback.print_exc()
        print(f"[Rank {rank}] async_trajectory_format_reward_fn Exception: {repr(e)}, Return 0.0")
        return 0.0

async def async_cycle_count_regulation_reward_fn(completion: str, alpha: float, beta: float, gamma: float, delta: float) -> float:
    try:
        format_reward, reflection_count, _, _ = await _get_cached_trajectory_async(completion)
        n = reflection_count
        if format_reward == 0.0 or n == 0:
            return 0.0
        elif n <= 5: # number of maximum reflection
            return 1.0
        return (
            1.0 / (1.0 + alpha * (n - 3) ** beta) *
            math.exp(-gamma * (n - 3)) *
            (1.0 - delta * math.sin(math.pi / 2.0 * (n - 3)))
        )
    except Exception as e: # OOM
        try:
            import torch
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else "N/A"
        except:
            rank = "N/A"
        traceback.print_exc()
        print(f"[Rank {rank}] async_cycle_count_regulation_reward_fn Exception: {repr(e)}, Return 0.0")
        return 0.0

async def async_iterative_quality_reward_fn(data_source: str, completion: str, ground_truth: Union[str, List[str]],
                                            lambda_: float, s: float, eps: float, h_pos: float, h_neg: float, r_max: float, eta: float) -> float:
    try:                                         
        format_reward, reflection_count, _, r_scores = await _get_cached_trajectory_async(
            completion, data_source, ground_truth
        )

        # Early returns for invalid trajectories
        if format_reward == 0.0 or reflection_count == 0:
            return 0.0
        if not r_scores:
            return 0.0

        # Time weights
        weights = [math.exp(lambda_ * t) for t in range(1, len(r_scores))]
        weight_sum = sum(weights)
        weights = [w / weight_sum for w in weights]

        m = [0.0]  # m_1 = 0 since no previous answer
        for t in range(1, len(r_scores)):
            diff = r_scores[t] - r_scores[t-1] # 0, 1

            if diff > eps:
                # Positive improvement
                m_t = math.tanh(diff / s) # +1.0
            elif abs(diff) <= eps:
                # No improvement
                if abs(r_scores[t-1] - r_max) <= eps:
                    # Already at maximum → no penalty
                    m_t = h_pos # +0.05
                else:
                    # Stagnation below maximum → penalty
                    m_t = -h_neg # -1.0
            else:  # diff < -eps
                # Decline in quality → negative reward
                m_t = -math.tanh(abs(diff) / s) # -0.1

            m.append(m_t) 

        # Trajectory reward
        return r_scores[-1] + eta * sum(weights[t-1] * m[t] for t in range(1, len(r_scores)))

    except Exception as e: # OOM
        try:
            import torch
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else "N/A"
        except:
            rank = "N/A"
        traceback.print_exc()
        print(f"[Rank {rank}] async_iterative_quality_reward_fn Exception: {repr(e)}, Return 0.0")
        return 0.0

async def async_efficiency_reward_fn(data_source: str, completion: str, ground_truth: Union[str, List[str]], epsilon: float) -> float:
    try:
        format_reward, reflection_count, _, r_scores = await _get_cached_trajectory_async(completion, data_source, ground_truth)
        if format_reward == 0.0 or reflection_count == 0:
            return 0.0

        if not r_scores:
            return 0.0

        avg_score = r_scores[-1] / reflection_count 
        if len(r_scores) < 2:
            return 0.0
        return avg_score + (r_scores[-1] - r_scores[0]) / (max(1, reflection_count - 1) + epsilon)
    except Exception as e: # OOM
        try:
            import torch
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else "N/A"
        except:
            rank = "N/A"
        traceback.print_exc()
        print(f"[Rank {rank}] async_efficiency_reward_fn Exception: {repr(e)}, Return 0.0")
        return 0.0

###########################################

async def async_cycle_count_iterative_quality_reward_fn(data_source: str, completion: str, ground_truth: Union[str, List[str]],
                                                        lambda_: float, s: float, eps: float, h_pos: float, h_neg: float, r_max: float, eta: float,
                                                        alpha: float, beta: float, gamma: float, delta: float) -> float:
    try:
        P_n = await async_cycle_count_regulation_reward_fn(completion, alpha, beta, gamma, delta)
        R_traj = await async_iterative_quality_reward_fn(data_source, completion, ground_truth, lambda_, s, eps, h_pos, h_neg, r_max, eta)
        return P_n * R_traj
    except Exception as e:
        try:
            import torch
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else "N/A"
        except:
            rank = "N/A"
        traceback.print_exc()
        print(f"[Rank {rank}] async_cycle_count_iterative_quality_reward_fn Exception: {repr(e)}, Return 0.0")
        return 0.0

async def async_cycle_count_efficient_reward_fn(data_source: str, completion: str, ground_truth: Union[str, List[str]],
                                                epsilon: float,
                                                alpha: float, beta: float, gamma: float, delta: float) -> float:
    try:
        P_n = await async_cycle_count_regulation_reward_fn(completion, alpha, beta, gamma, delta)
        E_traj = await async_efficiency_reward_fn(data_source, completion, ground_truth, epsilon)
        return P_n * E_traj
    except Exception as e:
        try:
            import torch
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else "N/A"
        except:
            rank = "N/A"
        traceback.print_exc()
        print(f"[Rank {rank}] async_cycle_count_iterative_quality_reward_fn Exception: {repr(e)}, Return 0.0")
        return 0.0

###########################################

async def async_first_code_reward_fn(data_source: str, completion: str, ground_truth: Union[str, List[str]]) -> float:
    try:
        format_reward, reflection_count, _, r_scores = await _get_cached_trajectory_async(completion, data_source, ground_truth)
        if format_reward == 0.0 or reflection_count == 0:
            return 0.0

        if not r_scores:
            return 0.0

        return r_scores[0]
    except Exception as e: # OOM
        try:
            import torch
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else "N/A"
        except:
            rank = "N/A"
        traceback.print_exc()
        print(f"[Rank {rank}] async_first_code_reward_fn Exception: {repr(e)}, Return 0.0")
        return 0.0

async def async_last_code_reward_fn(data_source: str, completion: str, ground_truth: Union[str, List[str]]) -> float:
    try:
        format_reward, reflection_count, _, r_scores = await _get_cached_trajectory_async(completion, data_source, ground_truth)
        if format_reward == 0.0 or reflection_count == 0:
            return 0.0

        if not r_scores:
            return 0.0

        return r_scores[-1]
    except Exception as e: # OOM
        try:
            import torch
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else "N/A"
        except:
            rank = "N/A"
        traceback.print_exc()
        print(f"[Rank {rank}] async_last_code_reward_fn Exception: {repr(e)}, Return 0.0")
        return 0.0

async def async_cycle_count_reward_fn(data_source: str, completion: str, ground_truth: Union[str, List[str]]) -> float:
    try:
        format_reward, reflection_count, _, r_scores = await _get_cached_trajectory_async(completion, data_source, ground_truth)
        if format_reward == 0.0 or reflection_count == 0:
            return 0.0

        if not r_scores:
            return 0.0

        return reflection_count
    except Exception as e: # OOM
        try:
            import torch
            rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else "N/A"
        except:
            rank = "N/A"
        traceback.print_exc()
        print(f"[Rank {rank}] async_first_code_reward_fn Exception: {repr(e)}, Return 0.0")
        return 0.0