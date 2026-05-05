"""
AgentFlow multi-turn custom convert.

Splits each multi-turn trajectory into independent training samples (one per turn),
with all turns from the same trajectory sharing the same normalized advantage.

Usage:
     --custom-convert-samples-to-train-data-path custom_convert.custom_convert
"""

import logging
from collections.abc import Sequence
from typing import Any

import torch
from slime.utils.types import Sample

logger = logging.getLogger(__name__)


def _to_float_list(values: Any) -> list[float]:
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    return [float(v) for v in values]


def _is_nested_log_probs(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return value.ndim > 1
    return bool(value) and isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and isinstance(
        value[0], (list, tuple, torch.Tensor)
    )


def _trim_sglang_teacher_output(teacher_out: dict[str, Any], response_length: int) -> list[float]:
    meta = teacher_out.get("meta_info", {})
    input_logprobs = meta.get("input_token_logprobs")
    if not input_logprobs:
        raise ValueError("teacher output is missing meta_info.input_token_logprobs")
    token_logprobs = [item[0] for item in input_logprobs[1:]]
    if response_length <= 0:
        return []
    if len(token_logprobs) < response_length:
        raise ValueError(
            f"teacher output has {len(token_logprobs)} token logprobs, "
            f"but response_length={response_length}"
        )
    return [float(x) for x in token_logprobs[-response_length:]]


def _get_scalar_reward(args, sample: Sample) -> float:
    try:
        return float(sample.get_reward_value(args))
    except Exception:
        reward = sample.reward
        if isinstance(reward, dict):
            for key in (getattr(args, "reward_key", None), "score", "reward"):
                if key and key in reward:
                    return float(reward[key])
            # Pure OPD reward_func may carry only teacher data.
            if "opd" in reward or "meta_info" in reward or "teacher_log_probs" in reward:
                return 0.0
        if reward is None:
            return 0.0
        return float(reward)


def _extract_teacher_log_probs_from_reward(sample: Sample, response_length: int) -> list[float] | None:
    if sample.teacher_log_probs is not None:
        return _to_float_list(sample.teacher_log_probs)

    reward = sample.reward
    if not isinstance(reward, dict):
        return None

    opd_payload = reward.get("opd")
    if isinstance(opd_payload, dict) and "teacher_log_probs" in opd_payload:
        log_probs = opd_payload["teacher_log_probs"]
        if _is_nested_log_probs(log_probs):
            return None
        return _to_float_list(log_probs)

    if "teacher_log_probs" in reward:
        log_probs = reward["teacher_log_probs"]
        if _is_nested_log_probs(log_probs):
            return None
        return _to_float_list(log_probs)

    if "meta_info" in reward:
        return _trim_sglang_teacher_output(reward, response_length)

    return None


def _extract_teacher_log_probs_by_turn(sample: Sample, turns: list[dict]) -> list[list[float]] | None:
    reward = sample.reward
    if isinstance(reward, dict):
        opd_payload = reward.get("opd")
        if isinstance(opd_payload, dict) and "teacher_log_probs" in opd_payload:
            log_probs = opd_payload["teacher_log_probs"]
            if _is_nested_log_probs(log_probs):
                return [_to_float_list(item) for item in log_probs]

        if "teacher_log_probs" in reward:
            log_probs = reward["teacher_log_probs"]
            if _is_nested_log_probs(log_probs):
                return [_to_float_list(item) for item in log_probs]

    if sample.teacher_log_probs is not None and _is_nested_log_probs(sample.teacher_log_probs):
        return [_to_float_list(item) for item in sample.teacher_log_probs]

    return None


def custom_convert(args, samples):
    # ── 1. Trajectory-level GRPO reward normalization ──
    raw_rewards = [_get_scalar_reward(args, s) for s in samples]
    rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float)

    if (
        getattr(args, "advantage_estimator", None) in ["grpo", "gspo", "reinforce_plus_plus_baseline"]
        and getattr(args, "rewards_normalization", False)
    ):
        n = getattr(args, "n_samples_per_prompt", 1)
        if rewards_tensor.shape[-1] == n * getattr(args, "rollout_batch_size", 1):
            rewards_tensor = rewards_tensor.reshape(-1, n)
        else:
            rewards_tensor = rewards_tensor.view(-1, rewards_tensor.shape[-1])

        mean = rewards_tensor.mean(dim=-1, keepdim=True)
        rewards_tensor = rewards_tensor - mean

        if (
            getattr(args, "advantage_estimator", None) in ["grpo", "gspo"]
            and getattr(args, "grpo_std_normalization", False)
        ):
            std = rewards_tensor.std(dim=-1, keepdim=True)
            rewards_tensor = rewards_tensor / (std + 1e-6)

    normalized_rewards = rewards_tensor.flatten().tolist()

    # ── 2. Split turns into independent training samples ──
    tokens_list = []
    response_lengths = []
    loss_masks = []
    rewards = []
    raw_reward_list = []
    truncated_list = []
    sample_indices = []
    rollout_log_probs_list = []
    has_rollout_log_probs = False
    teacher_log_probs_list = []
    has_teacher_log_probs = False
    require_teacher_log_probs = bool(getattr(args, "use_opd", False) and getattr(args, "opd_type", None) == "sglang")

    for i, sample in enumerate(samples):
        meta = sample.train_metadata
        if meta is None or "turns" not in meta:
            tokens_list.append(sample.tokens)
            response_lengths.append(sample.response_length)
            lm = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
            if sample.remove_sample:
                lm = [0] * sample.response_length
            loss_masks.append(lm)
            rewards.append(normalized_rewards[i])
            raw_reward_list.append(raw_rewards[i])
            truncated_list.append(1 if sample.status == sample.Status.TRUNCATED else 0)
            sample_indices.append(sample.index)
            if sample.rollout_log_probs is not None:
                rollout_log_probs_list.append(sample.rollout_log_probs)
                has_rollout_log_probs = True
            teacher_log_probs = _extract_teacher_log_probs_from_reward(sample, sample.response_length)
            if teacher_log_probs is not None:
                teacher_log_probs_list.append(teacher_log_probs)
                has_teacher_log_probs = True
            elif require_teacher_log_probs:
                raise ValueError(f"OPD requires teacher_log_probs for sample index={sample.index}")
            continue

        turns = meta["turns"]
        # Divide by T_i so that summing over turns gives (1/T_i) * sum_t,
        # matching the J_Flow-GRPO objective which averages across turns.
        norm_reward = normalized_rewards[i] / len(turns)
        is_truncated = 1 if sample.status == sample.Status.TRUNCATED else 0
        teacher_log_probs_by_turn = _extract_teacher_log_probs_by_turn(sample, turns)
        if require_teacher_log_probs and (
            teacher_log_probs_by_turn is None or len(teacher_log_probs_by_turn) != len(turns)
        ):
            got = None if teacher_log_probs_by_turn is None else len(teacher_log_probs_by_turn)
            raise ValueError(
                f"OPD requires one teacher_log_probs list per AgentFlow turn "
                f"(sample index={sample.index}, expected={len(turns)}, got={got})"
            )

        for turn_idx, turn in enumerate(turns):
            tokens_list.append(turn["tokens"])
            response_lengths.append(turn["response_length"])
            lm = list(turn["loss_mask"])
            if sample.remove_sample:
                lm = [0] * turn["response_length"]
            loss_masks.append(lm)
            rewards.append(norm_reward)
            raw_reward_list.append(raw_rewards[i])
            truncated_list.append(is_truncated)
            sample_indices.append(sample.index)
            if turn.get("rollout_log_probs"):
                rollout_log_probs_list.append(turn["rollout_log_probs"])
                has_rollout_log_probs = True
            if teacher_log_probs_by_turn is not None:
                teacher_log_probs = teacher_log_probs_by_turn[turn_idx]
                if len(teacher_log_probs) != turn["response_length"]:
                    raise ValueError(
                        f"teacher_log_probs length {len(teacher_log_probs)} != "
                        f"turn response_length {turn['response_length']} "
                        f"(sample index={sample.index}, turn={turn_idx})"
                    )
                teacher_log_probs_list.append(teacher_log_probs)
                has_teacher_log_probs = True

    # ── 3. Trim to global_batch_size multiple ──
    # After turn expansion the sample count is no longer guaranteed to be
    # divisible by global_batch_size.  The training data-iterator requires
    # num_local_samples % (global_batch_size / dp_size) == 0, which is
    # satisfied when the total is a multiple of global_batch_size.
    gbs = args.global_batch_size
    total = len(tokens_list)
    trim_to = (total // gbs) * gbs
    if trim_to == 0:
        trim_to = total
    if trim_to < total:
        logger.info(
            "custom_convert: trimming expanded samples from %d to %d "
            "(global_batch_size=%d)",
            total, trim_to, gbs,
        )
        tokens_list = tokens_list[:trim_to]
        response_lengths = response_lengths[:trim_to]
        loss_masks = loss_masks[:trim_to]
        rewards = rewards[:trim_to]
        raw_reward_list = raw_reward_list[:trim_to]
        truncated_list = truncated_list[:trim_to]
        sample_indices = sample_indices[:trim_to]
        if has_rollout_log_probs:
            rollout_log_probs_list = rollout_log_probs_list[:trim_to]
        if has_teacher_log_probs:
            teacher_log_probs_list = teacher_log_probs_list[:trim_to]

    train_data = {
        "tokens": tokens_list,
        "response_lengths": response_lengths,
        "loss_masks": loss_masks,
        "rewards": rewards,
        "raw_reward": raw_reward_list,
        "truncated": truncated_list,
        "sample_indices": sample_indices,
    }
    if has_rollout_log_probs:
        train_data["rollout_log_probs"] = rollout_log_probs_list
    if has_teacher_log_probs:
        if len(teacher_log_probs_list) != len(tokens_list):
            if require_teacher_log_probs:
                raise ValueError(
                    f"OPD teacher_log_probs count {len(teacher_log_probs_list)} "
                    f"does not match expanded sample count {len(tokens_list)}"
                )
            logger.warning(
                "custom_convert: dropping partial teacher_log_probs (%d/%d)",
                len(teacher_log_probs_list),
                len(tokens_list),
            )
        else:
            train_data["teacher_log_probs"] = teacher_log_probs_list

    return train_data


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the reward response (which contains sglang's logprob output)
    2. Trims them to match the response length
    3. Stores them in sample.teacher_log_probs for OPD KL penalty computation
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO

    Note: The reward_func calls the teacher server which returns token-level log-probs.
    For pure on-policy distillation without task rewards, we return 0.0 for each sample.
    The actual learning signal comes from the OPD KL penalty applied in compute_advantages_and_returns.
    """
    scalar_rewards = [_get_scalar_reward(args, sample) for sample in samples]

    for sample in samples:
        t_log_probs = _extract_teacher_log_probs_from_reward(sample, sample.response_length)
        if t_log_probs is None:
            raise ValueError(f"OPD requires teacher_log_probs for sample index={sample.index}")
        sample.teacher_log_probs = torch.tensor(t_log_probs, dtype=torch.float32)

    # Return scalar rewards for GRPO/PPO advantage estimator
    # For pure on-policy distillation, we use 0.0 as the task reward.
    # The learning signal comes entirely from the OPD KL penalty.
    # If you have task rewards, you can add them here.
    return scalar_rewards, scalar_rewards
