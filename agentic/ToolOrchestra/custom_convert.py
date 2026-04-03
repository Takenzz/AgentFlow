"""
ToolOrchestra multi-turn custom convert with preference-weighted GRPO reward.

Implements the full reward pipeline from the original ToolOrchestra:

1. Extract per-rollout features (correctness, cost, latency, tool_counts)
2. Group rollouts by example (n_samples_per_prompt consecutive samples)
3. Per-example min-max normalize all features across rollouts
4. Compute preference-weighted reward using pref_vec
5. Per-example GRPO standardization: (r - mean) / (std + eps), clip to [-3, 3]
6. Filter out examples with std < 0.1 (no learning signal)
7. Split multi-turn trajectories into independent training samples

Usage:
     --custom-convert-samples-to-train-data-path custom_convert.custom_convert
"""

import logging

import torch

logger = logging.getLogger(__name__)

MIN_STD_THRESHOLD = 0.1
REWARD_CLIP = 3.0


def _get_reward_features(sample) -> dict:
    """
    Read raw features from sample.train_metadata["reward_features"].
    Falls back to extracting from turns if not pre-computed.
    """
    meta = sample.train_metadata
    if isinstance(meta, dict) and "reward_features" in meta:
        return meta["reward_features"]

    features = {
        "correctness": sample.get_reward_value(None) if hasattr(sample, "get_reward_value") else 0.0,
        "total_cost": 0.0,
        "total_latency": 0.0,
        "tool_counts": {},
    }
    if isinstance(meta, dict) and "turns" in meta:
        for t in meta["turns"]:
            role = t.get("role_name")
            if role:
                features["tool_counts"][role] = features["tool_counts"].get(role, 0) + 1
            features["total_cost"] += t.get("input_tokens", 0) + t.get("output_tokens", 0)
            features["total_latency"] += t.get("latency_ms", 0.0)
    return features


def _compute_preference_rewards(
    group_features: list[dict],
    group_pref_vecs: list[dict],
) -> list[float]:
    """
    Compute preference-weighted rewards for a group of rollouts of the same
    example, using per-feature min-max normalization across the group.

    Mirrors the original generation_quick3.py logic:
    - Incorrect rollouts get reward = 0
    - Correct rollouts get reward = sum(pref_vec[f] * normalized(f))
      where features include: tool_counts per role, accuracy, cost, latency
    - cost and latency are negated (lower is better)
    """
    n = len(group_features)
    if n == 0:
        return []

    all_feature_keys: set[str] = set()
    feature_vectors: list[dict[str, float]] = []

    for feat in group_features:
        fv: dict[str, float] = {}
        for role, count in feat["tool_counts"].items():
            fv[role] = float(count)
            all_feature_keys.add(role)

        fv["accuracy"] = feat["correctness"]
        fv["cost"]     = -feat["total_cost"]
        fv["latency"]  = -feat["total_latency"]
        all_feature_keys.update(["accuracy", "cost", "latency"])
        feature_vectors.append(fv)

    feat_min: dict[str, float] = {}
    feat_max: dict[str, float] = {}
    for key in all_feature_keys:
        values = [fv.get(key, 0.0) for fv in feature_vectors]
        feat_min[key] = min(values)
        feat_max[key] = max(values)

    rewards = []
    for i, feat in enumerate(group_features):
        if feat["correctness"] < 0.5:
            rewards.append(0.0)
            continue

        pref_vec = group_pref_vecs[i] if i < len(group_pref_vecs) else {}
        reward = 0.0
        fv = feature_vectors[i]

        for key in all_feature_keys:
            if feat_max[key] > feat_min[key]:
                normalized = (fv.get(key, 0.0) - feat_min[key]) / (feat_max[key] - feat_min[key])
                weight = float(pref_vec.get(key, 0.0))
                reward += weight * normalized

        rewards.append(reward)

    return rewards


def _grpo_normalize_and_filter(
    rewards: list[float],
    n: int,
) -> tuple[list[float], list[bool]]:
    """
    Per-example GRPO standardization and filtering.

    Args:
        rewards: flat list of rewards, every consecutive n belong to the same example
        n:       n_samples_per_prompt

    Returns:
        normalized_rewards: standardized, clipped to [-REWARD_CLIP, REWARD_CLIP]
        keep_mask: True if the example group has enough variance to train on
    """
    num_examples = len(rewards) // n
    remainder = len(rewards) - num_examples * n

    normalized = []
    keep_mask = []

    for g in range(num_examples):
        group = rewards[g * n : (g + 1) * n]
        mean = sum(group) / len(group)
        var = sum((r - mean) ** 2 for r in group) / len(group)
        std = var ** 0.5

        has_signal = std > MIN_STD_THRESHOLD
        for r in group:
            if has_signal:
                nr = (r - mean) / (std + 1e-6)
                nr = max(-REWARD_CLIP, min(REWARD_CLIP, nr))
            else:
                nr = 0.0
            normalized.append(nr)
            keep_mask.append(has_signal)

    for j in range(remainder):
        normalized.append(0.0)
        keep_mask.append(False)

    return normalized, keep_mask


def _has_valid_answer(sample) -> bool:
    """
    Check if the rollout produced a valid answer.
    - QA: final_output is non-empty
    - func_call: tau2 reward_info exists (reward > -1)
    """
    meta = sample.metadata if isinstance(sample.metadata, dict) else {}
    category = meta.get("category", "qa")
    train_meta = sample.train_metadata if isinstance(sample.train_metadata, dict) else {}
    turns = train_meta.get("turns", [])

    if category == "func_call":
        for t in turns:
            ri = t.get("tau2_reward_info")
            if ri is not None:
                reward_val = ri.get("reward", 0) if isinstance(ri, dict) else ri
                return float(reward_val) > -1
        return False

    final_output = meta.get("final_output", "") or ""
    return len(final_output.strip()) > 0


def _turn_has_valid_format(turn: dict) -> bool:
    """
    Check if a turn's model output was in valid tool-call format.
    A turn is valid if it has a recognized tool_name or produced a response.
    """
    if turn.get("tool_name"):
        return True
    resp = turn.get("_response", "")
    return len(resp.strip()) > 0


def custom_convert(args, samples):
    n = getattr(args, "n_samples_per_prompt", 1)

    # ── 1. Extract features and compute preference-weighted rewards ──
    all_features = []
    all_pref_vecs = []
    valid_answer_mask = []
    for sample in samples:
        all_features.append(_get_reward_features(sample))
        meta = sample.metadata if isinstance(sample.metadata, dict) else {}
        all_pref_vecs.append(meta.get("pref_vec", {}))
        valid_answer_mask.append(_has_valid_answer(sample))

    pref_rewards = []
    num_groups = len(samples) // n
    for g in range(num_groups):
        group_feat = all_features[g * n : (g + 1) * n]
        group_pref = all_pref_vecs[g * n : (g + 1) * n]
        group_rewards = _compute_preference_rewards(group_feat, group_pref)
        pref_rewards.extend(group_rewards)

    for j in range(len(samples) - num_groups * n):
        pref_rewards.append(all_features[num_groups * n + j]["correctness"])

    # ── 2. GRPO per-example normalization + filtering ──
    normalized_rewards, keep_mask = _grpo_normalize_and_filter(pref_rewards, n)

    raw_rewards = pref_rewards

    # ── 3. Split turns into independent training samples ──
    tokens_list = []
    response_lengths = []
    loss_masks = []
    rewards = []
    raw_reward_list = []
    truncated_list = []
    sample_indices = []
    rollout_log_probs_list = []
    has_rollout_log_probs = False

    for i, sample in enumerate(samples):
        meta = sample.train_metadata
        sample_should_mask = (
            not keep_mask[i]
            or sample.remove_sample
            or not valid_answer_mask[i]
        )

        if meta is None or "turns" not in meta:
            prompt_len = len(sample.tokens) - sample.response_length
            if prompt_len < 1 or sample.response_length < 1:
                logger.warning(
                    "custom_convert: skipping sample with prompt_len=%d response_len=%d",
                    prompt_len, sample.response_length,
                )
                continue
            tokens_list.append(sample.tokens)
            response_lengths.append(sample.response_length)
            lm = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
            if sample_should_mask:
                lm = [0] * sample.response_length
            loss_masks.append(lm)
            rewards.append(normalized_rewards[i])
            raw_reward_list.append(raw_rewards[i])
            truncated_list.append(1 if sample.status == sample.Status.TRUNCATED else 0)
            sample_indices.append(sample.index)
            if sample.rollout_log_probs is not None:
                rollout_log_probs_list.append(sample.rollout_log_probs)
                has_rollout_log_probs = True
            continue

        turns = meta["turns"]
        norm_reward = normalized_rewards[i]
        is_truncated = 1 if sample.status == sample.Status.TRUNCATED else 0

        for turn in turns:
            # Skip degenerate turns: prompt_length must be >= 1 for data.py's loss_mask padding
            # (formula: F.pad(loss_mask, (prompt_length - 1, 1))) to work without error.
            prompt_len = len(turn["tokens"]) - turn["response_length"]
            if prompt_len < 1 or turn["response_length"] < 1:
                logger.warning(
                    "custom_convert: skipping turn with prompt_len=%d response_len=%d",
                    prompt_len, turn["response_length"],
                )
                continue
            turn_masked = sample_should_mask or not _turn_has_valid_format(turn)
            tokens_list.append(turn["tokens"])
            response_lengths.append(turn["response_length"])
            lm = turn["loss_mask"]
            if turn_masked:
                lm = [0] * turn["response_length"]
            loss_masks.append(lm)
            rewards.append(norm_reward)
            raw_reward_list.append(raw_rewards[i])
            truncated_list.append(is_truncated)
            sample_indices.append(sample.index)
            if turn.get("rollout_log_probs"):
                rollout_log_probs_list.append(turn["rollout_log_probs"])
                has_rollout_log_probs = True

    # ── 4. Trim to global_batch_size multiple ──
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

    return train_data
