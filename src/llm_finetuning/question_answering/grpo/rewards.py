def compute_hotpot_reward(pred, group):
    answer_match = 1.0 if group["answer"].lower() in pred.lower() else 0.5
    sup_facts = group["supporting_facts"]
    title_coverage = sum(1 for title in set(sup_facts["title"]) if title in pred)
    return answer_match + (0.1 * title_coverage)


def compute_musique_reward(pred, group):
    all_answers = [group["answer"]] + group["aliases"]
    normalized_pred = pred.lower().strip()
    if any(ans.lower().strip() == normalized_pred for ans in all_answers):
        return 1.0
    if any(ans.lower().strip() in normalized_pred for ans in all_answers):
        return 0.8
    decomp_bonus = 0.0
    for subq in group["decomposition"]:
        if subq.lower() in normalized_pred:
            decomp_bonus += 0.1
    return min(0.5, decomp_bonus)
