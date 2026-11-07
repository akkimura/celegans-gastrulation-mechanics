import csv
from itertools import combinations, product
from pathlib import Path


INPUT_CSV = Path("hmr1_intensity_data.csv")

EA_MS_COLUMNS = ["Ea_MSap", "Ea_MSpa", "Ea_MSpp"]
EA_AB_COLUMNS = ["Ea_ABplpa", "Ea_ABplpp"]

EP_P4_COLUMN = "Ep_P4"
EP_AB_COLUMNS = ["Ep_AB1", "Ep_AB2"]


def compare_values(target_value, control_value):
    """target側から見た勝敗を返す。"""
    if target_value > control_value:
        return "win"
    if target_value < control_value:
        return "loss"
    return "tie"


def count_ea_games(row):
    """
    Ea-MS系列3境界 × Ea-AB系列2境界 = 1個体6試合。
    MS側から見た勝敗を数える。
    """
    result = {"wins": 0, "losses": 0, "ties": 0}

    for ms_column in EA_MS_COLUMNS:
        for ab_column in EA_AB_COLUMNS:
            outcome = compare_values(
                float(row[ms_column]),
                float(row[ab_column]),
            )
            result_key = {
                "win": "wins",
                "loss": "losses",
                "tie": "ties",
            }[outcome]
            result[result_key] += 1

    return result


def count_ep_games(row):
    """
    Ep-P4 1境界 × Ep-AB系列2境界 = 1個体2試合。
    P4側から見た勝敗を数える。
    """
    result = {"wins": 0, "losses": 0, "ties": 0}

    p4_value = float(row[EP_P4_COLUMN])

    for ab_column in EP_AB_COLUMNS:
        outcome = compare_values(
            p4_value,
            float(row[ab_column]),
        )
        result_key = {
            "win": "wins",
            "loss": "losses",
            "tie": "ties",
        }[outcome]
        result[result_key] += 1

    return result


def add_results(total, result):
    for key in ("wins", "losses", "ties"):
        total[key] += result[key]


def count_pairwise_score(target_values, control_values):
    """勝ちを1、同値を0.5として系列間の全比較スコアを返す。"""
    score = 0.0

    for target_value in target_values:
        for control_value in control_values:
            outcome = compare_values(target_value, control_value)
            if outcome == "win":
                score += 1.0
            elif outcome == "tie":
                score += 0.5

    return score


def exact_permutation_p(rows, value_columns, target_count, observed_score):
    """
    各胚内でtargetラベルを付ける境界を完全列挙し、
    観測スコア以上となる片側exact p値を返す。
    """
    per_embryo_scores = []

    for row in rows:
        values = [float(row[column]) for column in value_columns]
        assignment_scores = []

        for target_indices in combinations(range(len(values)), target_count):
            target_index_set = set(target_indices)
            target_values = [
                value for index, value in enumerate(values)
                if index in target_index_set
            ]
            control_values = [
                value for index, value in enumerate(values)
                if index not in target_index_set
            ]
            assignment_scores.append(
                count_pairwise_score(target_values, control_values)
            )

        per_embryo_scores.append(assignment_scores)

    extreme_count = 0
    total_assignments = 0

    for scores in product(*per_embryo_scores):
        total_assignments += 1
        if sum(scores) >= observed_score:
            extreme_count += 1

    return extreme_count / total_assignments, extreme_count, total_assignments


def main():
    ea_total = {"wins": 0, "losses": 0, "ties": 0}
    ep_total = {"wins": 0, "losses": 0, "ties": 0}

    with INPUT_CSV.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))

    if len(rows) != 5:
        raise ValueError(
            f"5サンプルを想定していますが、CSVには{len(rows)}サンプルあります。"
        )

    print("個体別の対戦成績")
    print("-" * 65)

    for row in rows:
        embryo = row["embryo"]

        ea_result = count_ea_games(row)
        ep_result = count_ep_games(row)

        add_results(ea_total, ea_result)
        add_results(ep_total, ep_result)

        print(
            f"{embryo}: "
            f"Ea-MS = {ea_result['wins']}勝"
            f"{ea_result['losses']}敗"
            f"{ea_result['ties']}分、"
            f"Ep-P4 = {ep_result['wins']}勝"
            f"{ep_result['losses']}敗"
            f"{ep_result['ties']}分"
        )

    print("\n5個体の合計")
    print("-" * 65)

    print(
        f"Ea-MS系列: "
        f"{ea_total['wins']}勝"
        f"{ea_total['losses']}敗"
        f"{ea_total['ties']}分 "
        f"（全30試合）"
    )

    print(
        f"Ep-P4: "
        f"{ep_total['wins']}勝"
        f"{ep_total['losses']}敗"
        f"{ep_total['ties']}分 "
        f"（全10試合）"
    )

    ea_observed_score = ea_total["wins"] + 0.5 * ea_total["ties"]
    ep_observed_score = ep_total["wins"] + 0.5 * ep_total["ties"]

    ea_p, ea_extreme, ea_assignments = exact_permutation_p(
        rows,
        EA_MS_COLUMNS + EA_AB_COLUMNS,
        len(EA_MS_COLUMNS),
        ea_observed_score,
    )
    ep_p, ep_extreme, ep_assignments = exact_permutation_p(
        rows,
        [EP_P4_COLUMN] + EP_AB_COLUMNS,
        1,
        ep_observed_score,
    )

    print("\n片側exact permutation test")
    print("-" * 65)
    print(
        f"Ea側: p = {ea_p:.5f} "
        f"（{ea_extreme}/{ea_assignments}通り）"
    )
    print(
        "Ea側の判定:",
        "有意（p < 0.05）" if ea_p < 0.05 else "非有意（p >= 0.05）",
    )
    print(
        f"Ep側: p = {ep_p:.5f} "
        f"（{ep_extreme}/{ep_assignments}通り）"
    )
    print(
        "Ep側の判定:",
        "有意（p < 0.05）" if ep_p < 0.05 else "非有意（p >= 0.05）",
    )


if __name__ == "__main__":
    main()
