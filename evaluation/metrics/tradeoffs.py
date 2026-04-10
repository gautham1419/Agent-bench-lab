def compute_tradeoffs(master_records):

    results = []

    for r in master_records:

        energy = r.get("energy_mean", 0)
        successes = r.get("successes", 0)
        total_tasks = r.get("total_tasks", 0)
        success_rate = r.get("success_rate_mean", 0)
        avg_tool_calls = r.get("avg_tool_calls_mean", 0)

        energy_per_task = energy / total_tasks if total_tasks else 0
        energy_per_success = energy / successes if successes else 0
        success_per_energy = successes / energy if energy else 0

        tool_calls_per_success = (
            avg_tool_calls / success_rate if success_rate else 0
        )

        results.append({
            "model": r["model"],
            "size": r["size"],
            "quant": r["quant"],
            "domain": r["domain"],
            "success_rate_mean": success_rate,
            "energy_per_task": energy_per_task,
            "energy_per_success": energy_per_success,
            "success_per_energy": success_per_energy,
            "tool_calls_per_success": tool_calls_per_success
        })

    return results