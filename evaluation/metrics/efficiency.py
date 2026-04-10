import json


def compute_efficiency(resource_file):

    metrics = {}

    if resource_file.exists():
        with open(resource_file) as f:
            resource_metrics = json.load(f)

        metrics.update(resource_metrics)

    return metrics