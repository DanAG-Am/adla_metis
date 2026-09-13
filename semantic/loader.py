from pathlib import Path

import yaml

class SemanticModel:

    def __init__(
        self,
        schema_path,
        metrics_path,
    ):
        self.schema_path = Path(schema_path)
        self.metrics_path = Path(metrics_path)

        self.schema = self._load(
            self.schema_path
        )

        self.metrics = self._load(
            self.metrics_path
        )

    @staticmethod
    def _load(path):

        with open(
            path,
            "r",
            encoding="utf-8",
        ) as file:

            return yaml.safe_load(file)

    def get_metric(self, name):

        metrics = self.metrics.get(
            "metrics",
            {},
        )

        return metrics.get(name)

    def get_dimension(self, name):

        dimensions = self.metrics.get(
            "dimensions",
            {},
        )

        return dimensions.get(name)

    def list_metrics(self):

        return list(
            self.metrics.get(
                "metrics",
                {},
            ).keys()
        )

    def list_dimensions(self):

        return list(
            self.metrics.get(
                "dimensions",
                {},
            ).keys()
        )
