from bql.models import Query
from semantic.loader import SemanticModel

class BQLValidationError(Exception):
    pass

class BQLValidator:

    def __init__(self, semantic_model: SemanticModel):
        self.semantic_model = semantic_model

    def validate(self, query: Query):

        metric = self.semantic_model.get_metric(
            query.metric
        )

        if metric is None:
            raise BQLValidationError(
                f"Unknown metric: {query.metric}"
            )

        for dimension in query.group_by:

            definition = (
                self.semantic_model
                .get_dimension(dimension)
            )

            if definition is None:
                raise BQLValidationError(
                    f"Unknown dimension: {dimension}"
                )

        for filter_item in query.filters:

            definition = (
                self.semantic_model
                .get_dimension(filter_item.field)
            )

            if definition is None:
                raise BQLValidationError(
                    f"Unknown filter field: "
                    f"{filter_item.field}"
                )

            allowed_operators = {
                "=",
                "!=",
                ">",
                ">=",
                "<",
                "<=",
            }

            if filter_item.operator not in allowed_operators:
                raise BQLValidationError(
                    f"Unsupported operator: "
                    f"{filter_item.operator}"
                )

        if query.limit is not None:

            if query.limit <= 0:
                raise BQLValidationError(
                    "Limit must be greater than zero."
                )

        return True
