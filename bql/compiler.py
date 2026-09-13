from semantic.loader import SemanticModel

from bql.models import Query

class BQLCompiler:

    def __init__(
        self,
        semantic_model: SemanticModel,
    ):
        self.semantic_model = semantic_model

    def compile(self, query: Query):

        metric = self.semantic_model.get_metric(
            query.metric
        )

        metric_expression = metric[
            "expression"
        ]

        select_parts = []

        group_by_parts = []

        for dimension in query.group_by:

            definition = (
                self.semantic_model
                .get_dimension(dimension)
            )

            table = definition["entity"]
            field = definition["field"]

            qualified_field = (
                f"{table}.{field}"
            )

            select_parts.append(
                qualified_field
            )

            group_by_parts.append(
                qualified_field
            )

        select_parts.append(
            f"{metric_expression} AS "
            f"{query.metric}"
        )

        sql = "SELECT\n    "

        sql += ",\n    ".join(
            select_parts
        )

        sql += "\nFROM order_items"

        sql += """
JOIN orders
    ON order_items.order_id = orders.order_id
"""

        customer_needed = (
            "region" in query.group_by
            or "segment" in query.group_by
            or any(
                f.field in {
                    "region",
                    "segment",
                }
                for f in query.filters
            )
        )

        if customer_needed:

            sql += """
JOIN customers
    ON orders.customer_id = customers.customer_id
"""

        product_needed = (
            "category" in query.group_by
            or "product" in query.group_by
            or any(
                f.field in {
                    "category",
                    "product",
                }
                for f in query.filters
            )
        )

        if product_needed:

            sql += """
JOIN products
    ON order_items.product_id = products.product_id
"""

        if query.filters:

            conditions = []

            for filter_item in query.filters:

                dimension = (
                    self.semantic_model
                    .get_dimension(
                        filter_item.field
                    )
                )

                table = dimension["entity"]
                field = dimension["field"]

                qualified_field = (
                    f"{table}.{field}"
                )

                value = filter_item.value

                if isinstance(value, str):
                    escaped = value.replace(
                        "'",
                        "''",
                    )

                    value_sql = (
                        f"'{escaped}'"
                    )
                else:
                    value_sql = str(value)

                conditions.append(
                    f"{qualified_field} "
                    f"{filter_item.operator} "
                    f"{value_sql}"
                )

            sql += "\nWHERE "
            sql += "\n  AND ".join(
                conditions
            )

        if group_by_parts:

            sql += "\nGROUP BY "
            sql += ", ".join(
                group_by_parts
            )

            sql += (
                f"\nORDER BY "
                f"{query.metric} DESC"
            )

        if query.limit is not None:

            sql += (
                f"\nLIMIT {query.limit}"
            )

        sql += ";"

        return sql
