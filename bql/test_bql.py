from semantic.loader import SemanticModel

from bql.models import Query
from bql.compiler import BQLCompiler
from bql.validator import BQLValidator

def main():

    semantic_model = SemanticModel(
        "semantic/schema.yaml",
        "semantic/metrics.yaml",
    )

    validator = BQLValidator(
        semantic_model
    )

    compiler = BQLCompiler(
        semantic_model
    )

    query = Query(
        metric="revenue",
        group_by=["region"],
    )

    validator.validate(query)

    sql = compiler.compile(query)

    print("Generated SQL:")
    print()
    print(sql)

if __name__ == "__main__":
    main()
