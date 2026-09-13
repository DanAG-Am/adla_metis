# Business Brain

A local, domain-specific language model and semantic query engine for business analytics. Business Brain translates natural-language questions into validated business queries and SQL over a local SQLite dataset.

## Setup

```bash
python -m pip install -e .
```

Generate the demo database with:

```bash
python data/generate_demo.py
```

Run the BQL example with:

```bash
python bql/test_bql.py
```

The interactive entry point is available as `business-brain` after installation.
