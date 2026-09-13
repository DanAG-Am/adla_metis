from dataclasses import dataclass, field
from typing import Any, Optional

@dataclass
class Filter:
    field: str
    operator: str
    value: Any

@dataclass
class Query:
    metric: str
    group_by: list[str] = field(default_factory=list)
    filters: list[Filter] = field(default_factory=list)
    limit: Optional[int] = None
