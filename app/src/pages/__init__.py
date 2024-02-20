from ._1_bellingcat_titles import Bellingcat
from ..types import Page

from typing import Dict, Type


PAGE_MAP: Dict[str, Type[Page]] = {
    "Bellingcat": Bellingcat,
}

__all__ = ["PAGE_MAP"]
