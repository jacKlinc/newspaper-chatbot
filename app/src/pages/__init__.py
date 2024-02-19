from ._1_file_upload import FileUpload
from ..types import Page

from typing import Dict, Type


PAGE_MAP: Dict[str, Type[Page]] = {
    "File Upload": FileUpload,
}

__all__ = ["PAGE_MAP"]
