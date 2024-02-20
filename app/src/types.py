from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from enum import Enum


class HttpStatus(Enum):
    unknown = 0
    bad_request_400 = 400
    ok_200 = 200
    too_many_requests_429 = 429


class Page(ABC):
    @abstractmethod
    def write(self):
        pass


@dataclass
class Article:
    publish_date: str
    title: str
    url: str
    articles_text: str
    id: int

    def __getitem__(self, item):
        return getattr(self, item)
