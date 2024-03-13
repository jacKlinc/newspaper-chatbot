from abc import ABC, abstractmethod
from dataclasses import dataclass

from enum import Enum


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
