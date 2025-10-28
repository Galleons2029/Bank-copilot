# -*- coding: utf-8 -*-
# @Time    : 2025/07/09 3:28 AM
# @Author  : Galleons
# @File    : extractor_base.py

"""
文件加载器抽象接口
"""

from abc import ABC, abstractmethod


class BaseExtractor(ABC):
    """Interface for extract files."""

    @abstractmethod
    def extract(self):
        raise NotImplementedError
