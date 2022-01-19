""" Utils functions for models """
from datetime import datetime


def datetime_must_have_timezone(cls, v: datetime):
    """Enforce that this variable must have a timezone"""
    if v.tzinfo is None:
        raise ValueError(f"{v} must have a timezone, for cls {cls}")
    return v


def convert_to_camelcase(snake_str: str) -> str:
    """Converts a given snake_case string into camelCase"""
    first, *others = snake_str.split("_")
    return "".join([first.lower(), *map(str.title, others)])
