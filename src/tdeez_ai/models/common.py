"""Enums shared across domains."""

from enum import StrEnum


class FurnitureType(StrEnum):
    DINING_TABLE = "dining_table"
    COFFEE_TABLE = "coffee_table"
    DESK = "desk"


class Species(StrEnum):
    BLACK_WALNUT = "black_walnut"
    BIG_LEAF_MAPLE = "big_leaf_maple"
    WHITE_OAK = "white_oak"
    CHERRY = "cherry"
    ACACIA = "acacia"


class Availability(StrEnum):
    AVAILABLE = "available"
    RESERVED = "reserved"
    SOLD = "sold"
