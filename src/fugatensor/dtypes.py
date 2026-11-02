"""Public element types with stable IR and serialization spellings."""
from enum import Enum


class DType(str, Enum):
    bool = "bool"
    int32 = "i32"
    int64 = "i64"
    fp32 = "f32"
    fp16 = "f16"
    bf16 = "bf16"

    def __str__(self):
        return self.value

    def __repr__(self):
        return f"ft.{self.name}"


bool = DType.bool
int32 = DType.int32
int64 = DType.int64
fp32 = DType.fp32
fp16 = DType.fp16
bf16 = DType.bf16

__all__ = ["DType", "bool", "int32", "int64", "fp32", "fp16", "bf16"]
