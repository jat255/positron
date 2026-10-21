#
# Copyright (C) 2023-2024 Posit Software, PBC. All rights reserved.
# Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
#
from __future__ import annotations

import contextlib
import copy
import datetime
import inspect
import logging
import math
import numbers
import pydoc
import re
import sys
import types
from abc import ABC, abstractmethod
from collections.abc import (
    Hashable,
    Mapping,
    MutableMapping,
    MutableSequence,
    MutableSet,
    Sequence,
)
from collections.abc import (
    Set as AbstractSet,
)
from inspect import getattr_static
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Collection,
    FrozenSet,
    Generic,
    Iterable,
    Sized,
    Tuple,
    TypeVar,
    Union,
    cast,
)

from .third_party import _numpy, _pandas, _pyodbc, _torch
from .utils import (
    JsonData,
    get_qualname,
    numpy_numeric_scalars,
    safe_isinstance,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl
    import xarray as xr
    from dask.array.core import Array as DaskArray

    # temporary suppress for python 3.12
    with contextlib.suppress(ImportError):
        import torch  # type: ignore [reportMissingImports]

    # python >= 3.10
    with contextlib.suppress(ImportError):
        import ibis


# General display settings
ELLIPSIS = "…"
# Max characters after which the display value of a standalone object is truncated.
MAX_CHARACTERS = 2**16
# Max characters after which the display value of an object in a collection is truncated.
MAX_CHARACTERS_NESTED = 128
# Max number of items included in the display value of a collection, keyed by the level of nesting
# of the collection. Collections nested deeper than specified here will be totally truncated.
MAX_ITEMS_BY_LEVEL = (60, 20)


# Array-specific display settings
ARRAY_MAX_LINE_WIDTH = 1000
ARRAY_THRESHOLD = 20
ARRAY_EDGEITEMS = 9

logger = logging.getLogger(__name__)

#
# Base inspector
#

T = TypeVar("T")


SIMPLER_NAMES = {
    "geopandas.geoseries.GeoSeries": "geopandas.GeoSeries",
    "geopandas.geodataframe.GeoDataFrame": "geopandas.GeoDataFrame",
    "pandas.core.frame.DataFrame": "pandas.DataFrame",
    "pandas.core.series.Series": "pandas.Series",
    "polars.dataframe.frame.DataFrame": "polars.DataFrame",
    "polars.series.series.Series": "polars.Series",
    "polars.internals.series.series.Series": "polars.Series",
    "polars.internals.dataframe.frame.DataFrame": "polars.DataFrame",
    "pandas.core.indexes.base.Index": "pandas.Index",
    "pandas.core.indexes.datetimes.DatetimeIndex": "pandas.DatetimeIndex",
    "pandas.core.indexes.range.RangeIndex": "pandas.RangeIndex",
    "pandas.core.indexes.multi.MultiIndex": "pandas.MultiIndex",
    # Just display Int64Index as pandas.Index, since the former is deprecated since pandas v1.4.0.
    "pandas.core.indexes.numeric.Int64Index": "pandas.Index",
    "duckdb.duckdb.DuckDBPyConnection": "duckdb.DuckDBPyConnection",
    "ibis.expr.types.relations.Table": "ibis.Table",
}


def _remap_ibis_classnames(value):
    # We will provide more nuanced handling of Ibis expressions in the
    # inspector class for ibis.Expr and its many subclasses

    import ibis

    if isinstance(value, ibis.Expr):
        return "ibis.Expr"

    return get_qualname(value)


PACKAGE_REMAPPERS = {
    "ibis": _remap_ibis_classnames,
}


def _get_simplified_qualname(value):
    display_value = get_qualname(value)

    if display_value in SIMPLER_NAMES:
        return SIMPLER_NAMES[display_value]

    top_path = display_value.split(".")[0]

    if top_path in PACKAGE_REMAPPERS:
        return PACKAGE_REMAPPERS[top_path](value)

    return display_value


class PositronInspector(Generic[T]):
    """Base inspector for any type."""

    def __init__(self, value: T) -> None:
        self.value = value

    def get_display_name(self, key: Any) -> str:
        return str(key)

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        return _get_default_display_value(self.value, level=level)

    def get_display_type(self) -> str:
        type_name = type(self.value).__name__

        if isinstance(self.value, Sized):
            length = self.get_length()
            return f"{type_name} [{length}]"

        return type_name

    def get_kind(self) -> str:
        return _get_kind(self.value)

    def get_type_info(self) -> str:
        return get_qualname(type(self.value))

    def get_length(self) -> int:
        return len(self.value) if isinstance(self.value, Sized) else 0

    def get_size(self) -> int:
        return sys.getsizeof(self.value)

    def has_children(self) -> bool:
        return self.get_length() > 0

    def has_child(self, key: Any) -> bool:  # noqa: ARG002
        return False

    def get_child(self, key: Any) -> Any:  # noqa: ARG002
        raise TypeError(f"get_child() is not implemented for type: {type(self.value)}")

    def get_children(self) -> Iterable[Any]:
        raise TypeError(f"get_children() is not implemented for type: {type(self.value)}")

    def has_viewer(self) -> bool:
        return False

    def is_mutable(self) -> bool:
        # Without any additional information it's safer to assume that the inspector is mutable,
        # which also disables deepcopying. See the `deepcopy` docstring for more.
        return True

    def get_comparison_cost(self) -> int:
        return self.get_size()

    def equals(self, value: T) -> bool:
        try:
            return self.value == value
        except ValueError:
            # If a collection has a nested value that does not support
            # bool(x == y) (like NumPy arrays or other array-like
            # objects), this will error
            return False

    def deepcopy(self) -> T:
        """
        Special deepcopy.

        Inspectors of mutable types are not deepcopyable by default, since deepcopying may have
        unintended side-effects (see https://github.com/posit-dev/positron/issues/2833). To support
        deepcopying, sub-classes must override `deepcopy`.
        """
        if self.is_mutable():
            raise copy.Error(f"Deepcopying is not supported for type: {type(self.value)}")
        # If the value is immutable, the deepcopy may reference the same value.
        return self.value

    def to_html(self) -> str:
        return repr(self.value)

    def to_plaintext(self) -> str:
        return repr(self.value)

    def to_json(self) -> JsonData:
        return {"type": self.type_to_json(), "data": self.value_to_json()}

    def type_to_json(self) -> str:
        return self.get_type_info()

    def value_to_json(self) -> JsonData:
        raise NotImplementedError(
            f"value_to_json() is not implemented for this type. type: {type(self.value)}"
        )

    @classmethod
    def from_json(cls, json_data: JsonData) -> T:
        if not isinstance(json_data, dict):
            raise ValueError(f"Expected json_data to be dict, got {json_data}")

        if not isinstance(json_data["type"], str):
            raise ValueError(f"Expected json_data['type'] to be str, got {json_data['type']}")

        # TODO(pyright): cast shouldn't be necessary, recheck in a future version of pyright
        return cls.value_from_json(cast("str", json_data["type"]), json_data["data"])

    @classmethod
    def value_from_json(cls, type_name: str, data: JsonData) -> T:
        raise NotImplementedError(
            f"value_from_json() is not implemented for this type. type_name: {type_name}, data: {data}"
        )


#
# Scalars
#


class NoneInspector(PositronInspector[type(None)]):
    def is_mutable(self) -> bool:
        return False

    def value_to_json(self) -> JsonData:
        return None

    @classmethod
    def value_from_json(cls, type_name: str, data: JsonData) -> None:  # noqa: ARG003
        return None


class BooleanInspector(PositronInspector[bool]):
    def is_mutable(self) -> bool:
        return False

    def get_kind(self) -> str:
        return "boolean"

    def value_to_json(self) -> JsonData:
        return self.value

    @classmethod
    def value_from_json(cls, type_name: str, data: JsonData) -> bool:  # noqa: ARG003
        if not isinstance(data, bool):
            raise ValueError(f"Expected data to be bool, got {data}")

        return data


Bytes = TypeVar("Bytes", bytes, "bytearray", "memoryview")


class BytesInspector(PositronInspector[Bytes]):
    def is_mutable(self) -> bool:
        if isinstance(self.value, (bytearray, memoryview)):
            return True
        if isinstance(self.value, bytes):
            return False
        return super().is_mutable()

    def deepcopy(self) -> Bytes:
        if isinstance(self.value, bytearray):
            # Bytearrays are mutable, but can only hold bytes, so it's safe to use the default
            # deepcopy implementation.
            return copy.deepcopy(self.value)
        return super().deepcopy()

    def get_kind(self) -> str:
        return "bytes"

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        if isinstance(self.value, memoryview):
            return _get_default_display_value(self.value, level=level)
        return _get_string_display_value(self.value, level=level)

    def has_children(self) -> bool:
        return False

    def value_to_json(self) -> JsonData:
        if isinstance(self.value, bytes):
            return self.value.decode()
        return super().value_to_json()

    @classmethod
    def value_from_json(cls, type_name: str, data: JsonData) -> Bytes:
        if isinstance(data, str):
            return cast("Bytes", data.encode())
        return super().value_from_json(type_name, data)


#
# Objects
#


class ObjectInspector(PositronInspector[T], ABC):
    def has_child(self, key: str) -> bool:
        return hasattr(self.value, key)

    def get_length(self) -> int:
        if isinstance(self.value, property):
            return 0
        return len([p for p in dir(self.value) if not (p.startswith("_"))])

    def get_children(self):
        return (p for p in dir(self.value) if not (p.startswith("_")))

    def get_child(self, key: str) -> Any:
        # If the attr is a method, getattr_static will return the wrapped function, but we want the method
        attr = getattr_static(self.value, key)
        if callable(attr):
            return getattr(self.value, key)
        else:
            return attr


class ClassInspector(ObjectInspector[type]):
    def get_kind(self) -> str:
        return "class"

    def value_to_json(self) -> JsonData:
        return str(self.value)

    @classmethod
    def value_from_json(cls, type_name: str, data: JsonData) -> type:  # noqa: ARG003
        if not isinstance(data, str):
            raise ValueError(f"Expected data to be str, got {data}")

        pattern = "(?<=<class ').*(?='>)"
        match = re.search(pattern, data)
        if match is None:
            raise ValueError(f"Could not find class name in {data}")
        # pydoc.locate will work for both built-in classes as well as any classes on the path
        class_name = pydoc.locate(match.group(0))
        if not isinstance(class_name, type):
            raise ValueError(f"Could not locate a type named {data}")
        return class_name


class PropertyInspector(PositronInspector[property]):
    def is_mutable(self) -> bool:
        return False


class FunctionInspector(PositronInspector[Callable]):
    def is_mutable(self) -> bool:
        return False

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        sig = inspect.signature(self.value) if callable(self.value) else "()"
        return _maybe_truncate_string(f"{self.value.__qualname__}{sig}", level=level)

    def get_kind(self) -> str:
        return "function"


NT = TypeVar("NT", numbers.Number, "np.number")


class NumberInspector(PositronInspector[NT], ABC):
    def get_display_type(self) -> str:
        ty = type(self.value)
        mod = ty.__module__
        name = ty.__name__
        if mod == "builtins":
            return name
        else:
            return f"{mod}.{name}"

    def is_mutable(self) -> bool:
        return False

    def get_kind(self) -> str:
        return "number"

    def type_to_json(self) -> str:
        # Note that our serialization of numbers is lossy, since for example `numpy.int8(0)` would
        # be serialized as `int`. This is fine for our purposes, since `numpy.int8(0)` and `int(0)`
        # can be used interchangeably as keys in a dictionary.
        if isinstance(self.value, numbers.Integral):
            return "int"
        if isinstance(self.value, numbers.Real):
            return "float"
        if isinstance(self.value, numbers.Complex):
            return "complex"
        raise NotImplementedError(
            f"type_to_json() is not implemented for this type. type: {type(self.value)}"
        )

    def value_to_json(self) -> JsonData:
        if isinstance(self.value, numbers.Integral):
            return int(self.value)
        if isinstance(self.value, numbers.Real):
            return float(self.value)
        if isinstance(self.value, numbers.Complex):
            return str(self.value)
        return super().value_to_json()

    @classmethod
    def value_from_json(cls, type_name: str, data: JsonData) -> NT:
        if type_name == "int":
            if not isinstance(data, numbers.Integral):
                raise ValueError(f"Expected data to be int, got {data}")
            return cast("NT", data)

        if type_name == "float":
            if not isinstance(data, numbers.Real):
                raise ValueError(f"Expected data to be float, got {data}")
            return cast("NT", data)

        if type_name == "complex":
            if not isinstance(data, str):
                raise ValueError(f"Expected data to be str, got {data}")
            return cast("NT", complex(data))

        return super().value_from_json(type_name, data)


class NumpyNumberInspector(NumberInspector["np.number"]):
    CLASS_QNAME = numpy_numeric_scalars

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        # numpy numbers do not print cleanly as of numpy 2.0
        # use the self.value.item() to retrieve the actual number
        return _get_default_display_value(self.value.item(), level=level)


class StringInspector(PositronInspector[str]):
    def is_mutable(self) -> bool:
        return False

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        return _get_string_display_value(self.value, level=level)

    def get_display_type(self) -> str:
        # Don't include the length for strings
        return type(self.value).__name__

    def get_kind(self) -> str:
        return "string"

    def has_children(self) -> bool:
        return False

    def value_to_json(self) -> JsonData:
        return self.value

    @classmethod
    def value_from_json(cls, type_name: str, data: JsonData) -> str:  # noqa: ARG003
        if not isinstance(data, str):
            raise ValueError(f"Expected data to be str, got {data}")

        return data


Timestamp = TypeVar("Timestamp", datetime.datetime, "pd.Timestamp")


class _BaseTimestampInspector(PositronInspector[Timestamp], ABC):
    def is_mutable(self) -> bool:
        return False

    @classmethod
    @abstractmethod
    def value_from_isoformat(cls, string: str) -> Timestamp:
        pass

    def value_to_json(self) -> JsonData:
        return self.value.isoformat()

    @classmethod
    def value_from_json(cls, type_name: str, data: JsonData) -> Timestamp:  # noqa: ARG003
        if not isinstance(data, str):
            raise ValueError(f"Expected data to be str, got {data}")

        return cls.value_from_isoformat(data)


class DatetimeInspector(_BaseTimestampInspector[datetime.datetime]):
    CLASS_QNAME = "datetime.datetime"

    @classmethod
    def value_from_isoformat(cls, string: str) -> datetime.datetime:
        return datetime.datetime.fromisoformat(string)


class PandasTimestampInspector(_BaseTimestampInspector["pd.Timestamp"]):
    CLASS_QNAME = "pandas._libs.tslibs.timestamps.Timestamp"

    @classmethod
    def value_from_isoformat(cls, string: str) -> pd.Timestamp:
        return _pandas().Timestamp.fromisoformat(string)


#
# Collections
#

CollectionT = Union[bytearray, range, FrozenSet, Sequence, AbstractSet, Tuple]
CT = TypeVar("CT", CollectionT, "np.ndarray", "torch.Tensor")


class _BaseCollectionInspector(PositronInspector[CT], ABC):
    def get_kind(self) -> str:
        return "collection"

    def has_children(self) -> bool:
        # For ranges, we don't visualize the children as they're
        # implied as a contiguous set of integers in a range.
        # For sets, we don't visualize the children as they're
        # not subscriptable objects.
        if isinstance(self.value, (range, AbstractSet, FrozenSet)):
            return False

        return super().has_children()

    def has_child(self, key: int) -> bool:
        return key < self.get_length()

    def get_child(self, key: int) -> Any:
        # Don't allow indexing into ranges or sets.
        if isinstance(self.value, (range, AbstractSet, FrozenSet)):
            raise TypeError(f"get_child() is not implemented for type: {type(self.value)}")

        # TODO(pyright): type should be narrowed to exclude frozen set, retry in a future version of pyright
        return self.value[key]  # type: ignore

    def get_children(self) -> Iterable[int]:
        # Treat collection items as children, with the index as the name
        return range(self.get_length())


# We don't use typing.Sequence here since it includes mappings,
# for which we have a separate inspector.


class CollectionInspector(_BaseCollectionInspector[CollectionT]):
    def get_display_type(self) -> str:
        # Display length for various collections and maps
        # using the Python notation for the type
        type_name = type(self.value).__name__
        length = self.get_length()

        if isinstance(self.value, AbstractSet):
            return f"{type_name} {{{length}}}"
        elif isinstance(self.value, tuple):
            return f"{type_name} ({length})"
        else:
            return f"{type_name} [{length}]"

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        if isinstance(self.value, AbstractSet):
            if len(self.value) == 0:
                prefix, suffix = "set(", ")"
            else:
                prefix, suffix = "{", "}"
        elif isinstance(self.value, FrozenSet):
            if len(self.value) == 0:
                prefix, suffix = "frozenset(", ")"
            else:
                prefix, suffix = "{", "}"
        elif isinstance(self.value, bytearray):
            prefix, suffix = "bytearray(b'", "')"
        elif isinstance(self.value, range):
            return _get_default_display_value(self.value, level=level)
        elif isinstance(self.value, tuple):
            prefix, suffix = "(", ")"
        else:
            prefix, suffix = "[", "]"
        return _get_collection_display_value(self.value, prefix=prefix, suffix=suffix, level=level)

    def get_comparison_cost(self) -> int:
        # Placeholder estimate
        return self.get_length()

    def is_mutable(self) -> bool:
        return (
            isinstance(self.value, (MutableSequence, MutableSet))
            # fastcore's L is a mutable list but doesn't pass the isinstance check.
            or safe_isinstance(self.value, "fastcore.foundation", "L")
        )

    def has_viewer(self) -> bool:
        # Lists and tuples open in the Object Explorer.
        return isinstance(self.value, (list, tuple)) and self.has_children()

    def value_to_json(self) -> JsonData:
        if isinstance(self.value, range):
            return {
                "start": self.value.start,
                "stop": self.value.stop,
                "step": self.value.step,
            }

        if isinstance(self.value, tuple):
            return [get_inspector(item).to_json() for item in self.value]

        return super().value_to_json()

    @classmethod
    def value_from_json(cls, type_name: str, data: JsonData) -> CollectionT:
        if type_name == "range":
            if not isinstance(data, dict):
                raise ValueError(f"Expected data to be dict, got {data}")

            if not isinstance(data["start"], int):
                raise ValueError(f"Expected data['start'] to be int, got {data['start']}")

            if not isinstance(data["stop"], int):
                raise ValueError(f"Expected data['stop'] to be int, got {data['stop']}")

            if not isinstance(data["step"], int):
                raise ValueError(f"Expected data['step'] to be int, got {data['step']}")

            # TODO(pyright): cast shouldn't be necessary, recheck in a future version of pyright
            return range(
                cast("int", data["start"]),
                cast("int", data["stop"]),
                cast("int", data["step"]),
            )

        if type_name == "tuple":
            from .access_keys import access_key_from_json

            if not isinstance(data, list):
                raise ValueError(f"Expected data to be list, got {data}")

            return cast("CollectionT", tuple(access_key_from_json(item) for item in data))  # type: ignore

        return super().value_from_json(type_name, data)


Array = TypeVar("Array", "np.ndarray", "torch.Tensor")


class _BaseArrayInspector(_BaseCollectionInspector[Array], ABC):
    def get_kind(self) -> str:
        return "collection" if self.value.ndim > 0 else "number"

    def get_display_type(self) -> str:
        display_type = str(self.value.dtype)

        # Include shape information, only if it's not a scalar
        shape = self.value.shape
        if self.value.ndim == 1:
            # Remove the trailing comma for 1D arrays
            display_type = f"{display_type} ({shape[0]})"
        elif self.value.ndim != 0:
            display_type = f"{display_type} {tuple(shape)}"

        # Prepend the module name if it's not already there, to distinguish different types of
        # arrays e.g. numpy versus pytorch
        module = type(self.value).__module__
        if not display_type.startswith(module):
            display_type = f"{module}.{display_type}"

        return display_type

    def get_comparison_cost(self) -> int:
        # Placeholder estimate
        return self.get_num_cells()

    def get_num_cells(self) -> int:
        """Return the number of value cells in the array."""
        if self.value.ndim == 0:
            return 0

        num_elements = 1
        for dim in self.value.shape:
            num_elements *= dim

        return num_elements

    def get_size(self) -> int:
        dtype = self.value.dtype

        # If it's a non-NumPy dtype, use 8 as a conservative
        # placeholder
        itemsize = getattr(dtype, "itemsize", 8)
        return self.get_num_cells() * itemsize

    def get_length(self) -> int:
        return self.value.shape[0] if self.value.ndim > 0 else 0

    def is_mutable(self) -> bool:
        return True


def _has_array_viewer(value: Any) -> bool:
    # Imported here since data_explorer imports this module
    from .data_explorer import _is_viewable_array

    return _is_viewable_array(value)


class NumpyNdarrayInspector(_BaseArrayInspector["np.ndarray"]):
    CLASS_QNAME = "numpy.ndarray"

    def get_kind(self) -> str:
        # Arrays that open in the Data Explorer are tables, and the frontend labels the viewer of a
        # "table" as "View Data Table". Others, such as structured arrays, stay collections so
        # that %view opens them in the Object Explorer.
        if self.value.ndim in (1, 2) and self.has_viewer():
            return "table"
        return super().get_kind()

    def has_viewer(self) -> bool:
        return _has_array_viewer(self.value)

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        if level >= len(MAX_ITEMS_BY_LEVEL):
            return f"numpy.array({ELLIPSIS})", True
        # Items are formatted as in the xarray values preview, without padding to a common width,
        # and the lines of a 2-D array are joined, since the Variables pane shows one line.
        text = _numpy().array2string(
            self.value,
            max_line_width=ARRAY_MAX_LINE_WIDTH,
            threshold=ARRAY_THRESHOLD,
            edgeitems=ARRAY_EDGEITEMS,
            separator=", ",
            formatter={"all": _preview_item_formatter(self.value.dtype)},
        )
        return " ".join(text.split()), True

    def equals(self, value: np.ndarray) -> bool:
        return _numpy().array_equal(self.value, value)

    def deepcopy(self) -> np.ndarray:
        # TODO: ndarray.copy() is actually a shallow copy which could cause unexpected behavior for
        #       arrays of Python objects. We should raise a copy.Error in that case.
        return self.value.copy()


class TorchTensorInspector(_BaseArrayInspector["torch.Tensor"]):
    CLASS_QNAME = "torch.Tensor"

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        if level >= len(MAX_ITEMS_BY_LEVEL):
            return f"torch.Tensor({ELLIPSIS})", True

        # NOTE:
        # Once https://github.com/pytorch/pytorch/commit/e03800a93af55ef61f2e610d65ac7194c0614edc
        # is in a stable version we can use it to temporarily set print options
        torch = _torch()

        new_options = {
            "threshold": ARRAY_THRESHOLD,
            "edgeitems": ARRAY_EDGEITEMS,
            "linewidth": ARRAY_MAX_LINE_WIDTH,
        }
        options_obj = torch._tensor_str.PRINT_OPTS  # type: ignore[reportGeneralTypeIssues]  # noqa: SLF001
        original_options = {k: getattr(options_obj, k) for k in new_options}

        torch.set_printoptions(**new_options)

        display_value = str(self.value)
        # Strip the surrounding `tensor(...)`
        display_value = display_value[len("tensor(") : -len(")")]

        torch.set_printoptions(**original_options)

        return display_value, True

    def equals(self, value: torch.Tensor) -> bool:
        return _torch().equal(self.value, value)

    def deepcopy(self) -> torch.Tensor:
        # Detach the tensor from any existing computation graphs to
        # avoid gradients propagating through them.

        # TODO: This creates a completely new tensor using new
        #       memory. Is there a more memory-efficient way to do
        #       this?
        return self.value.detach().clone()

    def get_size(self) -> int:
        if self.value.ndim == 0:
            return self.value.element_size()

        num_elements = 1
        for dim in self.value.shape:
            num_elements *= dim

        return num_elements * self.value.element_size()


#
# Maps
#


MT = TypeVar(
    "MT",
    Mapping,
    "pd.DataFrame",
    "pl.DataFrame",
    "pd.Series",
    "pl.Series",
    "pd.Index",
)


class _BaseMapInspector(PositronInspector[MT], ABC):
    def get_kind(self) -> str:
        return "map"

    def get_num_cells(self) -> int:
        result = 1
        for dim in getattr(self.value, "shape", [len(self.value)]):
            result *= dim

        return result

    def get_size(self) -> int:
        return self.get_num_cells()

    def has_child(self, key: Any) -> bool:
        return key in self.get_children()

    def get_child(self, key: Any) -> Any:
        return self.value[key]


class MapInspector(_BaseMapInspector[Mapping]):
    def get_children(self) -> Collection[Any]:
        return self.value.keys()

    def is_mutable(self) -> bool:
        return isinstance(self.value, MutableMapping)

    def has_viewer(self) -> bool:
        # Mappings open in the Object Explorer.
        return self.has_children()

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        prefix, suffix = "{", "}"

        # If the mapping is empty, return the empty representation.
        if not self.value:
            return f"{prefix}{suffix}", False

        # If we're at the max nested level, just return ellipsis e.g. `{...}`.
        if level >= len(MAX_ITEMS_BY_LEVEL):
            return f"{prefix}{ELLIPSIS}{suffix}", True

        # Loop through the items in the mapping and build the display value.
        parts = [prefix]
        item_allowance = MAX_ITEMS_BY_LEVEL[level]
        should_add_separator = False
        truncated = False
        # From Python 3.6, dictionaries retain insertion order, so no need to sort.
        for key in self.value:
            # Add a separator, if needed.
            if should_add_separator:
                parts.append(", ")
            should_add_separator = True

            # If we've reached the item allowance, add ellipsis and break.
            item_allowance -= 1
            if item_allowance <= 0:
                parts.append(ELLIPSIS)
                truncated = True
                break

            # Get the display value for the key.
            key_inspector = get_inspector(key)
            key_repr, key_truncated = key_inspector.get_display_value(level=level + 1)
            truncated = truncated or key_truncated

            parts.append(key_repr)
            parts.append(": ")

            # Get the display value for the item.
            try:
                item = self.value[key]
            except Exception:
                parts.append("<?>")
            else:
                # For items that are the same object as the parent, avoid infinite recursion
                # and represent the item with ellipsis.
                if item is self.value:
                    item_repr, item_truncated = f"{prefix}{ELLIPSIS}{suffix}", True
                else:
                    item_inspector = get_inspector(item)
                    item_repr, item_truncated = item_inspector.get_display_value(level=level + 1)
                truncated = truncated or item_truncated

                parts.append(item_repr)

        parts.append(suffix)
        return "".join(parts), truncated


Column = TypeVar("Column", "pd.Series", "pl.Series", "pd.Index", "ibis.expr.types.generic.Column")


class BaseColumnInspector(_BaseMapInspector[Column], ABC):
    def is_mutable(self) -> bool:
        return True

    def get_child(self, key: Any) -> Any:
        return self.value[key]

    def get_children(self) -> Collection[Any]:
        if hasattr(self.value, "__len__"):
            return range(len(self.value))
        # Fallback for collections that don't implement len, like Ibis columns
        return []

    def get_display_type(self) -> str:
        dtype = getattr(self.value, "dtype", None)
        if dtype is None:
            # Fallback for objects without a dtype attribute
            return f"{type(self.value).__name__} [{self.get_length()}]"
        return f"{dtype} [{self.get_length()}]"

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        qualname = _get_simplified_qualname(self.value)
        return _get_collection_display_value(
            self.value, prefix=f"{qualname} [", suffix="]", level=level
        )

    def get_size(self) -> int:
        dtype = self.value.dtype

        # If it's a non-NumPy dtype, use 8 as a conservative
        # placeholder
        itemsize = getattr(dtype, "itemsize", 8)
        return self.get_num_cells() * itemsize


class PandasSeriesInspector(BaseColumnInspector["pd.Series"]):
    # Simplified names
    CLASS_QNAME = ("pandas.Series",)

    def get_display_name(self, key: int) -> str:
        return str(self.value.index[key])

    def get_child(self, key: int) -> Any:
        return self.value.iloc[key]

    def get_kind(self) -> str:
        # #2215 -- we are temporarily reclassifying Series as a table
        # #so that it shows up in the "data" section
        return "table"

    def equals(self, value: pd.Series) -> bool:
        return self.value.equals(value)

    def deepcopy(self) -> pd.Series:
        # Copies memory because pandas < 3.0 does not have
        # copy-on-write.
        return self.value.copy(deep=True)

    def to_html(self) -> str:
        # TODO: Support HTML
        return self.to_plaintext()

    def to_plaintext(self) -> str:
        return self.value.to_csv(path_or_buf=None, sep="\t")

    def has_viewer(self) -> bool:
        return True


class GeoSeriesInspector(PandasSeriesInspector):
    """Inspector for geopandas.GeoSeries - excludes from change detection."""

    CLASS_QNAME = ("geopandas.GeoSeries",)

    def deepcopy(self):
        """GeoSeries cannot be efficiently copied for change detection."""
        raise copy.Error("GeoSeries contains expensive geometry objects")


class PandasIndexInspector(BaseColumnInspector["pd.Index"]):
    CLASS_QNAME = (
        "pandas.Index",
        "pandas.DatetimeIndex",
        "pandas.RangeIndex",
        "pandas.MultiIndex",
        "pandas.Int64Index",
    )

    def is_mutable(self) -> bool:
        return False

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        # RangeIndexes don't need to be truncated.
        if isinstance(self.value, _pandas().RangeIndex):
            return str(self.value), False
        return super().get_display_value(level=level)

    def has_children(self) -> bool:
        # For ranges, we don't visualize the children as they're
        # implied as a contiguous set of integers in a range.
        if isinstance(self.value, _pandas().RangeIndex):
            return False

        return super().has_children()

    def equals(self, value: pd.Index) -> bool:
        return self.value.equals(value)

    def to_html(self) -> str:
        # TODO: Support HTML
        return self.to_plaintext()

    def to_plaintext(self) -> str:
        return self.value.to_series().to_csv(path_or_buf=None, sep="\t")


class PolarsSeriesInspector(BaseColumnInspector["pl.Series"]):
    # Simplified class names
    CLASS_QNAME = ("polars.Series",)

    def equals(self, value: pl.Series) -> bool:
        try:
            return self.value.equals(value)
        except AttributeError:  # polars.Series.equals was introduced in v0.19.16
            return self.value.series_equal(value)  # type: ignore

    def deepcopy(self) -> pl.Series:
        # Polars produces a shallow clone and does not copy any memory
        # in this operation.
        return self.value.clone()

    def to_html(self) -> str:
        # TODO: Support HTML
        return self.to_plaintext()

    def to_plaintext(self) -> str:
        return self.value.to_frame().write_csv(file=None, separator="\t")


Table = TypeVar("Table", "pd.DataFrame", "pl.DataFrame", "ibis.expr.types.relations.Table")


class BaseTableInspector(_BaseMapInspector[Table], Generic[Table, Column], ABC):  # noqa: PYI059
    """Base inspector for tabular data."""

    def get_display_type(self) -> str:
        type_name = type(self.value).__name__
        shape = self.value.shape
        return f"{type_name} [{shape[0]}x{shape[1]}]"

    def get_kind(self) -> str:
        return "table"

    def get_length(self) -> int:
        # send number of columns.
        # number of rows per column is handled by ColumnInspector
        return self.value.shape[1]

    def has_viewer(self) -> bool:
        return True

    def is_mutable(self) -> bool:
        return True

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        display_value = _get_simplified_qualname(self.value)
        if hasattr(self.value, "shape"):
            shape = self.value.shape
            display_value = f"[{shape[0]} rows x {shape[1]} columns] {display_value}"

        return (_maybe_truncate_string(display_value, level=level)[0], True)


#
# Custom inspectors for specific types
#


class PandasDataFrameInspector(BaseTableInspector["pd.DataFrame", "pd.Series"]):
    # Simplified names
    CLASS_QNAME = ("pandas.DataFrame",)

    def get_display_name(self, key: int) -> str:
        return str(self.value.columns[key])

    def get_children(self):
        return range(self.value.shape[1])

    def get_child(self, key: int) -> Any:
        return self.value.iloc[:, key]

    def equals(self, value: pd.DataFrame) -> bool:
        return self.value.equals(value)

    def deepcopy(self) -> pd.DataFrame:
        # Copies memory because pandas < 3.0 does not have
        # copy-on-write.
        return self.value.copy(deep=True)

    def to_html(self) -> str:
        return self.value.to_html()

    def to_plaintext(self) -> str:
        return self.value.to_csv(path_or_buf=None, sep="\t")


class GeoDataFrameInspector(PandasDataFrameInspector):
    """Inspector for geopandas.GeoDataFrame with optimized change detection."""

    CLASS_QNAME = ("geopandas.GeoDataFrame",)

    def _get_geometry_column_names(self) -> list[str]:
        """Get the names of all geometry columns."""
        geom_cols = []
        with contextlib.suppress(AttributeError, KeyError):
            # First try to get the active geometry column
            geom_cols.append(self.value.geometry.name)

        # Also check for columns with geometry dtype (handles unset active geometry)
        for col in self.value.columns:
            if str(self.value[col].dtype) == "geometry" and col not in geom_cols:
                geom_cols.append(col)

        return geom_cols

    def _get_non_geometry_df(self):
        """Get DataFrame with geometry columns dropped."""
        geom_cols = self._get_geometry_column_names()
        cols_to_drop = [col for col in geom_cols if col in self.value.columns]
        if cols_to_drop:
            return self.value.drop(columns=cols_to_drop)
        return self.value

    def get_comparison_cost(self) -> int:
        """Return cost based on non-geometry columns only."""
        geom_cols = self._get_geometry_column_names()
        num_geom_cols = sum(1 for col in geom_cols if col in self.value.columns)
        num_cols = self.value.shape[1] - num_geom_cols
        return self.value.shape[0] * num_cols

    def deepcopy(self):
        """Copy non-geometry columns deeply, geometry columns shallowly.

        Returns a GeoDataFrame to preserve the inspector type for change detection.
        Geometry columns are shallow-copied since we don't compare them.
        """
        # Shallow copy preserves GeoDataFrame type
        result = self.value.copy(deep=False)

        # Deep copy only non-geometry columns
        geom_cols = set(self._get_geometry_column_names())
        for col in self.value.columns:
            if col not in geom_cols:
                result[col] = self.value[col].copy(deep=True)

        return result

    def equals(self, value) -> bool:
        """Compare only non-geometry columns."""
        try:
            self_non_geom = self._get_non_geometry_df()
            other_non_geom = GeoDataFrameInspector(value)._get_non_geometry_df()  # noqa: SLF001
            return self_non_geom.equals(other_non_geom)
        except Exception:
            return False


class PolarsDataFrameInspector(BaseTableInspector["pl.DataFrame", "pl.Series"]):
    # Simplified class name
    CLASS_QNAME = ("polars.DataFrame",)

    def get_children(self):
        return self.value.columns

    def equals(self, value: pl.DataFrame) -> bool:
        try:
            return self.value.equals(value)
        except AttributeError:  # polars.DataFrame.equals was introduced in v0.19.16
            return self.value.frame_equal(value)  # type: ignore

    def deepcopy(self) -> pl.DataFrame:
        # Polars produces a shallow clone and does not copy any memory
        # in this operation.
        return self.value.clone()

    def to_html(self) -> str:
        return self.value._repr_html_()

    def to_plaintext(self) -> str:
        return self.value.write_csv(file=None, separator="\t")


class IbisDataFrameInspector(BaseTableInspector["ibis.Table", "ibis.Column"]):
    CLASS_QNAME = ("ibis.expr.types.relations.Table", "ibis.Table")

    def get_size(self) -> int:
        # size of the object in memory, not the data it represents
        return sys.getsizeof(self.value)

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        display_value = _get_simplified_qualname(self.value)
        columns = len(self.value.columns)
        display_value = f"[{columns} columns] {display_value}"

        return (_maybe_truncate_string(display_value, level=level)[0], True)

    def get_display_name(self, key: str) -> str:
        return str(key)

    def get_children(self):
        return self.value.columns

    def get_child(self, key: str) -> Any:
        return self.value[key]

    def get_display_type(self) -> str:
        type_name = type(self.value).__name__
        columns = len(self.value.columns)
        return f"{type_name} [{columns} columns]"

    def get_kind(self) -> str:
        return "table"

    def get_length(self) -> int:
        # send number of columns.
        # number of rows per column is handled by ColumnInspector
        return len(self.value.columns)

    def has_viewer(self) -> bool:
        return True

    def is_mutable(self) -> bool:
        return False


class BaseConnectionInspector(ObjectInspector):
    def has_viewer(self) -> bool:
        return self._is_active(self.value)

    def get_kind(self) -> str:
        return "connection"

    def is_mutable(self) -> bool:
        return True

    def deepcopy(self):
        # Connections are mutable but not deepcopiable.
        raise copy.Error("Connections are not copiable")

    def _is_active(self, value) -> bool:
        raise NotImplementedError


class SQLiteConnectionInspector(BaseConnectionInspector):
    # in older Python versions (eg 3.9) the qualname for sqlite3.Connection is just "Connection"
    CLASS_QNAME = ("Connection", "sqlite3.Connection")

    def _is_active(self, value) -> bool:
        try:
            # a connection is active if you can acquire a cursor from it
            value.cursor()
        except Exception:
            return False
        return True


class SQLAlchemyEngineInspector(BaseConnectionInspector):
    CLASS_QNAME = ("sqlalchemy.engine.base.Engine",)

    def _is_active(self, value) -> bool:
        try:
            # a connection is active if you can acquire a connection from it
            value.connect()
        except Exception:
            return False
        return True


class DuckDBConnectionInspector(BaseConnectionInspector):
    CLASS_QNAME = ("duckdb.DuckDBPyConnection",)

    def _is_active(self, value) -> bool:
        try:
            # a connection is active if you can acquire a cursor from it
            value.cursor()
        except Exception:
            return False
        return True


class SnowflakeConnectionInspector(BaseConnectionInspector):
    CLASS_QNAME = ("snowflake.connector.connection.SnowflakeConnection",)

    def _is_active(self, value) -> bool:
        try:
            # a connection is active if you can acquire a cursor from it
            value.cursor()
        except Exception:
            return False
        return True


class DatabricksConnectionInspector(BaseConnectionInspector):
    CLASS_QNAME = ("databricks.sql.client.Connection",)

    def _is_active(self, value) -> bool:
        try:
            # a connection is active if you can acquire a cursor from it
            value.cursor()
        except Exception:
            return False
        return True


class BigQueryConnectionInspector(BaseConnectionInspector):
    CLASS_QNAME = ("google.cloud.bigquery.client.Client",)

    def _is_active(self, value) -> bool:
        # a connection is always active if the client exists
        _ = value
        return True


class RedshiftConnectionInspector(BaseConnectionInspector):
    CLASS_QNAME = ("redshift_connector.Connection", "redshift_connector.core.Connection")

    def _is_active(self, value) -> bool:
        try:
            # a connection is active if you can acquire a cursor from it
            value.cursor()
        except Exception:
            return False
        return True


class SQLServerConnectionInspector(BaseConnectionInspector):
    CLASS_QNAME = ("pyodbc.Connection", "pymssql.Connection", "pymssql._pymssql.Connection")

    def _is_active(self, value) -> bool:
        # pyodbc can match multiple database, we only declare 'active', ie show the viewer
        # icon if it's a SQL Server connection
        if self.get_type_info() == "pyodbc.Connection":
            # check this is a SQL Server connection
            try:
                dbms_name = value.getinfo(_pyodbc().SQL_DBMS_NAME).upper()
                if all(x not in dbms_name for x in ("SQL SERVER", "AZURE SQL")):
                    return False
            except Exception:
                # unable to get the driver name or pyodbc not installed
                return False

        try:
            # a connection is active if you can acquire a cursor from it
            value.cursor()
        except Exception:
            return False

        return True


class IbisExprInspector(PositronInspector["ibis.Expr"]):
    def has_children(self) -> bool:
        return False

    def get_length(self) -> int:
        return 0

    def is_mutable(self):
        return False

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        simplified_name = get_qualname(self.value)
        return _maybe_truncate_string(simplified_name, level=level)[0], True

    def get_display_type(self) -> str:
        return "ibis.Expr"

    def to_html(self) -> str:
        return get_qualname(self.value)

    def to_plaintext(self) -> str:
        return get_qualname(self.value)


#
# xarray
#

_NONE_TEXT = "(none)"

XarrayArray = TypeVar("XarrayArray", "xr.DataArray", "xr.Variable")

_BYTE_UNITS = ("B", "KiB", "MiB", "GiB", "TiB", "PiB")


def _format_nbytes(n: int) -> str:
    """Format a byte count with binary units, using one decimal above bytes."""
    if n < 1024:
        return f"{n} B"
    size = float(n)
    unit = 0
    while size >= 1024 and unit < len(_BYTE_UNITS) - 1:
        size /= 1024
        unit += 1
    return f"{size:.1f} {_BYTE_UNITS[unit]}"


def _format_name_list(names: Iterable[Any], *, level: int = 0) -> tuple[str, bool]:
    """Join names with commas, truncating long lists."""
    return _maybe_truncate_string(", ".join(str(name) for name in names), level=level)


_DASK_ARRAY_QNAMES = (
    "dask.array.core.Array",
    # dask's opt-in array query planning uses a separate Array class
    "dask.array._array_expr._collection.Array",
)

_DASK_METADATA_LABELS = ("chunks", "nbytes", "graph layers")


def _is_dask_array(value: Any) -> bool:
    return "dask.array" in sys.modules and get_qualname(type(value)) in _DASK_ARRAY_QNAMES


def _dask_variables(value: Any) -> list[tuple[tuple[Hashable, ...] | None, DaskArray]]:
    """
    Return `(dims, dask array)` for a dask array or each dask-backed variable of an xarray object.

    Covers a Variable, a DataArray's data, and a Dataset's data variables. Bare dask arrays have
    no dims.
    """
    if "dask.array" not in sys.modules:
        return []
    if _is_dask_array(value):
        return [(None, value)]
    xr_mod = sys.modules.get("xarray")
    if xr_mod is None:
        return []
    if isinstance(value, xr_mod.Variable):
        variables = [value]
    elif isinstance(value, xr_mod.DataArray):
        variables = [value.variable]
    elif isinstance(value, xr_mod.Dataset):
        variables = [value.variables[name] for name in value.data_vars]
    else:
        return []
    # `Variable.chunks` reads the wrapped array's chunks without loading anything, and
    # `Variable.data` returns a chunked array as-is, so neither reads a lazily indexed variable.
    return [
        (var.dims, var.data)
        for var in variables
        if var.chunks is not None and _is_dask_array(var.data)
    ]


def _is_dask_backed(value: Any) -> bool:
    """Whether value is a dask array or an xarray object with dask-backed data."""
    return bool(_dask_variables(value))


def _dask_nbytes(array: DaskArray) -> int | None:
    """The size of a dask array in bytes, or None when its shape is not known until computed."""
    # A dim's length is NaN when it depends on data that has not been computed
    size = math.prod(array.shape)
    return int(size) * array.dtype.itemsize if math.isfinite(size) else None


def _first_chunk(chunks: tuple[int, ...]) -> int:
    # dask's own repr also reports the first chunk of each dim as the chunk size
    return chunks[0] if chunks else 0


def _dask_metadata(value: Any) -> tuple[str, list[tuple[str, str]]]:
    """
    Describe the chunk layout of a dask-backed value without computing it.

    Returns a summary such as "61 chunks, 90.2 MiB (lazy)" and ordered (label, text) rows for
    `chunks`, `nbytes`, and `graph layers`. `nbytes` sums the dask-backed variables only.

    The summary's count depends on the chunk layout:

    - Variables that chunk a shared dim differently: no count ("mixed chunks, ..."), and the
      `chunks` row reads "mixed (varies by variable)".
    - A Dataset whose dask variables all have the same number of chunks: that count
      ("61 chunks per variable, ...").
    - Otherwise: the total over the dask variables ("5 chunks, ..."), so that the count and
      the bytes describe the same data.
    """
    arrays = _dask_variables(value)
    sizes = [_dask_nbytes(array) for _, array in arrays]
    nbytes_text = "unknown" if None in sizes else _format_nbytes(sum(cast("list[int]", sizes)))

    # `numblocks` is each array's chunk grid, read from metadata.
    counts = [math.prod(array.numblocks) for _, array in arrays]
    mixed = False
    if len(arrays) == 1 and arrays[0][0] is None:
        array = arrays[0][1]
        chunks_text = str(tuple(_first_chunk(c) for c in array.chunks))
    else:
        grid: dict[Hashable, tuple[int, ...]] = {}
        for dims, array in arrays:
            for dim, chunks in zip(dims or (), array.chunks):
                if grid.setdefault(dim, chunks) != chunks:
                    mixed = True
        if mixed:
            chunks_text = "mixed (varies by variable)"
        else:
            # Without conflicts, each dim has one chunk size across the variables that use it.
            dims = [dim for dim in value.sizes if dim in grid]
            chunks_text = ", ".join(f"{dim}: {_first_chunk(grid[dim])}" for dim in dims)

    graph = value.__dask_graph__()
    hlg_mod = sys.modules.get("dask.highlevelgraph")
    if hlg_mod is not None and isinstance(graph, hlg_mod.HighLevelGraph):
        layers = len(graph.layers)
    else:
        layers = 1

    xr_mod = sys.modules.get("xarray")
    is_dataset = xr_mod is not None and isinstance(value, xr_mod.Dataset)
    if mixed:
        summary = f"mixed chunks, {nbytes_text} (lazy)"
    elif is_dataset and len(set(counts)) == 1:
        count = counts[0]
        summary = f"{count} chunk{'' if count == 1 else 's'} per variable, {nbytes_text} (lazy)"
    else:
        count = sum(counts)
        summary = f"{count} chunk{'' if count == 1 else 's'}, {nbytes_text} (lazy)"
    rows: list[tuple[str, str]] = list(
        zip(_DASK_METADATA_LABELS, (chunks_text, nbytes_text, str(layers)))
    )
    return summary, rows


def _attrs_equal(a: Mapping[Any, Any], b: Mapping[Any, Any]) -> bool:
    try:
        return bool(a == b)
    except (ValueError, TypeError):
        # Attribute values such as numpy arrays do not support bool(x == y)
        return False


# The key of a DataArray's or Variable's own data in `_xarray_variables`. Any hashable, None included, can name
# an xarray variable, so the key is an object that no user code can pass.
_OWN_DATA_KEY = object()


def _xarray_variables(
    value: xr.Dataset | xr.DataArray | xr.Variable,
) -> dict[Hashable, xr.Variable]:
    """
    Every variable of an xarray object.

    A DataArray's own data, and a Variable itself, are keyed by `_OWN_DATA_KEY`.
    """
    if hasattr(value, "data_vars"):
        return dict(cast("xr.Dataset", value).variables)
    if not hasattr(value, "coords"):
        return {_OWN_DATA_KEY: cast("xr.Variable", value)}
    array = cast("xr.DataArray", value)
    variables: dict[Hashable, xr.Variable] = {
        name: coord.variable for name, coord in array.coords.items()
    }
    variables[_OWN_DATA_KEY] = array.variable
    return variables


def _xarray_lazy_equals(a: Any, b: Any) -> bool:
    """
    Compare xarray objects when either one is dask-backed, without computing anything.

    Compares structure, names, and attrs; each chunked variable by its dask name (the graph
    token, which a copy keeps); and every other variable (data variables and coordinates) by
    value.
    """
    if type(a) is not type(b):
        return False
    # A Variable has no coordinates and no name
    coords_a = list(getattr(a, "coords", ()))
    if dict(a.sizes) != dict(b.sizes) or coords_a != list(getattr(b, "coords", ())):
        return False
    if hasattr(a, "data_vars"):
        if list(a.data_vars) != list(b.data_vars):
            return False
    elif getattr(a, "name", None) != getattr(b, "name", None):
        return False
    if not _attrs_equal(a.attrs, b.attrs):
        return False

    vars_a = _xarray_variables(a)
    vars_b = _xarray_variables(b)
    if list(vars_a) != list(vars_b):
        return False
    for name, var_a in vars_a.items():
        var_b = vars_b[name]
        if var_a.dims != var_b.dims or var_a.dtype != var_b.dtype:
            return False
        if not _attrs_equal(var_a.attrs, var_b.attrs):
            return False
        if var_a.chunks is not None or var_b.chunks is not None:
            if var_a.chunks is None or var_b.chunks is None:
                return False
            if getattr(var_a.data, "name", None) != getattr(var_b.data, "name", None):
                return False
        elif not var_a.identical(var_b):
            return False
    return True


def _xarray_lazy_copy(value: Any) -> Any:
    """Snapshot a dask-backed xarray object for change detection without computing it."""
    out = value.copy(deep=False)
    out.attrs = dict(value.attrs)
    # A shallow copy shares each dask Array object with the original. Dask's in-place assignment
    # gives the array it is called on a new name, so a shared object would pass the new name on
    # to the snapshot too. A shallow copy of each dask Array is a new object with the same name
    # and graph, and computes nothing.
    # A shallow copy also shares the buffers of eager variables, so an in-place edit of the
    # original would show up in the snapshot. Each eager variable other than an index gets a deep
    # copy. Index variables are immutable.
    index_variable = sys.modules["xarray"].IndexVariable
    for name, variable in _xarray_variables(out).items():
        if variable.chunks is not None and _is_dask_array(variable.data):
            variable.data = copy.copy(variable.data)
        elif name is not _OWN_DATA_KEY and not isinstance(variable, index_variable):
            owned = variable.copy(deep=True)
            if name in out.coords:
                out.coords[name] = owned
            else:
                out[name] = owned
    return out


def _eager_variable_nbytes(value: Any) -> int:
    """The bytes of every variable of an xarray object that is not chunked."""
    return sum(
        variable.nbytes for variable in _xarray_variables(value).values() if variable.chunks is None
    )


class MetadataRow:
    """A leaf row of descriptive text shown as a child of an array-like object."""

    def __init__(self, text: str) -> None:
        self.text = text


class ArrayGroup:
    """A group row (for example Coordinates) shown as a child of an array-like object."""

    COORDINATES = "Coordinates"
    DATA_VARIABLES = "Data variables"
    ATTRIBUTES = "Attributes"
    VALUES = "Values"
    DASK_METADATA = "Dask metadata"
    AXES = "Axes"

    def __init__(self, name: str, owner: Any) -> None:
        self.name = name
        self.owner = owner


class MetadataRowInspector(PositronInspector[MetadataRow]):
    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:  # noqa: ARG002
        return self.value.text, False

    def get_display_type(self) -> str:
        return ""

    def get_kind(self) -> str:
        return "other"

    def get_length(self) -> int:
        return 0

    def has_children(self) -> bool:
        return False

    def get_children(self) -> list[Any]:
        return []

    def is_mutable(self) -> bool:
        return False


_PREVIEW_THRESHOLD = 100
_PREVIEW_EDGEITEMS = 3


def _edge_values(array: xr.DataArray | xr.Variable) -> np.ndarray:
    """
    Read only the leading and trailing items of each dim of a DataArray or Variable.

    Each dim longer than twice the edge items keeps one extra leading item, so that
    `np.array2string` still elides the middle and shows exactly the edge items. A lazily indexed
    variable (opened from disk without chunks) then reads a bounded number of elements.
    """
    edge = _PREVIEW_EDGEITEMS
    per_dim = [
        [slice(0, edge + 1), slice(size - edge, size)] if size > 2 * edge else [slice(None)]
        for size in array.shape
    ]
    variable = getattr(array, "variable", array)

    def block(prefix: tuple[slice, ...]) -> Any:
        if len(prefix) == array.ndim:
            return variable[prefix].to_numpy()
        return [block((*prefix, part)) for part in per_dim[len(prefix)]]

    return _numpy().block(block(()))


def _pandas_time_formatter(kind: str) -> Callable[[Any], str] | None:
    """
    A function that shows a datetime64 or timedelta64 item as pandas does.

    `kind` is the dtype kind, "M" or "m". The result reads like "2020-01-01 00:00:00". Returns
    None when pandas has not been imported.
    """
    pd_mod = sys.modules.get("pandas")
    if pd_mod is None:
        return None
    convert = pd_mod.Timestamp if kind == "M" else pd_mod.Timedelta

    def format_item(item: Any) -> str:
        try:
            return str(convert(item))
        except (ValueError, OverflowError):
            # For example, a date outside the range of pandas' nanosecond timestamps.
            return str(item)

    return format_item


def _format_float(value: Any, digits: int) -> str:
    """
    A numpy float with at most `digits` significant digits, laid out like Python's "g" format.

    The numpy scalar is formatted as is: a cast to a Python float would lose the range and
    precision of a longdouble wider than float64. Integral values keep a ".0".
    """
    np = _numpy()
    scientific = np.format_float_scientific(value, precision=digits - 1, unique=True, trim="-")
    mantissa, _, exponent = scientific.partition("e")
    if not exponent:
        # nan or inf
        return mantissa
    if -4 <= int(exponent) < digits:
        text = np.format_float_positional(
            value, precision=digits, unique=True, fractional=False, trim="-"
        )
        return text if any(c in text for c in ".en") else f"{text}.0"
    return f"{mantissa}e{int(exponent):+03d}"


def _preview_item_formatter(dtype: Any) -> Callable[[Any], str]:
    """
    A function that formats one item of an array of `dtype` for the values preview.

    Items are not padded to a common width. A float shows the significant digits its dtype
    resolves (`np.finfo(dtype).precision`): a float32 decoded from a file as 28.019999, one
    float32 step from 28.02, shows as 28.02.
    """
    kind = dtype.kind
    if kind in "mM":
        return _pandas_time_formatter(kind) or str
    if kind == "f":
        digits = _numpy().finfo(dtype).precision
        return lambda item: _format_float(item, digits)
    if kind == "c":
        digits = _numpy().finfo(dtype).precision

        def format_complex(item: Any) -> str:
            np = _numpy()
            # signbit keeps the sign of a negative zero, as numpy shows it
            sign = "-" if np.signbit(item.imag) else "+"
            imag = _format_float(np.abs(item.imag), digits)
            return f"{_format_float(item.real, digits)}{sign}{imag}j"

        return format_complex
    if kind == "U":
        return lambda item: repr(str(item))

    def format_other(item: Any) -> str:
        # Quote strings in object arrays; str() gives numpy scalars their plain form
        return repr(str(item)) if isinstance(item, str) else str(item)

    return format_other


def _array_values_preview(array: xr.DataArray | xr.Variable, *, level: int = 0) -> tuple[str, bool]:
    """Preview an eager DataArray's or Variable's values, reading only the items shown."""
    if array.size <= _PREVIEW_THRESHOLD:
        values = array.to_numpy()
        threshold = _PREVIEW_THRESHOLD
    else:
        values = _edge_values(array)
        # The array is over the threshold, so always summarize the edge values.
        threshold = 0
    # Commas separate the items, as in the pandas Series preview.
    text = _numpy().array2string(
        values,
        threshold=threshold,
        edgeitems=_PREVIEW_EDGEITEMS,
        separator=", ",
        formatter={"all": _preview_item_formatter(values.dtype)},
    )
    return _maybe_truncate_string(" ".join(text.split()), level=level)


def _is_own_coordinate(array: xr.DataArray, name: Hashable) -> bool:
    """
    Whether the coordinate `name` of a DataArray is the DataArray itself.

    The DataArray of a coordinate (for example `ds["time"]`) has itself as a coordinate, and
    xarray gives both the same Variable object. A distinct coordinate can share the DataArray's
    name, or even its data, but never its Variable. Nothing is read or computed.

    A dim without a coordinate is not in `array.coords`. Looking it up anyway raises on some
    xarray versions and builds a default range coordinate of the dim's length on others.
    """
    return (
        name == array.name
        and name in array.coords
        and array.coords[name].variable is array.variable
    )


def _coordinate_names(owner: xr.Dataset | xr.DataArray) -> list[Hashable]:
    """
    The names listed in the Coordinates group of an xarray object.

    A DataArray omits the coordinate that is the DataArray itself. Listing it would nest the same
    row inside itself forever.
    """
    names = list(owner.coords)
    xr_mod = sys.modules.get("xarray")
    if xr_mod is not None and isinstance(owner, xr_mod.DataArray) and owner.name in owner.coords:
        array = cast("xr.DataArray", owner)
        names = [name for name in names if not _is_own_coordinate(array, name)]
    return names


class ArrayGroupInspector(PositronInspector[ArrayGroup]):
    def _names(self) -> list[Any]:
        group = self.value
        if group.name == ArrayGroup.COORDINATES:
            return _coordinate_names(group.owner)
        if group.name == ArrayGroup.DATA_VARIABLES:
            return list(group.owner.data_vars)
        if group.name == ArrayGroup.ATTRIBUTES:
            return list(group.owner.attrs)
        if group.name == ArrayGroup.DASK_METADATA:
            return list(_DASK_METADATA_LABELS)
        if group.name == ArrayGroup.AXES:
            # Axes are keyed by their index in HyperSpy's order, since names can be unset or repeat
            return list(range(len(_hyperspy_axes(group.owner))))
        return []

    def get_display_name(self, key: Any) -> str:
        if self.value.name == ArrayGroup.AXES and type(key) is int:
            return _hyperspy_axis_label(_hyperspy_axes(self.value.owner)[key], key)
        return str(key)

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        if self.value.name == ArrayGroup.VALUES:
            return _array_values_preview(self.value.owner, level=level)
        if self.value.name == ArrayGroup.DASK_METADATA:
            return _dask_metadata(self.value.owner)[0], False
        names = self._names()
        if not names:
            return _NONE_TEXT, False
        if self.value.name == ArrayGroup.AXES:
            names = [self.get_display_name(key) for key in names]
        return _format_name_list(names, level=level)

    def get_display_type(self) -> str:
        return ""

    def get_kind(self) -> str:
        return "other"

    def get_length(self) -> int:
        return len(self._names())

    def has_children(self) -> bool:
        return self.get_length() > 0

    def get_children(self) -> list[Any]:
        return self._names()

    def has_child(self, key: Any) -> bool:
        return key in self._names()

    def get_child(self, key: Any) -> Any:
        group = self.value
        if group.name == ArrayGroup.COORDINATES:
            return group.owner.coords[key]
        if group.name == ArrayGroup.DATA_VARIABLES:
            return group.owner[key]
        if group.name == ArrayGroup.ATTRIBUTES:
            return group.owner.attrs[key]
        if group.name == ArrayGroup.DASK_METADATA:
            return MetadataRow(dict(_dask_metadata(group.owner)[1])[key])
        if group.name == ArrayGroup.AXES and type(key) is int and key in self._names():
            return MetadataRow(_hyperspy_axis_text(_hyperspy_axes(group.owner)[key]))
        raise KeyError(key)

    def is_mutable(self) -> bool:
        return False


# Protocols of fsspec filesystems and URLs whose data is on this machine
_LOCAL_PROTOCOLS = frozenset({"file", "local", "memory"})
# A URL scheme, such as "https://" or "s3://". A Windows path, such as C:\data, has none.
_URL_SCHEME = re.compile(r"^([a-zA-Z][a-zA-Z0-9+.-]+)://")
# Remote stores of zarr's ObjectStore, which wraps an obstore store
_OBSTORE_REMOTE_STORES = frozenset({"AzureStore", "GCSStore", "HTTPStore", "S3Store"})


def _is_remote_url(source: Any) -> bool:
    """
    Whether source is the URL of data on another machine, such as "https://..." or "s3://...".

    A chained fsspec URL, such as "simplecache::s3://...", is remote when its last part is.
    """
    if not isinstance(source, str):
        return False
    match = _URL_SCHEME.match(source.split("::")[-1])
    return match is not None and match.group(1).lower() not in _LOCAL_PROTOCOLS


def _zarr_is_remote(node: Any) -> bool:
    """Whether a zarr array or group is stored on another machine, read without any I/O."""
    store = node.store
    # FsspecStore (zarr 3) and FSStore (zarr 2) name their filesystem's protocols.
    fs = getattr(store, "fs", None)
    if fs is not None:
        protocols = fs.protocol if isinstance(fs.protocol, (list, tuple)) else (fs.protocol,)
        return not _LOCAL_PROTOCOLS.intersection(protocols)
    return type(getattr(store, "store", None)).__name__ in _OBSTORE_REMOTE_STORES


def _xarray_flags(value: Any) -> str:
    """
    The suffix of an xarray object's display value, such as " (dask, remote)", or "".

    "remote" means that xarray opened it from a URL, which it records as the "source" encoding.
    Variables selected from a Dataset do not have that encoding.
    """
    flags = []
    if _is_dask_backed(value):
        flags.append("dask")
    if _is_remote_url(getattr(value, "encoding", {}).get("source")):
        flags.append("remote")
    return f" ({', '.join(flags)})" if flags else ""


class XarrayDatasetInspector(PositronInspector["xr.Dataset"]):
    CLASS_QNAME = ("xarray.core.dataset.Dataset", "xarray.Dataset")

    DIMENSIONS = "Dimensions"
    INDEXES = "Indexes"

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        ds = self.value
        text = f"[{len(ds.sizes)} dims x {len(ds.data_vars)} vars] xarray.Dataset"
        text += _xarray_flags(ds)
        return _maybe_truncate_string(text, level=level)[0], True

    def get_display_type(self) -> str:
        ds = self.value
        return f"Dataset [{len(ds.sizes)} dims x {len(ds.data_vars)} vars]"

    def get_kind(self) -> str:
        return "table"

    def get_length(self) -> int:
        # The Variables pane treats the length as the child count, and adds a "more values"
        # row when it is larger than the number of children it receives.
        return len(self.get_children())

    def get_size(self) -> int:
        return self.value.nbytes

    def has_children(self) -> bool:
        return True

    def get_children(self) -> list[str]:
        children = [
            self.DIMENSIONS,
            ArrayGroup.COORDINATES,
            ArrayGroup.DATA_VARIABLES,
            self.INDEXES,
            ArrayGroup.ATTRIBUTES,
        ]
        if _is_dask_backed(self.value):
            children.append(ArrayGroup.DASK_METADATA)
        return children

    def has_child(self, key: Any) -> bool:
        return key in self.get_children()

    def get_child(self, key: Any) -> Any:
        ds = self.value
        if key == self.DIMENSIONS:
            text = ", ".join(f"{dim}: {size}" for dim, size in ds.sizes.items())
            return MetadataRow(text or _NONE_TEXT)
        if key == self.INDEXES:
            return MetadataRow(_format_name_list(ds.indexes)[0] or _NONE_TEXT)
        if key in self.get_children():
            return ArrayGroup(key, ds)
        raise KeyError(key)

    def get_comparison_cost(self) -> int:
        if _is_dask_backed(self.value):
            return _eager_variable_nbytes(self.value)
        return self.value.nbytes

    def equals(self, value: xr.Dataset) -> bool:
        if _is_dask_backed(self.value) or _is_dask_backed(value):
            return _xarray_lazy_equals(self.value, value)
        return type(value) is type(self.value) and self.value.identical(value)

    def deepcopy(self) -> xr.Dataset:
        if _is_dask_backed(self.value):
            return _xarray_lazy_copy(self.value)
        return self.value.copy(deep=True)

    def to_html(self) -> str:
        return self.value._repr_html_()

    def to_plaintext(self) -> str:
        return repr(self.value)


class _XarrayArrayInspector(PositronInspector[XarrayArray], ABC):
    """Shared behavior of the DataArray and Variable inspectors."""

    def _type_name(self) -> str:
        return type(self.value).__name__

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        array = self.value
        if array.ndim == 0:
            shape = "scalar"
        else:
            shape = " x ".join(f"{dim}: {size}" for dim, size in zip(array.dims, array.shape))
        text = f"[{shape}] xarray.{self._type_name()}"
        text += _xarray_flags(array)
        return _maybe_truncate_string(text, level=level)[0], True

    def get_display_type(self) -> str:
        array = self.value
        shape = "x".join(str(size) for size in array.shape) if array.ndim > 0 else "scalar"
        return f"{self._type_name()} [{shape}] {array.dtype}"

    def get_kind(self) -> str:
        return "table"

    def has_viewer(self) -> bool:
        return _has_array_viewer(self.value)

    def get_length(self) -> int:
        # The Variables pane treats the length as the child count, and adds a "more values"
        # row when it is larger than the number of children it receives.
        return len(self.get_children())

    def get_size(self) -> int:
        return self.value.nbytes

    def has_children(self) -> bool:
        return self.get_length() > 0

    @abstractmethod
    def get_children(self) -> list[str]:
        pass

    def _is_bare_scalar(self) -> bool:
        """Whether the array is 0-D, eager, and without attrs: its display says everything."""
        array = self.value
        return array.ndim == 0 and len(array.attrs) == 0 and not _is_dask_backed(array)

    def _data_children(self) -> list[str]:
        """The Attributes, Values, and Dask metadata rows."""
        children = [ArrayGroup.ATTRIBUTES]
        # A chunked (dask) array has no values preview, since building one would compute.
        if self.value.chunks is None:
            children.append(ArrayGroup.VALUES)
        if _is_dask_backed(self.value):
            children.append(ArrayGroup.DASK_METADATA)
        return children

    def has_child(self, key: Any) -> bool:
        return key in self.get_children()

    def get_child(self, key: Any) -> Any:
        if key in self.get_children():
            return ArrayGroup(key, self.value)
        raise KeyError(key)

    def get_comparison_cost(self) -> int:
        if _is_dask_backed(self.value):
            return _eager_variable_nbytes(self.value)
        return self.value.nbytes

    def equals(self, value: XarrayArray) -> bool:
        if _is_dask_backed(self.value) or _is_dask_backed(value):
            return _xarray_lazy_equals(self.value, value)
        return type(value) is type(self.value) and self.value.identical(value)

    def deepcopy(self) -> XarrayArray:
        if _is_dask_backed(self.value):
            return _xarray_lazy_copy(self.value)
        return self.value.copy(deep=True)

    def to_plaintext(self) -> str:
        return repr(self.value)


class XarrayDataArrayInspector(_XarrayArrayInspector["xr.DataArray"]):
    CLASS_QNAME = ("xarray.core.dataarray.DataArray", "xarray.DataArray")

    def _type_name(self) -> str:
        return "DataArray"

    def get_children(self) -> list[str]:
        da = self.value
        if len(da.coords) == 0 and self._is_bare_scalar():
            return []
        children: list[str] = []
        # A coordinate's own DataArray hides the group when it is the only coordinate.
        # Without coordinates at all, the group stays and shows "(none)".
        if len(da.coords) == 0 or _coordinate_names(da):
            children.append(ArrayGroup.COORDINATES)
        return children + self._data_children()

    def to_html(self) -> str:
        return self.value._repr_html_()


class XarrayVariableInspector(_XarrayArrayInspector["xr.Variable"]):
    """
    An xarray Variable, such as an item of `Dataset.variables`.

    A Variable has dims, data, and attrs, without coordinates or a name. The display names the
    class, Variable or IndexVariable.
    """

    CLASS_QNAME = (
        "xarray.core.variable.Variable",
        "xarray.Variable",
        "xarray.core.variable.IndexVariable",
        "xarray.IndexVariable",
    )

    def get_children(self) -> list[str]:
        if self._is_bare_scalar():
            return []
        return self._data_children()


class DaskArrayInspector(PositronInspector["DaskArray"]):
    CLASS_QNAME = _DASK_ARRAY_QNAMES

    def get_display_type(self) -> str:
        array = self.value
        shape = "x".join(str(size) for size in array.shape) if array.ndim > 0 else "scalar"
        return f"{type(array).__name__} [{shape}] {array.dtype}"

    def get_kind(self) -> str:
        return "table"

    def has_viewer(self) -> bool:
        return _has_array_viewer(self.value)

    def get_length(self) -> int:
        # The Variables pane treats the length as the child count, and adds a "more values"
        # row when it is larger than the number of children it receives.
        return len(self.get_children())

    def get_size(self) -> int:
        return _dask_nbytes(self.value) or 0

    def has_children(self) -> bool:
        return True

    def get_children(self) -> list[str]:
        return [ArrayGroup.DASK_METADATA]

    def has_child(self, key: Any) -> bool:
        return key == ArrayGroup.DASK_METADATA

    def get_child(self, key: Any) -> Any:
        if key == ArrayGroup.DASK_METADATA:
            return ArrayGroup(key, self.value)
        raise KeyError(key)

    def get_comparison_cost(self) -> int:
        return 0

    def equals(self, value: DaskArray) -> bool:
        # The name is a token of the array's task graph; changing the array changes it.
        return type(value) is type(self.value) and value.name == self.value.name

    def deepcopy(self) -> DaskArray:
        # A shallow copy shares the task graph and keeps the name, without computing anything.
        # In-place assignment then gives the original a new name.
        return copy.copy(self.value)


#
# zarr and h5py
#

# The key of a group's Attributes row. zarr and HDF5 name group members with strings, so an int
# key never collides with a member.
_GROUP_ATTRIBUTES_KEY = 0

_ZARR_ARRAY_QNAMES = ("zarr.core.array.Array", "zarr.core.Array")  # zarr 3, zarr 2
_ZARR_GROUP_QNAMES = ("zarr.core.group.Group", "zarr.hierarchy.Group")  # zarr 3, zarr 2
_H5PY_DATASET_QNAMES = ("h5py._hl.dataset.Dataset",)
_H5PY_GROUP_QNAMES = ("h5py._hl.files.File", "h5py._hl.group.Group")


def _shape_text(shape: tuple[int, ...], dims: Sequence[Any] | None) -> str:
    """A shape such as "time: 366 x lat: 140 x lon: 220", or "366 x 140 x 220" without dim names."""
    if len(shape) == 0:
        return "scalar"
    if dims is not None and len(dims) == len(shape):
        return " x ".join(f"{dim}: {size}" for dim, size in zip(dims, shape))
    return " x ".join(str(size) for size in shape)


def _count(n: int, noun: str) -> str:
    """A count with its noun, such as "1 chunk" or "61 chunks"."""
    return f"{n} {noun}" if n == 1 else f"{n} {noun}s"


def _zarr_dims(array: Any) -> list[str] | None:
    """The dim names of a zarr array, or None when the store does not name every dim."""
    # zarr format 3 names dims in the array metadata. xarray names them in an attribute when it
    # writes zarr format 2.
    names = getattr(getattr(array, "metadata", None), "dimension_names", None)
    if names is None:
        names = array.attrs.get("_ARRAY_DIMENSIONS")
    return _valid_dims(names, array.ndim)


def _valid_dims(names: Any, ndim: int) -> list[str] | None:
    """
    Check dim names read from metadata, which any writer can set, such as a zarr attribute.

    Returns the names as a list when there is one distinct, non-empty string per dim of an array
    with ndim dims, else None.
    """
    if not isinstance(names, (list, tuple)) or len(names) != ndim:
        return None
    if not all(isinstance(name, str) and name for name in names):
        return None
    if len(set(names)) != len(names):
        return None
    return list(names)


def _h5py_dims(dataset: Any) -> list[str] | None:
    """The dim names of an h5py dataset, or None when the file does not name every dim."""
    # An anonymous dataset, not linked into the file, has no name.
    if dataset.name and dataset.ndim == 1 and sys.modules["h5py"].h5ds.is_scale(dataset.id):
        # A 1-D dimension scale, such as a netCDF4 coordinate, names the dim it measures.
        return [dataset.name.rsplit("/", 1)[-1]]
    names = []
    for dim in dataset.dims:
        # A dim is named by its label, or else by its dimension scale, as in netCDF4 files.
        scales = dim.values()
        name = dim.label or (scales[0].name.rsplit("/", 1)[-1] if scales else None)
        names.append(name)
    return _valid_dims(names, dataset.ndim)


def _is_stored_array(value: Any) -> bool:
    """Whether value is a zarr array or an h5py dataset of an open file."""
    qualname = get_qualname(type(value))
    if qualname in _ZARR_ARRAY_QNAMES:
        return True
    # A dataset of a closed file is falsy.
    return qualname in _H5PY_DATASET_QNAMES and bool(value)


def _stored_array_dims(value: Any) -> list[str] | None:
    """The dim names of a zarr array or an h5py dataset, or None when not every dim is named."""
    if get_qualname(type(value)) in _ZARR_ARRAY_QNAMES:
        return _zarr_dims(value)
    return _h5py_dims(value)


def _stored_array_name(value: Any) -> str | None:
    """The last part of the path of a zarr array or an h5py dataset, or None when it has none."""
    # An anonymous h5py dataset has no name, and an array at the root of a store has no last part.
    basename = (value.name or "").rsplit("/", 1)[-1]
    return basename or None


class _StoredArrayInspector(PositronInspector[Any], ABC):
    """
    Shared behavior of an array in a zarr store or an HDF5 file.

    Every row reads metadata only. The values stay in the store, which can be large, compressed,
    or remote.
    """

    LIBRARY: str
    CHUNKS = "Chunks"
    COMPRESSION = "Compression"

    def _is_open(self) -> bool:
        return True

    def _is_remote(self) -> bool:
        return False

    @abstractmethod
    def _dims(self) -> list[str] | None:
        """The dim names, or None when the store does not name every dim."""

    @abstractmethod
    def _metadata(self) -> dict[str, str]:
        """The text of each metadata row, keyed by the row's name, in display order."""

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        array = self.value
        if not self._is_open():
            return repr(array), False
        text = f"[{_shape_text(array.shape, self._dims())}] {self.LIBRARY}.{type(array).__name__}"
        if self._is_remote():
            text += " (remote)"
        return _maybe_truncate_string(text, level=level)[0], True

    def get_display_type(self) -> str:
        array = self.value
        if not self._is_open():
            return type(array).__name__
        shape = "x".join(str(size) for size in array.shape) if array.shape else "scalar"
        return f"{type(array).__name__} [{shape}] {array.dtype}"

    def get_kind(self) -> str:
        return "table"

    def has_viewer(self) -> bool:
        return _has_array_viewer(self.value)

    def get_length(self) -> int:
        # The Variables pane treats the length as the child count, and adds a "more values"
        # row when it is larger than the number of children it receives.
        return len(self.get_children())

    def get_size(self) -> int:
        # The size of the values once read, which is neither the bytes in memory nor in the store.
        return self.value.nbytes if self._is_open() else 0

    def has_children(self) -> bool:
        return self.get_length() > 0

    def get_children(self) -> list[str]:
        if not self._is_open():
            return []
        return [ArrayGroup.ATTRIBUTES, *self._metadata()]

    def has_child(self, key: Any) -> bool:
        return key in self.get_children()

    def get_child(self, key: Any) -> Any:
        if key == ArrayGroup.ATTRIBUTES:
            return ArrayGroup(key, self.value)
        metadata = self._metadata()
        if key in metadata:
            return MetadataRow(metadata[key])
        raise KeyError(key)

    def to_plaintext(self) -> str:
        return repr(self.value)


class _StoredGroupInspector(PositronInspector[Any], ABC):
    """Shared behavior of a group in a zarr store or an HDF5 file, listing its members."""

    LIBRARY: str

    def __init__(self, value: Any) -> None:
        super().__init__(value)
        self._member_names: list[str] | None = None

    def _is_open(self) -> bool:
        return True

    def _is_remote(self) -> bool:
        return False

    def _members(self) -> list[str]:
        # Listing a remote zarr store without consolidated metadata takes a request, so the
        # inspector lists the members once, for the row and all its children.
        if self._member_names is None:
            self._member_names = list(self.value.keys())
        return self._member_names

    def _member(self, name: str) -> Any:
        """The member called name, read without listing the group. Raises KeyError if missing."""
        return self.value[name]

    def get_display_name(self, key: Any) -> str:
        if type(key) is int and key == _GROUP_ATTRIBUTES_KEY:
            return ArrayGroup.ATTRIBUTES
        return str(key)

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        group = self.value
        if not self._is_open():
            return repr(group), False
        text = f"[{_count(len(self._members()), 'member')}] {self.LIBRARY}.{type(group).__name__}"
        if self._is_remote():
            text += " (remote)"
        return _maybe_truncate_string(text, level=level)[0], True

    def get_display_type(self) -> str:
        if not self._is_open():
            return type(self.value).__name__
        return f"{type(self.value).__name__} [{_count(len(self._members()), 'member')}]"

    def get_kind(self) -> str:
        return "table"

    def get_length(self) -> int:
        # The Variables pane treats the length as the child count, and adds a "more values"
        # row when it is larger than the number of children it receives.
        return len(self.get_children())

    def has_children(self) -> bool:
        return self.get_length() > 0

    def get_children(self) -> list[Any]:
        if not self._is_open():
            return []
        return [_GROUP_ATTRIBUTES_KEY, *self._members()]

    def has_child(self, key: Any) -> bool:
        return key in self.get_children()

    def get_child(self, key: Any) -> Any:
        if type(key) is int and key == _GROUP_ATTRIBUTES_KEY:
            return ArrayGroup(ArrayGroup.ATTRIBUTES, self.value)
        if isinstance(key, str):
            return self._member(key)
        raise KeyError(key)

    def to_plaintext(self) -> str:
        return repr(self.value)


def _zarr_location(node: Any) -> str:
    """Where a zarr array or group is stored, such as a URL or a directory, without any I/O."""
    store = node.store
    # FsspecStore (remote, zarr 3) and DirectoryStore (zarr 2) name their location `path`,
    # LocalStore (zarr 3) names it `root`.
    root = getattr(store, "path", None) or getattr(store, "root", None) or store
    root = str(root).rstrip("/")
    return f"{root}/{node.path}" if node.path else root


class ZarrArrayInspector(_StoredArrayInspector):
    CLASS_QNAME = _ZARR_ARRAY_QNAMES
    LIBRARY = "zarr"
    STORE = "Store"

    def _is_remote(self) -> bool:
        return _zarr_is_remote(self.value)

    def _dims(self) -> list[str] | None:
        return _zarr_dims(self.value)

    def _metadata(self) -> dict[str, str]:
        array = self.value
        chunks = f"{array.chunks}, {_count(array.nchunks, 'chunk')}"
        shards = getattr(array, "shards", None)  # zarr 3
        if shards is not None:
            chunks += f", in shards of {shards}"
        compressors = getattr(array, "compressors", None)  # zarr 3
        if compressors is None:
            compressors = () if array.compressor is None else (array.compressor,)
        codecs = [*(array.filters or ()), *compressors]
        return {
            self.CHUNKS: chunks,
            self.COMPRESSION: ", ".join(repr(codec) for codec in codecs) or "none",
            self.STORE: _zarr_location(array),
        }

    # Compared by identity. Comparing contents after every execution would read the store, which
    # can be remote. A row refreshes when its variable is reassigned.
    def is_mutable(self) -> bool:
        return False


class ZarrGroupInspector(_StoredGroupInspector):
    CLASS_QNAME = _ZARR_GROUP_QNAMES
    LIBRARY = "zarr"

    def _is_remote(self) -> bool:
        return _zarr_is_remote(self.value)

    # Compared by identity, like ZarrArrayInspector. Listing the members of a remote store without
    # consolidated metadata takes a request per listing.
    def is_mutable(self) -> bool:
        return False


class H5pyDatasetInspector(_StoredArrayInspector):
    """
    An HDF5 dataset.

    h5py objects are mutable and cannot be copied, so the Variables pane summarizes them again
    after every execution. A file closed by a `with` block then shows as closed. HDF5 files are
    local, so that reads only metadata from disk.
    """

    CLASS_QNAME = _H5PY_DATASET_QNAMES
    LIBRARY = "h5py"
    FILE = "File"

    def _is_open(self) -> bool:
        # A dataset of a closed file is falsy, and reading its metadata raises.
        return bool(self.value)

    def _dims(self) -> list[str] | None:
        return _h5py_dims(self.value)

    def _metadata(self) -> dict[str, str]:
        dataset = self.value
        if dataset.chunks is None:
            chunks = "none (contiguous)"
        else:
            count = math.prod(
                math.ceil(size / chunk) for size, chunk in zip(dataset.shape, dataset.chunks)
            )
            chunks = f"{dataset.chunks}, {_count(count, 'chunk')}"
        filters = []
        if dataset.compression is not None:
            options = dataset.compression_opts
            filters.append(dataset.compression + ("" if options is None else f" ({options})"))
        if dataset.shuffle:
            filters.append("shuffle")
        if dataset.scaleoffset is not None:
            filters.append(f"scale-offset ({dataset.scaleoffset})")
        if dataset.fletcher32:
            filters.append("fletcher32 checksum")
        return {
            self.CHUNKS: chunks,
            self.COMPRESSION: ", ".join(filters) or "none",
            self.FILE: dataset.file.filename,
        }


class H5pyGroupInspector(_StoredGroupInspector):
    """An HDF5 file or group. See H5pyDatasetInspector for how changes are detected."""

    CLASS_QNAME = _H5PY_GROUP_QNAMES
    LIBRARY = "h5py"

    def _is_open(self) -> bool:
        # A closed file and its groups are falsy, and listing their members raises.
        return bool(self.value)

    def _member(self, name: str) -> Any:
        group = self.value
        link = group.get(name, getlink=True)
        if link is None:
            raise KeyError(name)
        if isinstance(link, sys.modules["h5py"].ExternalLink):
            # Following an external link would open another file.
            return MetadataRow(f"external link to {link.filename}:{link.path}")
        # Soft links resolve within the same file. A soft link to a missing path resolves to None.
        member = group.get(name)
        if member is None:
            return MetadataRow(f"broken link to {link.path}")
        return member


#
# HyperSpy
#

_HYPERSPY_TREE_QNAMES = (
    "hyperspy.misc._utils.DictionaryTreeBrowser",  # hyperspy 2
    "hyperspy.misc.utils.DictionaryTreeBrowser",  # hyperspy 1
)


def _is_hyperspy_signal(value: Any) -> bool:
    """
    Whether value is a HyperSpy signal.

    That includes every signal class, lazy or not, and those of extensions such as exspy.
    """
    return safe_isinstance(value, "hyperspy.signal", "BaseSignal")


def _hyperspy_text(value: Any) -> str | None:
    """A name or units set on a HyperSpy axis, or None when unset."""
    # traits marks an unset value with its Undefined object, which is not a string.
    return value if isinstance(value, str) and value else None


def _hyperspy_title(signal: Any) -> str | None:
    """The title of a signal, or None when it has none."""
    title = signal.metadata.get_item("General.title")
    return title if isinstance(title, str) and title else None


def _hyperspy_axes(signal: Any) -> list[Any]:
    """The axes of a signal in HyperSpy's order: navigation axes, then signal axes."""
    axes_manager = signal.axes_manager
    return [*axes_manager.navigation_axes, *axes_manager.signal_axes]


def _hyperspy_axis_label(axis: Any, index: int) -> str:
    """The name of an axis, or "axis 0" for the unnamed first axis in HyperSpy's order."""
    return _hyperspy_text(axis.name) or f"axis {index}"


def _hyperspy_array_axes(signal: Any) -> list[tuple[str, Any]]:
    """
    The label and axis of each dim of a signal's data, in the data's order.

    The data lists each of the navigation and signal axes in reverse, so that the first signal
    axis is the last dim of the data.
    """
    labeled = [
        (_hyperspy_axis_label(axis, i), axis) for i, axis in enumerate(_hyperspy_axes(signal))
    ]
    navigation_dimension = signal.axes_manager.navigation_dimension
    return [*labeled[:navigation_dimension][::-1], *labeled[navigation_dimension:][::-1]]


def _hyperspy_axis_value(value: Any) -> str:
    """A calibrated axis value with 6 significant digits, such as "100.5"."""
    return _format_float(_numpy().float64(value), 6)


def _hyperspy_axis_labels(values: Any) -> list[str]:
    """
    The calibrated values of an axis as labels, such as "100.5".

    Each label has the fewest significant digits, from 6 to 12, that keep distinct values
    distinct. An axis computes its values as offset + scale * index, and 12 digits still hide the
    rounding error of that sum.
    """
    np = _numpy()
    values = np.asarray(values, dtype=np.float64)
    distinct = len({_format_float(value, 12) for value in values})
    for digits in range(6, 12):
        labels = [_format_float(value, digits) for value in values]
        if len(set(labels)) == distinct:
            return labels
    return [_format_float(value, 12) for value in values]


def _hyperspy_axis_text(axis: Any) -> str:
    """The description of an axis, such as "signal, 2048 points, 100.0 to 1123.5 eV, step 0.5"."""
    parts = ["navigation" if axis.navigate else "signal", _count(axis.size, "point")]
    if axis.size > 0:
        values = axis.axis
        # The two ends are formatted together, so that close but distinct ends stay distinct.
        first, last = _hyperspy_axis_labels([values[0], values[-1]])
        extent = f"{first} to {last}"
        units = _hyperspy_text(axis.units)
        parts.append(extent if units is None else f"{extent} {units}")
    # Axes of HyperSpy versions before 1.7 are all uniform and have no is_uniform.
    if getattr(axis, "is_uniform", True):
        parts.append(f"step {_hyperspy_axis_value(axis.scale)}")
    else:
        parts.append("non-uniform")
    return ", ".join(parts)


def _hyperspy_shape_text(signal: Any) -> str:
    """The axes of a signal as in HyperSpy's repr, such as "x: 64 x y: 64 | E: 2048"."""

    def part(axes: list[tuple[int, Any]]) -> str:
        return " x ".join(
            f"{name}: {axis.size}" if (name := _hyperspy_text(axis.name)) else str(axis.size)
            for _, axis in axes
        )

    axes = list(enumerate(_hyperspy_axes(signal)))
    navigation_dimension = signal.axes_manager.navigation_dimension
    navigation = part(axes[:navigation_dimension])
    sig = part(axes[navigation_dimension:])
    return f"{navigation} | {sig}".strip()


class HyperspySignalInspector(PositronInspector[Any]):
    """
    A HyperSpy signal of any signal class.

    Signals are mutable, and copying one would copy its data, so the Variables pane summarizes
    them again after every execution. That reads only the axes and metadata; a lazy signal is not
    computed.
    """

    TITLE = "Title"
    AXES = "Axes"
    METADATA = "Metadata"
    ORIGINAL_METADATA = "Original metadata"
    DATA = "Data"

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        signal = self.value
        cls = type(signal)
        library = cls.__module__.split(".", 1)[0]
        text = f"[{_hyperspy_shape_text(signal)}] {library}.{cls.__name__}"
        return _maybe_truncate_string(text, level=level)[0], True

    def get_display_type(self) -> str:
        # HyperSpy's notation for the navigation and signal shapes, such as "(64, 64|2048)"
        axes_manager = self.value.axes_manager
        navigation = ", ".join(str(size) for size in axes_manager.navigation_shape)
        sig = ", ".join(str(size) for size in axes_manager.signal_shape)
        return f"{type(self.value).__name__} ({navigation}|{sig}) {self.value.data.dtype}"

    def get_kind(self) -> str:
        return "table"

    def has_viewer(self) -> bool:
        return _has_array_viewer(self.value)

    def get_length(self) -> int:
        # The Variables pane treats the length as the child count, and adds a "more values"
        # row when it is larger than the number of children it receives.
        return len(self.get_children())

    def get_size(self) -> int:
        data = self.value.data
        if _is_dask_array(data):
            return _dask_nbytes(data) or 0
        return int(getattr(data, "nbytes", 0))

    def has_children(self) -> bool:
        return True

    def get_children(self) -> list[str]:
        children = [self.AXES, self.METADATA]
        if _hyperspy_title(self.value) is not None:
            children.insert(0, self.TITLE)
        if self.value.original_metadata.keys():
            children.append(self.ORIGINAL_METADATA)
        children.append(self.DATA)
        return children

    def has_child(self, key: Any) -> bool:
        return key in self.get_children()

    def get_child(self, key: Any) -> Any:
        signal = self.value
        if key == self.TITLE:
            return MetadataRow(_hyperspy_title(signal) or "")
        if key == self.AXES:
            return ArrayGroup(ArrayGroup.AXES, signal)
        if key == self.METADATA:
            return signal.metadata
        if key == self.ORIGINAL_METADATA:
            return signal.original_metadata
        if key == self.DATA:
            return signal.data
        raise KeyError(key)

    def to_plaintext(self) -> str:
        return repr(self.value)


class HyperspyTreeInspector(PositronInspector[Any]):
    """
    The metadata tree of a HyperSpy signal, such as `s.metadata`.

    Its keys omit the internal `_HyperSpy` node, as HyperSpy's own listing does.
    """

    CLASS_QNAME = _HYPERSPY_TREE_QNAMES

    def _keys(self) -> list[str]:
        return list(self.value.keys())

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        keys = self._keys()
        if not keys:
            return _NONE_TEXT, False
        return _format_name_list(keys, level=level)

    def get_display_type(self) -> str:
        return f"{type(self.value).__name__} [{len(self._keys())}]"

    def get_kind(self) -> str:
        return "map"

    def get_length(self) -> int:
        return len(self._keys())

    def has_children(self) -> bool:
        return self.get_length() > 0

    def get_children(self) -> list[str]:
        return self._keys()

    def has_child(self, key: Any) -> bool:
        return key in self._keys()

    def get_child(self, key: Any) -> Any:
        if key not in self._keys():
            raise KeyError(key)
        return self.value[key]

    def to_plaintext(self) -> str:
        return repr(self.value)


#
# astropy
#

_FITS_HEADER_QNAMES = ("astropy.io.fits.header.Header",)
_FITS_HDULIST_QNAMES = ("astropy.io.fits.hdu.hdulist.HDUList",)

# The numpy dtype of each FITS BITPIX value
_FITS_BITPIX_DTYPES = types.MappingProxyType(
    {8: "uint8", 16: "int16", 32: "int32", 64: "int64", -32: "float32", -64: "float64"}
)


def _is_quantity(value: Any) -> bool:
    """Whether value is an astropy Quantity, of any Quantity class, such as Angle or Distance."""
    return safe_isinstance(value, "astropy.units", "Quantity")


def _is_nddata(value: Any) -> bool:
    """Whether value is an astropy NDData, of any class, such as CCDData."""
    return safe_isinstance(value, "astropy.nddata", "NDData")


def _is_fits_hdu(value: Any) -> bool:
    """Whether value is a FITS HDU of any kind, image or table."""
    return safe_isinstance(value, "astropy.io.fits.hdu.base", "_BaseHDU")


def _is_fits_image_hdu(value: Any) -> bool:
    """Whether value is a FITS image HDU, such as a PrimaryHDU or an ImageHDU."""
    return safe_isinstance(value, "astropy.io.fits.hdu.image", "_ImageBaseHDU")


def _unit_text(unit: Any) -> str | None:
    """A unit as astropy writes it, such as "km / s", or None for no unit or a dimensionless one."""
    if unit is None:
        return None
    return unit.to_string() or None


def _library_name(value: Any) -> str:
    """The class of value prefixed with its top-level package, such as "astropy.CCDData"."""
    cls = type(value)
    return f"{cls.__module__.split('.', 1)[0]}.{cls.__name__}"


class AstropyQuantityInspector(NumpyNdarrayInspector):
    """An astropy Quantity, of any Quantity class: a numpy array with a unit."""

    def _values(self) -> np.ndarray:
        # The plain array. Quantity.value is a numpy scalar for a scalar Quantity.
        return self.value.view(_numpy().ndarray)

    def _unit_text(self) -> str | None:
        return _unit_text(cast("Any", self.value).unit)

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        text, is_truncated = NumpyNdarrayInspector(self._values()).get_display_value(level=level)
        unit = self._unit_text()
        return (text if unit is None else f"{text} {unit}"), is_truncated

    def get_display_type(self) -> str:
        values = self._values()
        text = f"{type(self.value).__name__} {values.dtype}"
        if values.ndim == 1:
            text += f" ({values.shape[0]})"
        elif values.ndim > 1:
            text += f" {values.shape}"
        unit = self._unit_text()
        return text if unit is None else f"{text} [{unit}]"

    def equals(self, value: Any) -> bool:
        # Compared as Any, since type checkers know a Quantity only as the ndarray it subclasses
        old: Any = self.value
        return (
            type(value) is type(old)
            and value.unit == old.unit
            and _numpy().array_equal(self._values(), value.view(_numpy().ndarray))
        )


def _wcs_text(wcs: Any) -> str:
    """
    The world axes of a WCS, such as "WCS: pos.eq.ra (deg), pos.eq.dec (deg)".

    Read through the shared astropy WCS interface, so it also describes a gwcs WCS. An object
    without that interface is described by its class.
    """
    name = type(wcs).__name__
    types = getattr(wcs, "world_axis_physical_types", None)
    units = getattr(wcs, "world_axis_units", None)
    if types is None or units is None:
        return name
    axes = [
        (kind or f"world axis {i}") + (f" ({unit})" if unit else "")
        for i, (kind, unit) in enumerate(zip(types, units))
    ]
    return f"{name}: {', '.join(axes)}"


class AstropyNDDataInspector(PositronInspector[Any]):
    """
    An astropy NDData of any class, such as CCDData.

    It is an array with a unit, an uncertainty, a mask, a WCS, and metadata. NDData objects are mutable, so the Variables pane summarizes them again after every execution.
    That reads no values.
    """

    UNIT = "Unit"
    UNCERTAINTY = "Uncertainty"
    MASK = "Mask"
    WCS = "WCS"
    META = "Meta"
    DATA = "Data"

    def get_display_name(self, key: Any) -> str:
        uncertainty = self.value.uncertainty
        if key == self.UNCERTAINTY and uncertainty is not None:
            return f"{self.UNCERTAINTY} ({type(uncertainty).__name__})"
        return str(key)

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        nddata = self.value
        text = f"[{_shape_text(nddata.data.shape, None)}] {_library_name(nddata)}"
        unit = _unit_text(nddata.unit)
        if unit is not None:
            text += f" {unit}"
        return _maybe_truncate_string(text, level=level)[0], True

    def get_display_type(self) -> str:
        data = self.value.data
        shape = "x".join(str(size) for size in data.shape) if data.shape else "scalar"
        return f"{type(self.value).__name__} [{shape}] {data.dtype}"

    def get_kind(self) -> str:
        return "table"

    def has_viewer(self) -> bool:
        return _has_array_viewer(self.value)

    def get_length(self) -> int:
        # The Variables pane treats the length as the child count, and adds a "more values"
        # row when it is larger than the number of children it receives.
        return len(self.get_children())

    def get_size(self) -> int:
        data = self.value.data
        if _is_dask_array(data):
            return _dask_nbytes(data) or 0
        return int(getattr(data, "nbytes", 0))

    def has_children(self) -> bool:
        return True

    def get_children(self) -> list[str]:
        nddata = self.value
        children = []
        if _unit_text(nddata.unit) is not None:
            children.append(self.UNIT)
        if nddata.uncertainty is not None:
            children.append(self.UNCERTAINTY)
        if nddata.mask is not None:
            children.append(self.MASK)
        if nddata.wcs is not None:
            children.append(self.WCS)
        if nddata.meta:
            children.append(self.META)
        children.append(self.DATA)
        return children

    def has_child(self, key: Any) -> bool:
        return key in self.get_children()

    def get_child(self, key: Any) -> Any:
        nddata = self.value
        if key not in self.get_children():
            raise KeyError(key)
        if key == self.UNIT:
            return MetadataRow(_unit_text(nddata.unit) or "")
        if key == self.UNCERTAINTY:
            # The values of the uncertainty, or its class for one without values
            array = nddata.uncertainty.array
            return MetadataRow(type(nddata.uncertainty).__name__) if array is None else array
        if key == self.MASK:
            return nddata.mask
        if key == self.WCS:
            return MetadataRow(_wcs_text(nddata.wcs))
        if key == self.META:
            return nddata.meta
        return nddata.data

    def to_plaintext(self) -> str:
        return repr(self.value)


def _fits_header_int(header: Any, keyword: str, default: int = 0) -> int:
    """An integer keyword of a FITS header, or default when it is missing or malformed."""
    value = header.get(keyword, default)
    return value if isinstance(value, int) and not isinstance(value, bool) else default


def _fits_image_shape(header: Any) -> tuple[int, ...]:
    """
    The shape of the data of a FITS image HDU, read from its header, in numpy order.

    FITS numbers the axes from the fastest varying, so NAXIS1 is the last dim of the array.
    """
    naxis = _fits_header_int(header, "NAXIS")
    return tuple(_fits_header_int(header, f"NAXIS{n}") for n in range(naxis, 0, -1))


def _fits_image_dims(header: Any) -> list[str] | None:
    """The dims of a FITS image named by its CTYPEn keywords, or None when not every dim is."""
    naxis = _fits_header_int(header, "NAXIS")
    return _valid_dims([header.get(f"CTYPE{n}") for n in range(naxis, 0, -1)], naxis)


def _fits_data_nbytes(header: Any) -> int:
    """The bytes of the data of a FITS HDU, read from its header, as the FITS standard defines."""
    naxis = _fits_header_int(header, "NAXIS")
    if naxis == 0:
        return 0
    elements = math.prod(_fits_header_int(header, f"NAXIS{n}") for n in range(1, naxis + 1))
    bitpix = abs(_fits_header_int(header, "BITPIX", 8))
    gcount = _fits_header_int(header, "GCOUNT", 1)
    pcount = _fits_header_int(header, "PCOUNT", 0)
    return bitpix // 8 * gcount * (pcount + elements)


def _fits_is_scaled(header: Any) -> bool:
    """Whether the BSCALE or BZERO keywords of a FITS image scale its values when they are read."""
    return header.get("BSCALE", 1) != 1 or header.get("BZERO", 0) != 0


def _fits_image_read_nbytes(header: Any) -> int:
    """
    The memory that reading a FITS image takes, read from its header.

    astropy reads scaled values as a wider dtype than BITPIX, at most 8 bytes per value, so a
    scaled image counts 8 bytes per value.
    """
    elements = math.prod(_fits_image_shape(header))
    if _fits_is_scaled(header):
        return 8 * elements
    return abs(_fits_header_int(header, "BITPIX", 8)) // 8 * elements


def _fits_data_readable(hdu: Any) -> bool:
    """
    Whether reading the data of a FITS HDU would succeed.

    Data that was not read before its file was closed can no longer be read.
    """
    if getattr(hdu, "_data_loaded", True):
        return True
    return not getattr(getattr(hdu, "_file", None), "closed", False)


class FitsHDUInspector(PositronInspector[Any]):
    """
    A FITS HDU of any kind, such as an ImageHDU or a BinTableHDU.

    Every row reads the header only. The data is read when its row is expanded or viewed.
    HDUs are mutable and cannot be copied, so the Variables pane summarizes them again after every
    execution.
    """

    HEADER = "Header"
    DATA = "Data"

    def _has_data(self) -> bool:
        return _fits_header_int(self.value.header, "NAXIS") > 0

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        hdu = self.value
        header = hdu.header
        if not self._has_data():
            shape = "no data"
        elif _is_fits_image_hdu(hdu):
            shape = _shape_text(_fits_image_shape(header), _fits_image_dims(header))
        elif "TFIELDS" in header:
            rows = _fits_header_int(header, "NAXIS2")
            shape = (
                f"{_count(rows, 'row')} x {_count(_fits_header_int(header, 'TFIELDS'), 'column')}"
            )
        else:
            return _library_name(hdu), True
        text = f"[{shape}] {_library_name(hdu)}"
        return _maybe_truncate_string(text, level=level)[0], True

    def get_display_type(self) -> str:
        hdu = self.value
        name = type(hdu).__name__
        if not (_is_fits_image_hdu(hdu) and self._has_data()):
            return name
        header = hdu.header
        shape = "x".join(str(size) for size in _fits_image_shape(header))
        # Scaled data (BSCALE or BZERO) is read as another dtype than BITPIX gives.
        dtype = _FITS_BITPIX_DTYPES.get(_fits_header_int(header, "BITPIX"))
        if _fits_is_scaled(header) or dtype is None:
            return f"{name} [{shape}]"
        return f"{name} [{shape}] {dtype}"

    def get_kind(self) -> str:
        return "table"

    def has_viewer(self) -> bool:
        return _has_array_viewer(self.value)

    def get_length(self) -> int:
        # The Variables pane treats the length as the child count, and adds a "more values"
        # row when it is larger than the number of children it receives.
        return len(self.get_children())

    def get_size(self) -> int:
        return _fits_data_nbytes(self.value.header)

    def has_children(self) -> bool:
        return True

    def get_children(self) -> list[str]:
        if self._has_data() and _fits_data_readable(self.value):
            return [self.HEADER, self.DATA]
        return [self.HEADER]

    def has_child(self, key: Any) -> bool:
        return key in self.get_children()

    def get_child(self, key: Any) -> Any:
        if key == self.HEADER:
            return self.value.header
        if key == self.DATA and key in self.get_children():
            return self.value.data
        raise KeyError(key)

    def to_plaintext(self) -> str:
        return repr(self.value)


class FitsHDUListInspector(PositronInspector[Any]):
    """
    The HDUs of a FITS file, as astropy.io.fits.open returns them.

    Listing the HDUs reads every header, but no data. HDULists are mutable and cannot be copied,
    so the Variables pane summarizes them again after every execution.
    """

    CLASS_QNAME = _FITS_HDULIST_QNAMES

    def get_display_name(self, key: Any) -> str:
        # HDUs are keyed by index, since extension names can be missing or repeat.
        if type(key) is int and 0 <= key < len(self.value):
            name = self.value[key].name
            return f"{key}: {name}" if name else str(key)
        return str(key)

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        text = f"[{_count(len(self.value), 'HDU')}] {_library_name(self.value)}"
        return _maybe_truncate_string(text, level=level)[0], True

    def get_display_type(self) -> str:
        return f"{type(self.value).__name__} [{_count(len(self.value), 'HDU')}]"

    def get_kind(self) -> str:
        return "table"

    def get_length(self) -> int:
        return len(self.value)

    def get_size(self) -> int:
        return sum(_fits_data_nbytes(hdu.header) for hdu in self.value)

    def has_children(self) -> bool:
        return self.get_length() > 0

    def get_children(self) -> list[int]:
        return list(range(len(self.value)))

    def has_child(self, key: Any) -> bool:
        return type(key) is int and 0 <= key < len(self.value)

    def get_child(self, key: Any) -> Any:
        if not self.has_child(key):
            raise KeyError(key)
        return self.value[key]

    def to_plaintext(self) -> str:
        return repr(self.value)


class FitsRecordsInspector(NumpyNdarrayInspector):
    """The rows of a FITS table HDU, a numpy record array, shown as one row without children."""

    CLASS_QNAME = "astropy.io.fits.fitsrec.FITS_rec"

    def get_display_type(self) -> str:
        names = ", ".join(self.value.dtype.names or ())
        return f"{type(self.value).__name__} [{_count(len(self.value), 'row')}] ({names})"

    def get_kind(self) -> str:
        return "other"

    def get_length(self) -> int:
        return 0

    def has_children(self) -> bool:
        return False

    def get_children(self) -> list[Any]:
        return []


def _fits_card_text(card: Any) -> str:
    """The value of a FITS header card and its comment, such as "-32 / array data type"."""
    if card.keyword in ("COMMENT", "HISTORY", ""):
        # Commentary cards have text only.
        return str(card.value)
    value = card.value
    text = repr(value) if isinstance(value, str) else str(value)
    return f"{text} / {card.comment}" if card.comment else text


class FitsHeaderInspector(PositronInspector[Any]):
    """
    A FITS header.

    Its cards are keyed by index, since keywords such as COMMENT and HISTORY repeat.
    """

    CLASS_QNAME = _FITS_HEADER_QNAMES

    def get_display_name(self, key: Any) -> str:
        if self.has_child(key):
            return self.value.cards[key].keyword or f"card {key}"
        return str(key)

    def get_display_value(self, *, level: int = 0) -> tuple[str, bool]:
        keywords = list(self.value.keys())
        if not keywords:
            return _NONE_TEXT, False
        return _format_name_list(keywords, level=level)

    def get_display_type(self) -> str:
        return f"{type(self.value).__name__} [{_count(len(self.value), 'card')}]"

    def get_kind(self) -> str:
        return "map"

    def get_length(self) -> int:
        return len(self.value)

    def has_children(self) -> bool:
        return self.get_length() > 0

    def get_children(self) -> list[int]:
        return list(range(len(self.value)))

    def has_child(self, key: Any) -> bool:
        return type(key) is int and 0 <= key < len(self.value)

    def get_child(self, key: Any) -> Any:
        if not self.has_child(key):
            raise KeyError(key)
        return MetadataRow(_fits_card_text(self.value.cards[key]))

    def to_plaintext(self) -> str:
        return self.value.tostring(sep="\n")


INSPECTOR_CLASSES: dict[str, type[PositronInspector]] = {
    # xarray inspectors and their child rows
    **dict.fromkeys(XarrayDatasetInspector.CLASS_QNAME, XarrayDatasetInspector),
    **dict.fromkeys(XarrayDataArrayInspector.CLASS_QNAME, XarrayDataArrayInspector),
    **dict.fromkeys(XarrayVariableInspector.CLASS_QNAME, XarrayVariableInspector),
    **dict.fromkeys(DaskArrayInspector.CLASS_QNAME, DaskArrayInspector),
    # zarr and h5py inspectors
    **dict.fromkeys(ZarrArrayInspector.CLASS_QNAME, ZarrArrayInspector),
    **dict.fromkeys(ZarrGroupInspector.CLASS_QNAME, ZarrGroupInspector),
    **dict.fromkeys(H5pyDatasetInspector.CLASS_QNAME, H5pyDatasetInspector),
    **dict.fromkeys(H5pyGroupInspector.CLASS_QNAME, H5pyGroupInspector),
    # HyperSpy metadata trees and FITS files. HyperSpy signals and astropy quantities, NDData, and
    # HDUs have many classes, so get_inspector matches them by base class instead.
    **dict.fromkeys(HyperspyTreeInspector.CLASS_QNAME, HyperspyTreeInspector),
    **dict.fromkeys(FitsHDUListInspector.CLASS_QNAME, FitsHDUListInspector),
    **dict.fromkeys(FitsHeaderInspector.CLASS_QNAME, FitsHeaderInspector),
    FitsRecordsInspector.CLASS_QNAME: FitsRecordsInspector,
    get_qualname(MetadataRow): MetadataRowInspector,
    get_qualname(ArrayGroup): ArrayGroupInspector,
    # Geopandas inspectors (before pandas to take precedence)
    **dict.fromkeys(GeoDataFrameInspector.CLASS_QNAME, GeoDataFrameInspector),
    **dict.fromkeys(GeoSeriesInspector.CLASS_QNAME, GeoSeriesInspector),
    **dict.fromkeys(PandasDataFrameInspector.CLASS_QNAME, PandasDataFrameInspector),
    **dict.fromkeys(PandasSeriesInspector.CLASS_QNAME, PandasSeriesInspector),
    **dict.fromkeys(PandasIndexInspector.CLASS_QNAME, PandasIndexInspector),
    PandasTimestampInspector.CLASS_QNAME: PandasTimestampInspector,
    # pandas 3.0 changed Timestamp.__module__ from pandas._libs.tslibs.timestamps to pandas
    "pandas.Timestamp": PandasTimestampInspector,
    **dict.fromkeys(NumpyNumberInspector.CLASS_QNAME, NumpyNumberInspector),
    NumpyNdarrayInspector.CLASS_QNAME: NumpyNdarrayInspector,
    TorchTensorInspector.CLASS_QNAME: TorchTensorInspector,
    **dict.fromkeys(PolarsDataFrameInspector.CLASS_QNAME, PolarsDataFrameInspector),
    **dict.fromkeys(PolarsSeriesInspector.CLASS_QNAME, PolarsSeriesInspector),
    DatetimeInspector.CLASS_QNAME: DatetimeInspector,
    **dict.fromkeys(SQLiteConnectionInspector.CLASS_QNAME, SQLiteConnectionInspector),
    **dict.fromkeys(SQLAlchemyEngineInspector.CLASS_QNAME, SQLAlchemyEngineInspector),
    **dict.fromkeys(DuckDBConnectionInspector.CLASS_QNAME, DuckDBConnectionInspector),
    **dict.fromkeys(IbisDataFrameInspector.CLASS_QNAME, IbisDataFrameInspector),
    **dict.fromkeys(SnowflakeConnectionInspector.CLASS_QNAME, SnowflakeConnectionInspector),
    **dict.fromkeys(DatabricksConnectionInspector.CLASS_QNAME, DatabricksConnectionInspector),
    **dict.fromkeys(BigQueryConnectionInspector.CLASS_QNAME, BigQueryConnectionInspector),
    **dict.fromkeys(RedshiftConnectionInspector.CLASS_QNAME, RedshiftConnectionInspector),
    **dict.fromkeys(SQLServerConnectionInspector.CLASS_QNAME, SQLServerConnectionInspector),
    "ibis.Expr": IbisExprInspector,
    "boolean": BooleanInspector,
    "bytes": BytesInspector,
    "class": ClassInspector,
    "collection": CollectionInspector,
    "empty": NoneInspector,
    "function": FunctionInspector,
    "map": MapInspector,
    "number": NumberInspector,
    "other": ObjectInspector,
    "property": PropertyInspector,
    "string": StringInspector,
}

#
# Helper functions
#


def get_inspector(value: T) -> PositronInspector[T]:
    # Look for a specific inspector by qualified classname
    if isinstance(value, type):
        qualname = "type"
    elif isinstance(value, property):
        qualname = "property"
    else:
        qualname = _get_simplified_qualname(value)
    inspector_cls = INSPECTOR_CLASSES.get(qualname)

    # Guard: MapInspector should only be used for Mapping types, not map() iterators.
    # Python's builtin map() has qualname 'map' which collides with the 'map' key
    # intended for Mapping types like dict.
    if inspector_cls is MapInspector and not isinstance(value, Mapping):
        inspector_cls = None

    if inspector_cls is None:
        inspector_cls = _base_class_inspector(value)

    if inspector_cls is None:
        # Otherwise, look for an inspector by kind
        kind = _get_kind(value)
        inspector_cls = INSPECTOR_CLASSES.get(kind)

    # Otherwise, default to generic inspector
    if inspector_cls is None:
        inspector_cls = PositronInspector

    return inspector_cls(value)


def _base_class_inspector(value: Any) -> type[PositronInspector] | None:
    """
    The inspector of a library class that has many subclasses, matched by its base class.

    For example, HyperSpy and its extensions define many signal classes, all subclasses of
    BaseSignal. Each check is skipped when its library has not been imported.
    """
    if _is_hyperspy_signal(value):
        return HyperspySignalInspector
    if _is_quantity(value):
        return AstropyQuantityInspector
    if _is_nddata(value):
        return AstropyNDDataInspector
    if _is_fits_hdu(value):
        return FitsHDUInspector
    return None


def _get_kind(value: Any) -> str:
    if isinstance(value, str):
        return "string"
    elif isinstance(value, bool):
        return "boolean"
    elif isinstance(value, numbers.Number):
        return "number"
    elif isinstance(value, Mapping):
        return "map"
    elif isinstance(value, (bytes, bytearray, memoryview)):
        return "bytes"
    elif isinstance(value, (Sequence, AbstractSet)):
        return "collection"
    elif isinstance(value, (types.FunctionType, types.MethodType)):
        return "function"
    elif isinstance(value, type):
        return "class"
    elif value is not None:
        return "other"
    else:
        return "empty"


def _get_default_display_value(value: Any, *, level: int = 0) -> tuple[str, bool]:
    # Try to get the representation of the object.
    try:
        display_value = repr(value)
    except Exception:
        try:
            display_value = object.__repr__(value)
        except Exception:
            try:
                display_value = f"<no preview available for {get_qualname(value)}>"
            except Exception:
                display_value = "<no repr available for object>"

    return _maybe_truncate_string(display_value, level=level)


def _get_string_display_value(
    value: str | bytes | bytearray, *, level: int = 0
) -> tuple[str, bool]:
    # Limit the length of the displayed string; the limit depends on whether the string
    # is nested in another object.
    max_characters = MAX_CHARACTERS_NESTED if level > 0 else MAX_CHARACTERS

    # If the string fits the limit, return it as is.
    # NOTE: We check the actual string length but return the repr which may differ e.g.
    #       some characters may be escaped. We don't check `len(repr(value))` to avoid
    #       possibly copying a huge string.
    if len(value) <= max_characters:
        return repr(value), False

    # Split the character budget into two parts: 2/3 for the start and 1/3 for the end.
    start_characters = max(1, int(2 * max_characters / 3))
    end_characters = max(1, int(max_characters / 3))

    # Only compute the repr after slicing to avoid duplicating large strings
    # and remove surrounding quotes (and string prefixes).
    start = repr(value[:start_characters])
    start = start[: start.rindex("'")]
    end = repr(value[-end_characters:])
    end = end[end.index("'") + 1 :]
    return f"{start}{ELLIPSIS}{end}", True


def _get_collection_display_value(
    value: CollectionT | pd.Series | pl.Series | pd.Index,
    *,
    prefix: str,
    suffix: str,
    level: int = 0,
):
    # If the collection is empty, return the empty representation.
    # Use getattr with a default to handle objects that might not implement len
    if getattr(value, "__len__", lambda: 0)() == 0:
        return f"{prefix}{suffix}", False

    # If we're at the max nested level, just return ellipsis e.g. `[...]`.
    if level >= len(MAX_ITEMS_BY_LEVEL):
        return f"{prefix}{ELLIPSIS}{suffix}", True

    # Loop through the items in the collection and build the display value.
    parts = [prefix]
    item_allowance = MAX_ITEMS_BY_LEVEL[level]
    should_add_separator = False
    truncated = False
    for item in value:
        # Add a separator, if needed.
        if should_add_separator:
            parts.append(", ")
        should_add_separator = True

        # If we've reached the item allowance, add ellipsis and break.
        item_allowance -= 1
        if item_allowance <= 0:
            parts.append(ELLIPSIS)
            truncated = True
            break

        # Get the display value for the item.
        #
        # For items that are the same object as the parent, avoid infinite recursion
        # and represent the item with ellipsis.
        if item is value:
            item_repr, item_truncated = f"{prefix}{ELLIPSIS}{suffix}", True
        else:
            item_inspector = get_inspector(item)
            item_repr, item_truncated = item_inspector.get_display_value(level=level + 1)
        truncated = truncated or item_truncated

        parts.append(item_repr)

    # Special case: add a trailing comma for single-element tuples.
    if isinstance(value, tuple) and len(value) == 1:
        parts.append(",")

    parts.append(suffix)
    return "".join(parts), truncated


def _maybe_truncate_string(value: str, *, level: int = 0) -> tuple[str, bool]:
    # Limit the length of the displayed string; the limit depends on whether the string
    # is nested in another object.
    max_characters = MAX_CHARACTERS if level == 0 else MAX_CHARACTERS_NESTED

    # If the string is short enough, return it as is.
    if len(value) <= max_characters:
        return value, False

    # Truncate the string.
    # Split the character budget into two parts: 2/3 for the start and 1/3 for the end.
    start_characters = max(1, int(2 * max_characters / 3))
    end_characters = max(1, int(max_characters / 3))
    return f"{value[:start_characters]}{ELLIPSIS}{value[-end_characters:]}", True
