#
# Copyright (C) 2026 Posit Software, PBC. All rights reserved.
# Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
#

from __future__ import annotations

import logging
import sys
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd
import pytest

from ..data_explorer import (
    _VALUE_INF,
    _VALUE_NA,
    _VALUE_NAN,
    _VALUE_NAT,
    _VALUE_NEGINF,
    ARRAY_VIEW_MAX_LAZY_BYTES,
    ArrayView,
    DataExplorerService,
    _array_view_unavailable_message,
    _ArrayTable,
    _count_objects,
    _value_type_is_supported,
)
from ..data_explorer_comm import SupportStatus
from ..inspectors import get_inspector
from ..object_explorer import is_explorable
from ..utils import guid
from . import test_data_explorer
from .test_data_explorer import (
    _check_update_variable,
    _get_frequency_table,
    _get_histogram,
    _get_null_count,
    _profile_request,
)
from .utils import json_rpc_request, make_dataset

if TYPE_CHECKING:
    from .conftest import DummyComm
    from .test_data_explorer import DataExplorerFixture

dxf = test_data_explorer.dxf


def _column_values(dxf: DataExplorerFixture, name: str, column_index: int, num_rows: int):
    columns = [
        {"column_index": column_index, "spec": {"first_index": 0, "last_index": num_rows - 1}}
    ]
    return dxf.get_data_values(name, columns=columns)["columns"][0]


def _row_labels(dxf: DataExplorerFixture, name: str, num_rows: int):
    selection = {"first_index": 0, "last_index": num_rows - 1}
    return dxf.get_row_labels(name, selection)["row_labels"][0]


def test_array_view_numpy_2d(dxf: DataExplorerFixture):
    arr = np.arange(6, dtype="int64").reshape(2, 3)
    dxf.register_table("arr", arr)
    state = dxf.get_state("arr")
    assert (state["table_shape"]["num_rows"], state["table_shape"]["num_columns"]) == (2, 3)
    assert state["display_name"] == "arr"
    assert not state["has_row_labels"]
    assert state.get("row_axis_name") is None
    assert state.get("column_axis_name") is None
    schema = dxf.get_schema("arr")
    assert [c["column_name"] for c in schema] == ["0", "1", "2"]
    assert schema[0]["type_display"] == "integer"
    assert schema[0]["type_name"] == "int64"
    assert _column_values(dxf, "arr", 1, 2) == ["1", "4"]


def test_array_view_dataarray_2d(dxf: DataExplorerFixture):
    da = make_dataset()["sst"].isel(time=0)  # dims (lat, lon)
    dxf.register_table("sst", da)
    state = dxf.get_state("sst")
    assert state["display_name"] == "sst (lat, lon)"
    assert state["has_row_labels"]
    assert state["row_axis_name"] == "lat"
    assert state["column_axis_name"] == "lon"
    schema = dxf.get_schema("sst")
    assert [c["column_name"] for c in schema] == [str(v) for v in da.lon.to_numpy()]
    assert schema[0]["type_display"] == "floating"
    num_rows = da.sizes["lat"]
    assert _row_labels(dxf, "sst", num_rows) == [str(v) for v in da.lat.to_numpy()]
    assert _column_values(dxf, "sst", 1, num_rows) == ["1.00", "6.00", "11.00"]


def test_array_view_corner_names_units(dxf: DataExplorerFixture):
    da = make_dataset()["sst"].isel(time=0)
    da.lat.attrs["units"] = "degrees_north"
    da.lon.attrs["units"] = "degrees_east"
    dxf.register_table("sst", da)
    state = dxf.get_state("sst")
    assert (state["row_axis_name"], state["column_axis_name"]) == (
        "lat (degrees_north)",
        "lon (degrees_east)",
    )
    # The title names the dims alone.
    assert state["display_name"] == "sst (lat, lon)"


def test_array_view_index_variable_units(dxf: DataExplorerFixture):
    ds = make_dataset()
    ds.lat.attrs["units"] = "degrees_north"
    dxf.register_table("lat", ds.variables["lat"])
    assert dxf.get_state("lat")["row_axis_name"] == "lat (degrees_north)"
    # A Variable that is not a coordinate has no row axis units.
    ds.sst.attrs["units"] = "degC"
    dxf.register_table("v", ds.variables["sst"].isel(time=0))
    assert dxf.get_state("v")["row_axis_name"] == "lat"


def test_array_view_dataarray_2d_without_coordinates(dxf: DataExplorerFixture):
    xr = pytest.importorskip("xarray")
    da = xr.DataArray(np.zeros((2, 3)), dims=("y", "x"), name="z")
    dxf.register_table("z", da)
    assert [c["column_name"] for c in dxf.get_schema("z")] == ["0", "1", "2"]
    # Without a row coordinate the grid shows its own row numbers.
    assert not dxf.get_state("z")["has_row_labels"]


def test_array_view_1d(dxf: DataExplorerFixture):
    dxf.register_table("vec", np.array([1.5, 2.5, 3.5]))
    state = dxf.get_state("vec")
    assert (state["table_shape"]["num_rows"], state["table_shape"]["num_columns"]) == (3, 1)
    assert [c["column_name"] for c in dxf.get_schema("vec")] == ["0"]
    assert _column_values(dxf, "vec", 0, 3) == ["1.50", "2.50", "3.50"]

    lat = make_dataset()["lat"]
    dxf.register_table("lat", lat)
    state = dxf.get_state("lat")
    assert state["display_name"] == "lat (lat)"
    # ds.lat is its own dim coordinate: its column already names the dim, so the corner does not.
    assert state.get("row_axis_name") is None
    assert state.get("column_axis_name") is None
    assert [c["column_name"] for c in dxf.get_schema("lat")] == ["lat"]


def test_array_view_axis_names_1d(dxf: DataExplorerFixture):
    ds = make_dataset()
    # The time coordinate itself: no axis names, since its column is already named "time".
    dxf.register_table("time", ds["time"])
    state = dxf.get_state("time")
    assert state.get("row_axis_name") is None
    assert state.get("column_axis_name") is None

    # A data variable along time: the rows run along time.
    dxf.register_table("point", ds["sst"].isel(lat=0, lon=0))
    state = dxf.get_state("point")
    assert state["row_axis_name"] == "time"
    assert state.get("column_axis_name") is None


def test_array_view_axis_names_without_row_coordinate(dxf: DataExplorerFixture):
    xr = pytest.importorskip("xarray")
    # Rows without a coordinate show row numbers, but the name still says what they run along.
    dxf.register_table("z", xr.DataArray(np.zeros((2, 3)), dims=("y", "x"), name="z"))
    state = dxf.get_state("z")
    assert (state["row_axis_name"], state["column_axis_name"]) == ("y", "x")

    dxf.register_table("v", xr.DataArray(np.zeros(2), dims=("y",), name="v"))
    assert dxf.get_state("v")["row_axis_name"] == "y"


def test_array_view_axis_names_of_variables(dxf: DataExplorerFixture):
    ds = make_dataset()
    dxf.register_table("grid", ds.variables["sst"][0])
    state = dxf.get_state("grid")
    assert (state["row_axis_name"], state["column_axis_name"]) == ("lat", "lon")

    # An unnamed IndexVariable's column is "0", so the corner names the dim its rows run along.
    dxf.register_table("t", ds.variables["time"])
    state = dxf.get_state("t")
    assert state["row_axis_name"] == "time"
    assert state.get("column_axis_name") is None


def test_array_view_datetime_1d(dxf: DataExplorerFixture):
    da = make_dataset()["time"]
    dxf.register_table("time", da)
    schema = dxf.get_schema("time")
    assert schema[0]["type_display"] == "datetime"
    values = _column_values(dxf, "time", 0, da.sizes["time"])
    series = pd.Series(da.to_numpy())
    dxf.register_table("series", series)
    expected = _column_values(dxf, "series", 0, len(series))
    assert values == expected
    assert values[0] == "2020-01-01 00:00:00"


def test_array_view_unnamed_1d_dataarray_column_name(dxf: DataExplorerFixture):
    xr = pytest.importorskip("xarray")
    dxf.register_table("vec", xr.DataArray(np.array([1.0, 2.0]), dims=("x",)))
    assert [c["column_name"] for c in dxf.get_schema("vec")] == ["0"]


def test_array_view_masked_array(dxf: DataExplorerFixture):
    arr = np.ma.array([[1.0, 2.0]], mask=[[False, True]])
    dxf.register_table("arr", arr)
    assert _column_values(dxf, "arr", 0, 1) == ["1.00"]
    assert _column_values(dxf, "arr", 1, 1) == [_VALUE_NA]
    result = dxf.get_column_profiles("arr", [_get_null_count(0), _get_null_count(1)])
    assert [r["null_count"] for r in result] == [0, 1]


def test_array_view_dtype_display(dxf: DataExplorerFixture):
    cases = [
        (np.array([True, False]), "boolean"),
        (np.array([1, 2], dtype="uint8"), "integer"),
        (np.array(["a", "b"]), "string"),
        (np.array([b"a", b"b"]), "string"),
        (np.array([1 + 1j, 2j]), "object"),
        (np.array([object(), None], dtype=object), "object"),
    ]
    for i, (arr, expected) in enumerate(cases):
        name = f"arr{i}"
        dxf.register_table(name, arr)
        assert dxf.get_schema(name)[0]["type_display"] == expected, arr.dtype


def test_array_view_nan_values(dxf: DataExplorerFixture):
    arr = np.array([[1.0, np.nan], [np.inf, -np.inf]])
    dxf.register_table("arr", arr)
    assert _column_values(dxf, "arr", 0, 2) == ["1.00", _VALUE_INF]
    assert _column_values(dxf, "arr", 1, 2) == [_VALUE_NAN, _VALUE_NEGINF]

    times = np.array(["2020-01-01", "NaT"], dtype="datetime64[ns]")
    dxf.register_table("times", times)
    assert _column_values(dxf, "times", 0, 2)[1] == _VALUE_NAT


def test_array_view_dask_under_limit(dxf: DataExplorerFixture):
    dask_array = pytest.importorskip("dask.array")
    arr = np.arange(12, dtype="int64").reshape(4, 3)
    lazy = dask_array.from_array(arr, chunks=(2, 3))
    dxf.register_table("lazy", lazy)
    state = dxf.get_state("lazy")
    assert (state["table_shape"]["num_rows"], state["table_shape"]["num_columns"]) == (4, 3)
    assert state.get("row_axis_name") is None
    assert state.get("column_axis_name") is None
    assert _column_values(dxf, "lazy", 2, 4) == ["2", "5", "8", "11"]


def test_value_type_is_supported_arrays():
    assert _value_type_is_supported(np.zeros(2))
    assert _value_type_is_supported(np.zeros((2, 2)))
    assert not _value_type_is_supported(np.zeros(()))
    assert not _value_type_is_supported(np.zeros((2, 2, 2)))
    ds = make_dataset()
    assert not _value_type_is_supported(ds)  # Dataset never
    assert _value_type_is_supported(ds["sst"].isel(time=0))
    assert not _value_type_is_supported(ds["sst"])

    dask_array = pytest.importorskip("dask.array")
    assert _value_type_is_supported(dask_array.zeros((10, 10)))
    big = dask_array.zeros((6000, 6000))  # 288 MB, never computed
    assert big.nbytes > ARRAY_VIEW_MAX_LAZY_BYTES
    assert not _value_type_is_supported(big)


def test_array_view_variable_update(dxf: DataExplorerFixture):
    comm_id = dxf.assign_and_open_viewer("arr", np.zeros((2, 2)))
    assert isinstance(dxf.de_service.table_views[comm_id], ArrayView)

    dxf.assign_variable("arr", np.ones((3, 2)))
    assert dxf.get_state("arr")["table_shape"]["num_rows"] == 3

    # A 3-D array has no Data Explorer view, so the explorer closes
    dxf.assign_variable("arr", np.zeros((2, 2, 2)))
    assert comm_id not in dxf.de_service.comms


def test_array_view_open_data_explorer(de_service: DataExplorerService):
    comm_id = guid()
    de_service.register_table(np.zeros((2, 3)), "arr", comm_id=comm_id)
    comm = cast("DummyComm", de_service.comms[comm_id].comm)
    comm.handle_msg(json_rpc_request("open_data_explorer", comm_id=comm_id))

    new_comm_ids = [cid for cid in de_service.comms if cid != comm_id]
    assert len(new_comm_ids) == 1
    new_view = de_service.table_views[new_comm_ids[0]]
    assert isinstance(new_view, ArrayView)
    assert new_view.table.shape == (2, 3)


def _summary(dxf: DataExplorerFixture, name: str, column_index: int):
    request = _profile_request(column_index, [{"profile_type": "summary_stats"}])
    return dxf.get_column_profiles(name, [request])[0]["summary_stats"]


def test_array_view_summary_stats(dxf: DataExplorerFixture):
    arr = np.array([[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]])
    dxf.register_table("arr", arr)
    dxf.register_table("df", pd.DataFrame(arr))
    for i in range(2):
        assert _summary(dxf, "arr", i) == _summary(dxf, "df", i)
    stats = _summary(dxf, "arr", 0)["number_stats"]
    assert (stats["min_value"], stats["max_value"]) == ("1.00", "3.00")
    assert (stats["mean"], stats["median"]) == ("2.00", "2.00")


def test_array_view_summary_stats_other_dtypes(dxf: DataExplorerFixture):
    cases = {
        "ints": np.array([1, 2, 3, 4]),
        "bools": np.array([True, False, True]),
        "strs": np.array(["a", "", "b", "a"]),
        "times": np.array(["2020-01-01", "2020-01-03", "2020-01-08"], dtype="datetime64[ns]"),
    }
    for name, arr in cases.items():
        dxf.register_table(name, arr)
        dxf.register_table(f"{name}_pd", pd.DataFrame({"0": arr}))
        assert _summary(dxf, name, 0) == _summary(dxf, f"{name}_pd", 0), name


def test_array_view_stats_ignore_nan(dxf: DataExplorerFixture):
    arr = np.array([[1.0], [np.nan], [3.0]])
    dxf.register_table("arr", arr)
    result = dxf.get_column_profiles("arr", [_get_null_count(0)])
    assert result[0]["null_count"] == 1
    assert _summary(dxf, "arr", 0)["number_stats"]["mean"] == "2.00"


def test_array_view_histogram(dxf: DataExplorerFixture):
    arr = np.array([[1.0], [2.0], [2.5], [np.nan], [10.0]])
    dxf.register_table("arr", arr)
    dxf.register_table("df", pd.DataFrame(arr))
    request = _get_histogram(0, bins=4)
    result = dxf.get_column_profiles("arr", [request])[0]["small_histogram"]
    assert result == dxf.get_column_profiles("df", [request])[0]["small_histogram"]
    assert len(result["bin_edges"]) == 5
    assert sum(result["bin_counts"]) == 4  # NaN excluded


def test_array_view_histogram_datetime(dxf: DataExplorerFixture):
    times = np.array(["2020-01-01", "2020-01-03", "2020-01-08"], dtype="datetime64[ns]")
    dxf.register_table("times", times)
    dxf.register_table("times_pd", pd.DataFrame({"0": times}))
    request = _get_histogram(0, bins=3)
    result = dxf.get_column_profiles("times", [request])[0]["small_histogram"]
    assert result == dxf.get_column_profiles("times_pd", [request])[0]["small_histogram"]


def test_array_view_frequency_table(dxf: DataExplorerFixture):
    arr = np.array([3, 1, 3, 2, 3, 1])
    dxf.register_table("arr", arr)
    result = dxf.get_column_profiles("arr", [_get_frequency_table(0, 2)])
    table = result[0]["small_frequency_table"]
    assert table == {"values": ["3", "1"], "counts": [3, 2], "other_count": 1}


def test_array_view_features():
    f = ArrayView.FEATURES
    assert f.get_column_profiles.support_status == SupportStatus.Supported
    assert f.set_sort_columns.support_status == SupportStatus.Unsupported
    assert f.set_row_filters.support_status == SupportStatus.Unsupported
    assert f.search_schema.support_status == SupportStatus.Unsupported
    assert f.export_data_selection.support_status == SupportStatus.Unsupported
    assert f.convert_to_code.support_status == SupportStatus.Unsupported


#
# Lazily indexed (not dask) DataArrays, such as variables opened from a file without chunks
#


def _lazily_indexed_dataarray(shape: tuple[int, int], dtype: Any, values: Any = None) -> Any:
    """A 2-D DataArray read lazily from a backend. Without values, any read fails the test."""
    xr = pytest.importorskip("xarray")
    from xarray.backends import BackendArray
    from xarray.core import indexing

    class LazyBackendArray(BackendArray):
        def __init__(self) -> None:
            self.shape = shape
            self.dtype = np.dtype(dtype)

        def __getitem__(self, key: Any) -> np.ndarray:
            return indexing.explicit_indexing_adapter(
                key, self.shape, indexing.IndexingSupport.BASIC, self._raw_getitem
            )

        def _raw_getitem(self, key: tuple) -> np.ndarray:
            if values is None:
                raise AssertionError("the lazily indexed array was read")
            return values[key]

    variable = xr.Variable(("y", "x"), indexing.LazilyIndexedArray(LazyBackendArray()))
    return xr.DataArray(variable, name="z")


def test_lazily_indexed_dataarray_over_limit_is_not_viewable():
    # 40000 x 40000 float32 is 6.4 GB on disk, never read
    da = _lazily_indexed_dataarray((40000, 40000), "float32")
    assert da.nbytes > ARRAY_VIEW_MAX_LAZY_BYTES
    assert not _value_type_is_supported(da)
    assert not get_inspector(da).has_viewer()
    with pytest.raises(TypeError):
        _ArrayTable.from_value(da)


def test_lazily_indexed_dataarray_under_limit_opens(dxf: DataExplorerFixture):
    values = np.arange(6.0).reshape(2, 3)
    da = _lazily_indexed_dataarray((2, 3), values.dtype, values)
    assert _value_type_is_supported(da)
    assert get_inspector(da).has_viewer()
    dxf.register_table("lazy", da)
    assert _column_values(dxf, "lazy", 1, 2) == ["1.00", "4.00"]


def _lazy_object_arrays() -> list[Any]:
    """Lazy object-dtype arrays, whose itemsize counts pointers, not the objects they load."""
    xr = pytest.importorskip("xarray")
    from .utils import make_compute_bomb

    # The bomb makes any computation fail the test.
    bomb = make_compute_bomb(np.array(["x" * 1000] * 4, dtype=object), chunks=2)
    return [
        bomb,
        xr.DataArray(bomb, dims=("x",), name="lazy"),
        _lazily_indexed_dataarray((2, 3), object),
    ]


def test_lazy_object_arrays_are_not_viewable():
    for value in _lazy_object_arrays():
        assert not _value_type_is_supported(value)
        assert not get_inspector(value).has_viewer()
        assert _array_view_unavailable_message(value, "lazy") == (
            "lazy has object dtype and is not in memory, so the memory it needs is not known "
            "until it is loaded. Call .compute() first."
        )


def test_in_memory_object_arrays_stay_viewable():
    xr = pytest.importorskip("xarray")
    values = np.array(["a", None, 3], dtype=object)
    for value in (values, xr.DataArray(values, dims=("x",), name="obj")):
        assert _value_type_is_supported(value)
        assert get_inspector(value).has_viewer()
        assert _array_view_unavailable_message(value, "obj") is None


#
# Variable updates of an open array explorer
#


def test_array_view_update_same_schema_is_data_update(dxf: DataExplorerFixture):
    dxf.assign_and_open_viewer("arr", np.zeros((2, 2)))
    dxf.assign_variable("arr", np.ones((2, 2)))
    _check_update_variable(dxf.de_service, "arr", update_type="data")
    assert _column_values(dxf, "arr", 0, 2) == ["1.00", "1.00"]

    da = make_dataset()["sst"].isel(time=0)
    dxf.assign_and_open_viewer("da", da)
    dxf.assign_variable("da", da + 1)
    _check_update_variable(dxf.de_service, "da", update_type="data")


def test_array_view_update_changed_schema_is_schema_update(dxf: DataExplorerFixture):
    dxf.assign_and_open_viewer("arr", np.zeros((2, 2)))
    dxf.assign_variable("arr", np.zeros((3, 2)))
    _check_update_variable(dxf.de_service, "arr", update_type="schema")
    dxf.assign_variable("arr", np.arange(6, dtype="int64").reshape(3, 2))
    _check_update_variable(dxf.de_service, "arr", update_type="schema")
    dxf.assign_variable("arr", np.arange(12, dtype="int64").reshape(3, 4))
    _check_update_variable(dxf.de_service, "arr", update_type="schema")

    da = make_dataset()["sst"].isel(time=0)
    dxf.assign_and_open_viewer("da", da)
    # Same shape and dtype, different row labels
    dxf.assign_variable("da", da.assign_coords(lat=da.lat + 1))
    _check_update_variable(dxf.de_service, "da", update_type="schema")


def test_array_view_update_renamed_dims_is_schema_update(dxf: DataExplorerFixture):
    # The title names the dims, so renaming them needs a schema update even when the shape,
    # dtype, column names, and row labels all match
    da = make_dataset()["sst"].isel(time=0)
    dxf.assign_and_open_viewer("da", da)
    dxf.assign_variable("da", da.rename({"lat": "y", "lon": "x"}))
    _check_update_variable(dxf.de_service, "da", update_type="schema")


#
# Messages for arrays the Data Explorer cannot show
#


def test_array_view_unavailable_message():
    ds = make_dataset()
    assert _array_view_unavailable_message(ds, "ds") == (
        "ds is an xarray Dataset; view one of its variables, e.g. %view ds['sst']"
    )
    assert _array_view_unavailable_message(ds["sst"], "ds.sst") == (
        "ds.sst has 3 dimensions; the Data Explorer shows 1-D and 2-D arrays. "
        "Select a slice first, e.g. ds.sst.isel(time=0)"
    )
    assert _array_view_unavailable_message(np.zeros((2, 2, 2)), "arr") == (
        "arr has 3 dimensions; the Data Explorer shows 1-D and 2-D arrays. "
        "Select a slice first, e.g. arr[0]"
    )
    assert _array_view_unavailable_message(np.zeros((2, 2)), "arr") is None
    assert _array_view_unavailable_message(object(), "x") is None

    da = _lazily_indexed_dataarray((40000, 40000), "float32")
    assert _array_view_unavailable_message(da, "z") == (
        "z is 6.0 GiB (not in memory), over the 100.0 MiB limit for the Data Explorer. "
        "Select a smaller slice or call .compute() first."
    )

    dask_array = pytest.importorskip("dask.array")
    assert _array_view_unavailable_message(dask_array.zeros((2, 2, 2)), "lazy") == (
        "lazy has 3 dimensions; the Data Explorer shows 1-D and 2-D arrays. "
        "Select a slice first, e.g. lazy[0]"
    )
    big = dask_array.zeros((6000, 6000))
    assert _array_view_unavailable_message(big, "big") == (
        "big is 274.7 MiB (not in memory), over the 100.0 MiB limit for the Data Explorer. "
        "Select a smaller slice or call .compute() first."
    )


def test_array_view_unavailable_message_unsafe_names():
    xr = pytest.importorskip("xarray")
    # A variable named like a Dataset method gets bracket indexing, which cannot resolve to the
    # method
    ds = xr.Dataset({"isel": (("a", "b"), np.zeros((2, 2)))})
    assert _array_view_unavailable_message(ds, "ds") == (
        "ds is an xarray Dataset; view one of its variables, e.g. %view ds['isel']"
    )
    # A variable name with a space needs the %view argument quoted
    ds = xr.Dataset({"my var": (("a", "b"), np.zeros((2, 2)))})
    assert _array_view_unavailable_message(ds, "ds") == (
        "ds is an xarray Dataset; view one of its variables, e.g. %view \"ds['my var']\""
    )
    # A dim named with a keyword cannot be a keyword argument
    da = xr.DataArray(np.zeros((2, 2, 2)), dims=("class", "y", "x"))
    assert _array_view_unavailable_message(da, "da") == (
        "da has 3 dimensions; the Data Explorer shows 1-D and 2-D arrays. "
        "Select a slice first, e.g. da.isel({'class': 0})"
    )


def test_array_view_unavailable_message_parenthesizes_expressions():
    xr = pytest.importorskip("xarray")
    da = xr.DataArray(np.zeros((2, 2, 2)), dims=("time", "y", "x"))
    assert _array_view_unavailable_message(da, "left + right") == (
        "left + right has 3 dimensions; the Data Explorer shows 1-D and 2-D arrays. "
        "Select a slice first, e.g. (left + right).isel(time=0)"
    )
    assert _array_view_unavailable_message(np.zeros((2, 2, 2)), "-arr") == (
        "-arr has 3 dimensions; the Data Explorer shows 1-D and 2-D arrays. "
        "Select a slice first, e.g. (-arr)[0]"
    )
    # Name, attribute, subscript, and call chains bind tighter than the suggested suffix
    for expr in ("ds.sst", "ds['sst']", "ds.sel(lat=0)", "f(x)[0].y"):
        assert f"e.g. {expr}.isel(time=0)" in cast("str", _array_view_unavailable_message(da, expr))

    ds = xr.Dataset({"sst": (("a", "b"), np.zeros((2, 2)))})
    assert _array_view_unavailable_message(ds, "a+b") == (
        "a+b is an xarray Dataset; view one of its variables, e.g. %view \"(a + b)['sst']\""
    )


def test_array_view_unavailable_message_ignores_comments():
    # Suggestions are built from the parsed expression, so a trailing comment cannot swallow
    # the appended slice or variable
    xr = pytest.importorskip("xarray")
    da = xr.DataArray(np.zeros((2, 2, 2)), dims=("time", "y", "x"))
    assert _array_view_unavailable_message(da, "arr # note") == (
        "arr # note has 3 dimensions; the Data Explorer shows 1-D and 2-D arrays. "
        "Select a slice first, e.g. arr.isel(time=0)"
    )
    assert "e.g. (left + right).isel(time=0)" in cast(
        "str", _array_view_unavailable_message(da, "left + right  # sum")
    )
    ds = xr.Dataset({"sst": (("a", "b"), np.zeros((2, 2)))})
    assert _array_view_unavailable_message(ds, "ds # note") == (
        "ds # note is an xarray Dataset; view one of its variables, e.g. %view ds['sst']"
    )


def test_array_view_unavailable_message_without_a_valid_command():
    # %view cannot take an argument with a double quote on Windows, where only double quotes
    # quote arguments, so these names get plain guidance instead of a command
    xr = pytest.importorskip("xarray")
    # IPython replaces {x} in a magic's arguments with the value of x
    for name in ("a' \"b", "it's", "{x}"):
        ds = xr.Dataset({name: (("a", "b"), np.zeros((2, 2)))})
        assert _array_view_unavailable_message(ds, "ds") == (
            f"ds is an xarray Dataset; view one of its data variables: {name}"
        )
    # A numpy string name has a literal form
    ds = xr.Dataset({np.str_("sst"): (("a", "b"), np.zeros((2, 2)))})
    assert _array_view_unavailable_message(ds, "ds") == (
        "ds is an xarray Dataset; view one of its variables, e.g. %view ds['sst']"
    )
    # A name with no literal form
    ds = xr.Dataset({float("nan"): (("a", "b"), np.zeros((2, 2)))})
    assert _array_view_unavailable_message(ds, "ds") == (
        "ds is an xarray Dataset; view one of its data variables: nan"
    )


#
# Structured dtypes and datetime labels
#


def test_structured_array_is_not_viewable():
    arr = np.zeros(3, dtype=[("a", "i4"), ("b", "f8")])
    assert not _value_type_is_supported(arr)
    assert not get_inspector(arr).has_viewer()


def test_structured_array_falls_back_to_object_explorer():
    # A structured array has no Data Explorer view, so it keeps kind "collection" and %view opens
    # it in the Object Explorer
    for arr in (
        np.zeros(3, dtype=[("a", "i4"), ("b", "f8")]),
        np.zeros((2, 3), dtype=[("a", "i4"), ("b", "f8")]),
    ):
        inspector = get_inspector(arr)
        assert inspector.get_kind() == "collection"
        assert not inspector.has_viewer()
        assert is_explorable(arr)


def test_array_view_datetime_labels(dxf: DataExplorerFixture):
    da = make_dataset()["sst"].isel(lat=0)  # dims (time, lon)
    dxf.register_table("rows", da)
    assert _row_labels(dxf, "rows", 2)[:2] == ["2020-01-01 00:00:00", "2020-01-02 00:00:00"]

    dxf.register_table("cols", da.transpose("lon", "time"))
    columns = [c["column_name"] for c in dxf.get_schema("cols")]
    assert columns[:2] == ["2020-01-01 00:00:00", "2020-01-02 00:00:00"]


#
# Column labels (the column name tooltip) and redundant row labels
#


def _column_labels(dxf: DataExplorerFixture, name: str) -> list[Any]:
    return [c["column_label"] for c in dxf.get_schema(name)]


def test_array_view_column_labels_name_dim_and_coordinate(dxf: DataExplorerFixture):
    xr = pytest.importorskip("xarray")
    da = xr.DataArray(
        np.zeros((2, 2), dtype="float32"),
        dims=("lat", "lon"),
        coords={
            "lat": [10.5, 11.5],
            "lon": ("lon", [275.125, 275.375], {"units": "degrees_east"}),
        },
        name="sst",
    )
    dxf.register_table("sst", da)
    assert _column_labels(dxf, "sst") == [
        "lon = 275.125 degrees_east",
        "lon = 275.375 degrees_east",
    ]
    # Without units, the label has no trailing units
    dxf.register_table("flipped", da.transpose("lon", "lat"))
    assert _column_labels(dxf, "flipped") == ["lat = 10.5", "lat = 11.5"]


def test_array_view_column_labels_use_index_without_coordinate(dxf: DataExplorerFixture):
    xr = pytest.importorskip("xarray")
    dxf.register_table("z", xr.DataArray(np.zeros((2, 4)), dims=("y", "x"), name="z"))
    assert _column_labels(dxf, "z") == ["x = 0", "x = 1", "x = 2", "x = 3"]


def test_array_view_column_labels_format_datetimes(dxf: DataExplorerFixture):
    da = make_dataset()["sst"].isel(lat=0).transpose("lon", "time")
    dxf.register_table("cols", da)
    labels = _column_labels(dxf, "cols")
    assert labels[:2] == ["time = 2020-01-01 00:00:00", "time = 2020-01-02 00:00:00"]


def test_array_view_column_label_1d(dxf: DataExplorerFixture):
    xr = pytest.importorskip("xarray")
    da = make_dataset()["sst"].isel(lat=0, lon=0)
    dxf.register_table("series", da)
    assert _column_labels(dxf, "series") == ["sst (time)"]
    da = da.assign_attrs(units="degC")
    dxf.register_table("with_units", da)
    assert _column_labels(dxf, "with_units") == ["sst (time) degC"]
    dxf.register_table("unnamed", xr.DataArray(np.zeros(2), dims=("x",)))
    assert _column_labels(dxf, "unnamed") == ["0 (x)"]


def test_array_view_numpy_has_no_column_labels(dxf: DataExplorerFixture):
    dxf.register_table("arr", np.zeros((2, 3)))
    assert _column_labels(dxf, "arr") == [None, None, None]
    dxf.register_table("vec", np.zeros(3))
    assert _column_labels(dxf, "vec") == [None]


def test_array_view_column_labels_of_lazily_indexed_array(dxf: DataExplorerFixture):
    values = np.arange(6.0).reshape(2, 3)
    da = _lazily_indexed_dataarray((2, 3), values.dtype, values)
    da = da.assign_coords(x=("x", [1, 2, 3], {"units": "m"}))
    dxf.register_table("lazy", da)
    assert _column_labels(dxf, "lazy") == ["x = 1 m", "x = 2 m", "x = 3 m"]


def test_array_view_own_coordinate_has_no_row_labels(dxf: DataExplorerFixture):
    ds = make_dataset()
    for name in ("time", "lat"):
        dxf.register_table(name, ds[name])
        assert not dxf.get_state(name)["has_row_labels"]


def test_array_view_1d_data_variable_keeps_row_labels(dxf: DataExplorerFixture):
    da = make_dataset()["sst"].isel(lat=0, lon=0)  # indexed by time
    dxf.register_table("series", da)
    assert dxf.get_state("series")["has_row_labels"]
    assert _row_labels(dxf, "series", 2) == ["2020-01-01 00:00:00", "2020-01-02 00:00:00"]
    # A same-named coordinate that is not the DataArray itself still labels the rows
    xr = pytest.importorskip("xarray")
    distinct = xr.DataArray([1, 2], dims="x", coords={"x": [10, 20]}, name="x")
    dxf.register_table("distinct", distinct)
    assert dxf.get_state("distinct")["has_row_labels"]
    assert _row_labels(dxf, "distinct", 2) == ["10", "20"]


def test_array_view_dataarray_named_after_dim_without_coordinate(dxf: DataExplorerFixture):
    xr = pytest.importorskip("xarray")
    dxf.register_table("x", xr.DataArray([1, 2], dims="x", name="x"))
    schema = dxf.get_schema("x")
    assert [c["column_name"] for c in schema] == ["x"]
    assert [c["column_label"] for c in schema] == ["x (x)"]
    # Without a coordinate the grid shows its own row numbers.
    assert not dxf.get_state("x")["has_row_labels"]
    assert _column_values(dxf, "x", 0, 2) == ["1", "2"]

    # 2-D, with a coordinate on the row dim only
    da = xr.DataArray(np.zeros((2, 3)), dims=("y", "x"), coords={"y": [5, 6]}, name="x")
    dxf.register_table("grid", da)
    assert _row_labels(dxf, "grid", 2) == ["5", "6"]
    assert [c["column_label"] for c in dxf.get_schema("grid")] == ["x = 0", "x = 1", "x = 2"]


def test_array_view_row_dim_without_coordinate_has_no_row_labels(dxf: DataExplorerFixture):
    xr = pytest.importorskip("xarray")
    dxf.register_table("vec", xr.DataArray(np.arange(3.0), dims=("x",), name="v"))
    assert not dxf.get_state("vec")["has_row_labels"]

    # 2-D: a coordinate on the column dim only. Columns keep coordinate names and labels.
    da = xr.DataArray(np.zeros((2, 3)), dims=("y", "x"), coords={"x": [7, 8, 9]}, name="z")
    dxf.register_table("cols", da)
    assert not dxf.get_state("cols")["has_row_labels"]
    schema = dxf.get_schema("cols")
    assert [c["column_name"] for c in schema] == ["7", "8", "9"]
    assert [c["column_label"] for c in schema] == ["x = 7", "x = 8", "x = 9"]

    # 2-D without any coordinate: columns are named by index and keep the tooltip.
    dxf.register_table("bare", xr.DataArray(np.zeros((2, 2)), dims=("y", "x")))
    assert not dxf.get_state("bare")["has_row_labels"]
    assert [c["column_label"] for c in dxf.get_schema("bare")] == ["x = 0", "x = 1"]


#
# Object columns
#


def test_array_view_object_nulls(dxf: DataExplorerFixture):
    arr = np.array(["a", None, "b", np.nan], dtype=object)
    dxf.register_table("arr", arr)
    dxf.register_table("df", pd.DataFrame({"0": arr}))
    result = dxf.get_column_profiles("arr", [_get_null_count(0)])
    assert result[0]["null_count"] == 2
    assert result == dxf.get_column_profiles("df", [_get_null_count(0)])
    # pandas infers a string column here, while every object array column reads as object
    assert _summary(dxf, "arr", 0)["other_stats"]["num_unique"] == 2
    assert _summary(dxf, "df", 0)["string_stats"]["num_unique"] == 2


def test_array_view_object_frequency_table_by_value(dxf: DataExplorerFixture):
    arr = np.array([1, "1", 1], dtype=object)
    dxf.register_table("arr", arr)
    dxf.register_table("df", pd.DataFrame({"0": arr}))
    request = _get_frequency_table(0, 5)
    table = dxf.get_column_profiles("arr", [request])[0]["small_frequency_table"]
    assert table == {"values": ["1", "1"], "counts": [2, 1], "other_count": 0}
    assert table == dxf.get_column_profiles("df", [request])[0]["small_frequency_table"]
    assert _summary(dxf, "arr", 0)["other_stats"]["num_unique"] == 2


def test_array_view_object_unorderable_values(dxf: DataExplorerFixture):
    arr = np.array([1, "a", None, [1, 2], [1, 2], {"k": 1}], dtype=object)
    dxf.register_table("arr", arr)
    table = dxf.get_column_profiles("arr", [_get_frequency_table(0, 2)])[0]
    assert table["small_frequency_table"]["counts"] == [2, 1]
    assert table["small_frequency_table"]["other_count"] == 2
    assert _summary(dxf, "arr", 0)["other_stats"]["num_unique"] == 4


def test_count_objects_unhashable_by_equality():
    # Equal dicts with different insertion order are one value
    assert _count_objects([{"a": 1, "b": 2}, {"b": 2, "a": 1}]) == ([{"a": 1, "b": 2}], [2])
    assert _count_objects([[1, 2], [1, 2], [2, 1]]) == ([[1, 2], [2, 1]], [2, 1])
    # A comparison that does not return a bool, or that raises, counts as not equal
    first, second = np.array([1, 2]), np.array([1, 2])
    assert _count_objects([first, second])[1] == [1, 1]  # == returns an array
    uniques, counts = _count_objects([[first], [second]])
    assert counts == [1, 1]
    assert uniques[0][0] is first
    assert uniques[1][0] is second


class _HashableRaisingEq:
    def __hash__(self) -> int:
        return 0

    def __eq__(self, other: object) -> bool:
        raise ValueError("no comparison")


def test_count_objects_equality_across_hashable_and_unhashable():
    # A set and a frozenset with the same items are equal, in either order
    assert _count_objects([{1}, frozenset({1})]) == ([{1}], [2])
    assert _count_objects([frozenset({1}), {1}]) == ([frozenset({1})], [2])
    values = [1, {"a": 1}, 1.0, frozenset({1}), {"a": 1}, True, {1}, [1], (1,)]
    assert _count_objects(values) == ([1, {"a": 1}, frozenset({1}), [1], (1,)], [3, 2, 2, 1, 1])
    # A hashable value whose == raises counts as not equal, also inside the hash table
    a, b = _HashableRaisingEq(), _HashableRaisingEq()
    assert _count_objects([a, b, {1}])[1] == [1, 1, 1]


def test_array_view_object_unhashable_equal_values(dxf: DataExplorerFixture):
    arr = np.empty(2, dtype=object)
    arr[0] = {"a": 1, "b": 2}
    arr[1] = {"b": 2, "a": 1}
    dxf.register_table("arr", arr)
    table = dxf.get_column_profiles("arr", [_get_frequency_table(0, 2)])[0]
    assert table["small_frequency_table"]["counts"] == [2]
    assert _summary(dxf, "arr", 0)["other_stats"]["num_unique"] == 1


#
# Datetime summaries without pandas
#


def test_array_view_datetime_summary_without_pandas(
    dxf: DataExplorerFixture, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    times = np.array(["2020-01-01", "2020-01-03"], dtype="datetime64[ns]")
    dxf.register_table("times", times)
    # A None entry in sys.modules makes `import pandas` raise ImportError
    monkeypatch.setitem(sys.modules, "pandas", None)
    with caplog.at_level(logging.WARNING):
        summary = _summary(dxf, "times", 0)
    assert summary["type_display"] == "datetime"
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


#
# xarray Variable
#


def test_array_view_variable_2d(dxf: DataExplorerFixture):
    ds = make_dataset()
    variable = ds.variables["sst"][0]  # dims (lat, lon), no coordinates
    assert get_inspector(variable).has_viewer()
    dxf.register_table("v", variable)
    state = dxf.get_state("v")
    assert state["display_name"] == "v (lat, lon)"
    # A Variable has no coordinates, so the grid shows its own row numbers.
    assert not state["has_row_labels"]
    assert [c["column_name"] for c in dxf.get_schema("v")] == ["0", "1", "2", "3", "4"]
    assert _column_labels(dxf, "v") == ["lon = 0", "lon = 1", "lon = 2", "lon = 3", "lon = 4"]
    assert _column_values(dxf, "v", 1, 3) == ["1.00", "6.00", "11.00"]


def test_array_view_index_variable(dxf: DataExplorerFixture):
    # Like any Variable, an IndexVariable has no name and no coordinate to label its rows.
    ds = make_dataset()
    dxf.register_table("t", ds.variables["time"])
    assert dxf.get_state("t")["display_name"] == "t (time)"
    assert not dxf.get_state("t")["has_row_labels"]
    assert [c["column_name"] for c in dxf.get_schema("t")] == ["0"]
    assert _column_values(dxf, "t", 0, 2) == ["2020-01-01 00:00:00", "2020-01-02 00:00:00"]


def test_variable_viewer_respects_size_limit():
    ds = make_dataset()
    assert not _value_type_is_supported(ds.variables["sst"])  # 3-D
    assert _value_type_is_supported(ds.variables["lat"])

    big = _lazily_indexed_dataarray((40000, 40000), "float32").variable
    assert not _value_type_is_supported(big)
    assert not get_inspector(big).has_viewer()
    with pytest.raises(TypeError):
        _ArrayTable.from_value(big)

    dask_array = pytest.importorskip("dask.array")
    xr = pytest.importorskip("xarray")
    assert not _value_type_is_supported(xr.Variable(("y", "x"), dask_array.zeros((6000, 6000))))
    assert _value_type_is_supported(xr.Variable(("y", "x"), dask_array.zeros((10, 10))))


def test_lazily_indexed_variable_under_limit_opens(dxf: DataExplorerFixture):
    values = np.arange(6.0).reshape(2, 3)
    variable = _lazily_indexed_dataarray((2, 3), values.dtype, values).variable
    dxf.register_table("lazy", variable)
    assert _column_values(dxf, "lazy", 1, 2) == ["1.00", "4.00"]


def test_array_view_unavailable_message_variable():
    ds = make_dataset()
    assert _array_view_unavailable_message(ds.variables["sst"], "v") == (
        "v has 3 dimensions; the Data Explorer shows 1-D and 2-D arrays. "
        "Select a slice first, e.g. v.isel(time=0)"
    )


#
# zarr arrays and h5py datasets
#


def _zarr_array(tmp_path, values: np.ndarray, dims: list[str] | None) -> Any:
    zarr = pytest.importorskip("zarr")
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    if hasattr(root, "create_array"):
        array = root.create_array("sst", shape=values.shape, dtype=values.dtype)
    else:
        array = root.create_dataset("sst", shape=values.shape, dtype=values.dtype)
    array[...] = values
    if dims is not None:
        # The attribute xarray writes, which zarr 2 and zarr 3 both read
        array.attrs["_ARRAY_DIMENSIONS"] = dims
    return array


def test_array_view_zarr_2d(dxf: DataExplorerFixture, tmp_path):
    array = _zarr_array(tmp_path, np.arange(12.0).reshape(3, 4), ["lat", "lon"])
    assert get_inspector(array).has_viewer()
    dxf.register_table("z", array)
    state = dxf.get_state("z")
    assert state["display_name"] == "z (lat, lon)"
    assert (state["row_axis_name"], state["column_axis_name"]) == ("lat", "lon")
    assert not state["has_row_labels"]
    assert [c["column_name"] for c in dxf.get_schema("z")] == ["0", "1", "2", "3"]
    assert _column_values(dxf, "z", 1, 3) == ["1.00", "5.00", "9.00"]


def test_array_view_zarr_without_dims(dxf: DataExplorerFixture, tmp_path):
    array = _zarr_array(tmp_path, np.arange(6.0).reshape(3, 2), None)
    dxf.register_table("z", array)
    assert dxf.get_state("z").get("row_axis_name") is None
    assert _column_values(dxf, "z", 1, 3) == ["1.00", "3.00", "5.00"]


def test_array_view_h5py(dxf: DataExplorerFixture, tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "data.h5", "w") as f:
        grid = f.create_dataset("grid", data=np.arange(12.0).reshape(3, 4))
        grid.dims[0].label = "lat"
        grid.dims[1].label = "lon"
        f.create_dataset("plain", data=np.arange(3))
        time = f.create_dataset("time", data=np.arange(3.0))
        time.make_scale("time")
    with h5py.File(tmp_path / "data.h5", "r") as f:
        assert get_inspector(f["grid"]).has_viewer()
        dxf.register_table("g", f["grid"])
        state = dxf.get_state("g")
        assert (state["row_axis_name"], state["column_axis_name"]) == ("lat", "lon")
        assert _column_values(dxf, "g", 2, 3) == ["2.00", "6.00", "10.00"]

        # Without dim names, the dataset opens like a numpy array.
        dxf.register_table("p", f["plain"])
        assert dxf.get_state("p").get("row_axis_name") is None
        assert _column_values(dxf, "p", 0, 3) == ["0", "1", "2"]
        # A dimension scale names its dim in its column, so the corner leaves the rows unnamed.
        dxf.register_table("t", f["time"])
        assert dxf.get_state("t").get("row_axis_name") is None
        assert [c["column_name"] for c in dxf.get_schema("t")] == ["time"]
        dataset = f["plain"]
    # A dataset of a closed file has no view.
    assert not _value_type_is_supported(dataset)


def test_stored_array_viewer_respects_size_limit(tmp_path):
    h5py = pytest.importorskip("h5py")
    with h5py.File(tmp_path / "big.h5", "w") as f:
        # Chunked datasets take no space until written.
        big = f.create_dataset("big", shape=(40000, 40000), dtype="f4", chunks=(1000, 1000))
        cube = f.create_dataset("cube", shape=(2, 3, 4), dtype="f4")
        assert not _value_type_is_supported(big)
        assert not get_inspector(big).has_viewer()
        assert _array_view_unavailable_message(big, "f['big']") == (
            "f['big'] is 6.0 GiB (not in memory), over the 100.0 MiB limit for the Data "
            "Explorer. Select a smaller slice first."
        )
        assert _array_view_unavailable_message(cube, "cube") == (
            "cube has 3 dimensions; the Data Explorer shows 1-D and 2-D arrays. "
            "Select a slice first, e.g. cube[0]"
        )


@pytest.mark.parametrize("names", [[], ["lat"], ["lat", "lat"]])
def test_array_view_zarr_malformed_dim_names(dxf: DataExplorerFixture, tmp_path, names):
    # Malformed dim names open the array like a numpy array.
    array = _zarr_array(tmp_path, np.arange(6.0).reshape(3, 2), names)
    dxf.register_table("z", array)
    assert dxf.get_state("z").get("row_axis_name") is None
    assert _column_values(dxf, "z", 1, 3) == ["1.00", "3.00", "5.00"]


#
# HyperSpy signals
#


def _spectrum_image() -> Any:
    hs = pytest.importorskip("hyperspy.api")
    s = hs.signals.Signal1D(np.arange(60.0).reshape(3, 4, 5))
    x, y = s.axes_manager.navigation_axes
    (energy,) = s.axes_manager.signal_axes
    x.name, y.name, energy.name = "x", "y", "Energy loss"
    energy.units = "eV"
    energy.scale = 0.5
    energy.offset = 100.0
    s.metadata.General.title = "EELS SI"
    return s


def test_array_view_hyperspy_line(dxf: DataExplorerFixture):
    # A line of a spectrum image: rows are the navigation positions, columns the energies.
    line = _spectrum_image().inav[0]
    dxf.register_table("line", line)
    state = dxf.get_state("line")
    assert (state["row_axis_name"], state["column_axis_name"]) == ("y", "Energy loss")
    # The y axis is uncalibrated, so the grid numbers the rows.
    assert not state["has_row_labels"]
    schema = dxf.get_schema("line")
    assert [c["column_name"] for c in schema] == ["100.0", "100.5", "101.0", "101.5", "102.0"]
    assert schema[1]["column_label"] == "Energy loss = 100.5 eV"
    assert _column_values(dxf, "line", 1, 3) == ["1.00", "21.00", "41.00"]


def test_array_view_hyperspy_spectrum(dxf: DataExplorerFixture):
    spectrum = _spectrum_image().inav[1, 0]
    dxf.register_table("sp", spectrum)
    state = dxf.get_state("sp")
    assert state["row_axis_name"] == "Energy loss"
    # The energies label the rows.
    assert state["has_row_labels"]
    assert [c["column_name"] for c in dxf.get_schema("sp")] == ["EELS SI"]
    assert _column_values(dxf, "sp", 0, 5) == ["5.00", "6.00", "7.00", "8.00", "9.00"]


def test_array_view_hyperspy_lazy_signal(dxf: DataExplorerFixture):
    line = _spectrum_image().inav[0].as_lazy()
    assert get_inspector(line).has_viewer()
    dxf.register_table("lazy", line)
    assert _column_values(dxf, "lazy", 0, 3) == ["0.00", "20.00", "40.00"]


def test_array_view_hyperspy_unavailable_message():
    s = _spectrum_image()
    assert _array_view_unavailable_message(s, "s") == (
        "s has 3 dimensions; the Data Explorer shows 1-D and 2-D arrays. "
        "Select a slice first, e.g. s.inav[0]"
    )
    hs = pytest.importorskip("hyperspy.api")
    stack = hs.signals.BaseSignal(np.zeros((2, 3, 4)))
    assert _array_view_unavailable_message(stack, "stack") == (
        "stack has 3 dimensions; the Data Explorer shows 1-D and 2-D arrays. "
        "Select a slice first, e.g. stack.isig[0]"
    )
    assert _array_view_unavailable_message(s.inav[0], "line") is None


def test_array_view_hyperspy_non_uniform_axis(dxf: DataExplorerFixture):
    hs = pytest.importorskip("hyperspy.api")
    energy = {"axis": np.array([1.0, 2.0, 4.0]), "name": "Energy", "units": "eV"}
    position = {"size": 2, "name": "x", "navigate": True}
    s = hs.signals.Signal1D(np.arange(6.0).reshape(2, 3), axes=[position, energy])
    dxf.register_table("s", s)
    # The non-uniform axis labels the columns with its values, even though it starts near 0
    assert [c["column_name"] for c in dxf.get_schema("s")] == ["1.0", "2.0", "4.0"]
    assert dxf.get_state("s")["column_axis_name"] == "Energy"

    spectrum = hs.signals.Signal1D(np.arange(3.0), axes=[energy])
    dxf.register_table("sp", spectrum)
    assert dxf.get_state("sp")["has_row_labels"]
