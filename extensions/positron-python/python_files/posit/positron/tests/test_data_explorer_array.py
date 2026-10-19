#
# Copyright (C) 2026 Posit Software, PBC. All rights reserved.
# Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
#

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
import pandas as pd
import pytest

from ..data_explorer import (
    _VALUE_INF,
    _VALUE_NA,
    _VALUE_NAN,
    _VALUE_NAT,
    _VALUE_NEGINF,
    ARRAY_VIEW_MAX_DASK_BYTES,
    ArrayView,
    DataExplorerService,
    _value_type_is_supported,
)
from ..data_explorer_comm import SupportStatus
from ..utils import guid
from . import test_data_explorer
from .test_data_explorer import (
    _get_frequency_table,
    _get_histogram,
    _get_null_count,
    _profile_request,
)
from .test_inspectors_xarray import make_dataset
from .utils import json_rpc_request

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
    schema = dxf.get_schema("sst")
    assert [c["column_name"] for c in schema] == [str(v) for v in da.lon.values]
    assert schema[0]["type_display"] == "floating"
    num_rows = da.sizes["lat"]
    assert _row_labels(dxf, "sst", num_rows) == [str(v) for v in da.lat.values]
    assert _column_values(dxf, "sst", 1, num_rows) == ["1.00", "6.00", "11.00"]


def test_array_view_dataarray_2d_without_coordinates(dxf: DataExplorerFixture):
    xr = pytest.importorskip("xarray")
    da = xr.DataArray(np.zeros((2, 3)), dims=("y", "x"), name="z")
    dxf.register_table("z", da)
    assert [c["column_name"] for c in dxf.get_schema("z")] == ["0", "1", "2"]
    assert _row_labels(dxf, "z", 2) == ["0", "1"]


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
    assert [c["column_name"] for c in dxf.get_schema("lat")] == ["lat"]
    assert _row_labels(dxf, "lat", 3) == [str(v) for v in lat.values]


def test_array_view_datetime_1d(dxf: DataExplorerFixture):
    da = make_dataset()["time"]
    dxf.register_table("time", da)
    schema = dxf.get_schema("time")
    assert schema[0]["type_display"] == "datetime"
    values = _column_values(dxf, "time", 0, da.sizes["time"])
    series = pd.Series(da.values)
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
    assert big.nbytes > ARRAY_VIEW_MAX_DASK_BYTES
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
