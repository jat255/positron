#
# Copyright (C) 2026 Posit Software, PBC. All rights reserved.
# Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
#

from __future__ import annotations

import copy
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import pytest

from positron.inspectors import (
    ArrayGroup,
    ArrayGroupInspector,
    DaskArrayInspector,
    MetadataRow,
    MetadataRowInspector,
    XarrayDataArrayInspector,
    XarrayDatasetInspector,
    _format_name_list,
    _format_nbytes,
    _is_dask_backed,
    get_inspector,
)

from .utils import make_dataset, make_lazy_dataset

if TYPE_CHECKING:
    import xarray as xr
else:
    xr = pytest.importorskip("xarray")

BackendArray = pytest.importorskip("xarray.backends").BackendArray


def test_inspect_xarray_dataset():
    ds = make_dataset()
    ins = get_inspector(ds)
    assert type(ins) is XarrayDatasetInspector
    assert ins.get_display_value() == ("[3 dims x 2 vars] xarray.Dataset", True)
    assert ins.get_display_type() == "Dataset [3 dims x 2 vars]"
    assert ins.get_kind() == "table"
    assert ins.get_type_info() == "xarray.core.dataset.Dataset"
    assert ins.get_size() == ds.nbytes
    assert ins.get_length() == 5
    assert ins.has_children()
    assert not ins.has_viewer()
    assert list(ins.get_children()) == [
        "Dimensions",
        "Coordinates",
        "Data variables",
        "Indexes",
        "Attributes",
    ]
    assert type(ins.get_child("Dimensions")) is MetadataRow
    assert type(ins.get_child("Indexes")) is MetadataRow
    for key in ("Coordinates", "Data variables", "Attributes"):
        assert type(ins.get_child(key)) is ArrayGroup
    assert ins.has_child("Attributes")
    assert not ins.has_child("Nope")


def test_inspect_xarray_dataset_groups():
    ds = make_dataset()
    ins = get_inspector(ds)
    rows = {k: get_inspector(ins.get_child(k)) for k in ins.get_children()}
    assert rows["Dimensions"].get_display_value() == ("time: 4, lat: 3, lon: 5", False)
    assert not rows["Dimensions"].has_children()
    assert rows["Coordinates"].get_display_value()[0] == "time, lat, lon"
    assert rows["Data variables"].get_display_value()[0] == "sst, anom"
    assert rows["Indexes"].get_display_value()[0] == "time, lat, lon"
    assert rows["Attributes"].get_display_value()[0] == "Conventions, title, source"
    assert list(rows["Data variables"].get_children()) == ["sst", "anom"]
    assert rows["Data variables"].get_child("sst").identical(ds["sst"])
    assert rows["Attributes"].get_child("title") == "NOAA OISST v2.1"
    assert rows["Data variables"].has_child("anom")
    assert not rows["Data variables"].has_child("nope")


def test_inspect_xarray_empty_groups():
    ds = xr.Dataset({"a": (("x",), np.arange(3))})
    ins = get_inspector(ds)
    for key in ("Coordinates", "Attributes", "Indexes"):
        row = get_inspector(ins.get_child(key))
        assert row.get_display_value()[0] == "(none)"
        assert not row.has_children()


def test_metadata_row_inspector():
    row = MetadataRow("time: 4")
    ins = get_inspector(row)
    assert type(ins) is MetadataRowInspector
    assert ins.get_display_value() == ("time: 4", False)
    assert ins.get_display_type() == ""
    assert ins.get_kind() == "other"
    assert not ins.has_children()
    assert not ins.is_mutable()


def test_array_group_inspector_type():
    ds = make_dataset()
    ins = get_inspector(ArrayGroup("Attributes", ds))
    assert type(ins) is ArrayGroupInspector
    assert ins.get_kind() == "other"


def test_inspect_xarray_dataarray():
    da = make_dataset()["sst"]
    ins = get_inspector(da)
    assert type(ins) is XarrayDataArrayInspector
    assert ins.get_display_value() == ("[time: 4 x lat: 3 x lon: 5] xarray.DataArray", True)
    assert ins.get_display_type() == "DataArray [4x3x5] float32"
    assert ins.get_kind() == "table"
    assert ins.get_size() == da.nbytes
    assert ins.get_length() == 3
    assert ins.has_children()
    assert list(ins.get_children()) == ["Coordinates", "Attributes", "Values"]
    assert not ins.has_viewer()  # piece 2 changes this for 1-D/2-D
    values = get_inspector(ins.get_child("Values"))
    assert "0." in values.get_display_value()[0]


def test_inspect_xarray_dataarray_0d():
    da = make_dataset()["sst"].mean()
    ins = get_inspector(da)
    assert ins.get_display_value()[0] == "[scalar] xarray.DataArray"
    assert ins.get_length() == 0
    assert not ins.has_viewer()
    assert ins.get_display_type() == "DataArray [scalar] float32"


def test_inspect_xarray_dataarray_0d_no_children():
    da = xr.DataArray(3.0)
    ins = get_inspector(da)
    assert not ins.has_children()


def test_xarray_equals_detects_changes():
    ds = make_dataset()
    ins = get_inspector(ds)
    copy = ins.deepcopy()
    assert ins.equals(copy)
    copy.attrs["title"] = "x"
    assert not ins.equals(copy)
    copy = ins.deepcopy()
    copy["sst"][0, 0, 0] = 1
    assert not ins.equals(copy)
    assert not ins.equals(pd.DataFrame({"a": [1]}))  # type: ignore

    da = make_dataset()["sst"]
    ins = get_inspector(da)
    copy = ins.deepcopy()
    assert ins.equals(copy)
    copy.attrs["title"] = "x"
    assert not ins.equals(copy)
    copy = ins.deepcopy()
    copy[0, 0, 0] = 1
    assert not ins.equals(copy)
    assert not ins.equals(pd.DataFrame({"a": [1]}))  # type: ignore


def test_xarray_comparison_cost():
    ds = make_dataset()
    assert get_inspector(ds).get_comparison_cost() == ds.nbytes
    assert get_inspector(ds["sst"]).get_comparison_cost() == ds["sst"].nbytes


def test_format_nbytes():
    assert _format_nbytes(512) == "512 B"
    assert _format_nbytes(1536) == "1.5 KiB"
    assert _format_nbytes(94_582_784) == "90.2 MiB"


def test_format_name_list():
    assert _format_name_list(["a", 1, "c"]) == ("a, 1, c", False)
    names = [f"name_{i}" for i in range(20000)]
    text, truncated = _format_name_list(names)
    assert truncated
    assert len(text) < len(", ".join(names))


class _CountingBackendArray(BackendArray):
    """A lazily indexed backend array that counts how many elements are read."""

    def __init__(self, array: np.ndarray) -> None:
        self.array = array
        self.shape = array.shape
        self.dtype = array.dtype
        self.elements_read = 0

    def __getitem__(self, key: Any) -> np.ndarray:
        from xarray.core import indexing

        return indexing.explicit_indexing_adapter(
            key, self.shape, indexing.IndexingSupport.BASIC, self._raw_getitem
        )

    def _raw_getitem(self, key: tuple) -> np.ndarray:
        result = self.array[key]
        self.elements_read += result.size
        return result


def make_lazily_indexed_dataarray(shape: tuple[int, ...]) -> tuple[xr.DataArray, Any]:
    from xarray.core import indexing

    backend = _CountingBackendArray(np.arange(np.prod(shape), dtype="float64").reshape(shape))
    dims = tuple(f"d{i}" for i in range(len(shape)))
    variable = xr.Variable(dims, indexing.LazilyIndexedArray(backend))
    return xr.DataArray(variable), backend


def test_values_preview_reads_only_edge_items():
    shape = (200, 300, 400)
    da, backend = make_lazily_indexed_dataarray(shape)
    preview = get_inspector(get_inspector(da).get_child("Values")).get_display_value()[0]
    # At most 7 items per dim are read: 3 shown at each edge, plus one hidden by the ellipsis.
    assert 0 < backend.elements_read <= 7**3
    # The preview matches numpy's own summary of the full array.
    expected = np.array2string(backend.array, threshold=100, edgeitems=3)
    assert preview == " ".join(expected.split())


def test_values_preview_small_array_unchanged():
    da = xr.DataArray(np.arange(10.0), dims=("x",))
    preview = get_inspector(get_inspector(da).get_child("Values")).get_display_value()[0]
    assert preview == "[0. 1. 2. 3. 4. 5. 6. 7. 8. 9.]"


def test_values_preview_mixed_dim_lengths():
    shape = (2, 50, 4)
    da, backend = make_lazily_indexed_dataarray(shape)
    preview = get_inspector(get_inspector(da).get_child("Values")).get_display_value()[0]
    assert backend.elements_read <= 2 * 7 * 4
    expected = np.array2string(backend.array, threshold=100, edgeitems=3)
    assert preview == " ".join(expected.split())


#
# Dask-backed objects
#


def test_is_dask_backed():
    ds = make_lazy_dataset()
    assert _is_dask_backed(ds)
    assert _is_dask_backed(ds["sst"])
    assert _is_dask_backed(ds["sst"].data)
    assert not _is_dask_backed(make_dataset())
    assert not _is_dask_backed(make_dataset()["sst"])
    assert not _is_dask_backed(np.ones(3))
    assert not _is_dask_backed(ds["time"])


def test_inspect_dask_backed_dataset():
    ins = get_inspector(make_lazy_dataset())
    assert ins.get_display_value() == ("[3 dims x 2 vars] xarray.Dataset (dask)", True)
    assert list(ins.get_children())[-1] == "Dask metadata"
    group = get_inspector(ins.get_child("Dask metadata"))
    assert group.get_display_value()[0] == "2 chunks, 1.4 KiB (lazy)"  # 12*3*5*4*2 bytes
    rows = {
        k: get_inspector(group.get_child(k)).get_display_value()[0] for k in group.get_children()
    }
    assert list(rows) == ["chunks", "nbytes", "graph layers"]
    assert rows["chunks"] == "time: 6, lat: 3, lon: 5"
    assert rows["nbytes"] == "1.4 KiB"
    assert rows["graph layers"] == "4"  # from_array and map_blocks for each variable


def _walk(value: Any, depth: int = 0) -> int:
    """Call every inspector method used by the Variables pane on value and its descendants."""
    ins = get_inspector(value)
    ins.get_display_value()
    ins.get_display_type()
    ins.get_kind()
    ins.get_size()
    ins.get_length()
    ins.has_viewer()
    ins.get_comparison_cost()
    try:
        snapshot = ins.deepcopy()
    except copy.Error:
        pass
    else:
        assert ins.equals(snapshot)
    visited = 1
    if ins.has_children() and depth < 6:
        for key in ins.get_children():
            assert ins.has_child(key)
            visited += _walk(ins.get_child(key), depth + 1)
    return visited


def test_dask_backed_inspection_never_computes():
    ds = make_lazy_dataset()
    # The bomb makes any computation fail the test.
    assert _walk(ds) > 20
    assert _walk(ds["sst"]) > 5
    assert _walk(ds["sst"].data) > 3


def test_inspect_dask_dataarray():
    ins = get_inspector(make_lazy_dataset()["sst"])
    assert ins.get_display_value() == ("[time: 12 x lat: 3 x lon: 5] xarray.DataArray (dask)", True)
    assert list(ins.get_children()) == ["Coordinates", "Attributes", "Dask metadata"]
    group = get_inspector(ins.get_child("Dask metadata"))
    assert group.get_display_value()[0] == "2 chunks, 720 B (lazy)"
    chunks = get_inspector(group.get_child("chunks")).get_display_value()[0]
    assert chunks == "time: 6, lat: 3, lon: 5"


def test_inspect_bare_dask_array():
    arr = make_lazy_dataset()["sst"].data
    ins = get_inspector(arr)
    assert type(ins) is DaskArrayInspector
    assert ins.get_kind() == "table"
    assert ins.get_display_type() == "Array [12x3x5] float32"
    assert ins.get_length() == 1
    assert ins.get_size() == 720
    assert list(ins.get_children()) == ["Dask metadata"]
    group = get_inspector(ins.get_child("Dask metadata"))
    assert group.get_display_value()[0] == "2 chunks, 720 B (lazy)"
    chunks = get_inspector(group.get_child("chunks")).get_display_value()[0]
    assert chunks == "(6, 3, 5)"
    layers = get_inspector(group.get_child("graph layers")).get_display_value()[0]
    assert layers == "2"


def test_dask_mixed_chunks():
    ds = make_lazy_dataset()
    ds["anom"] = ds["anom"].chunk({"time": 4})
    group = get_inspector(get_inspector(ds).get_child("Dask metadata"))
    assert group.get_display_value()[0] == "mixed chunks, 1.4 KiB (lazy)"
    chunks = get_inspector(group.get_child("chunks")).get_display_value()[0]
    assert chunks == "mixed (varies by variable)"


def test_dask_equals_detects_attr_change():
    ds = make_lazy_dataset()
    ins = get_inspector(ds)
    snapshot = ins.deepcopy()
    assert ins.equals(snapshot)
    snapshot.attrs["title"] = "x"
    assert not ins.equals(snapshot)
    snapshot = ins.deepcopy()
    snapshot["sst"].attrs["units"] = "K"
    assert not ins.equals(snapshot)
    snapshot = ins.deepcopy()
    snapshot["sst"] = snapshot["sst"] + 1
    assert not ins.equals(snapshot)
    # Eager in both directions: comparing never computes the dask side.
    assert not ins.equals(make_dataset(time=12))
    assert not get_inspector(make_dataset(time=12)).equals(ds)

    da = ds["sst"]
    ins = get_inspector(da)
    snapshot = ins.deepcopy()
    assert ins.equals(snapshot)
    snapshot.attrs["title"] = "x"
    assert not ins.equals(snapshot)
    assert not get_inspector(make_dataset(time=12)["sst"]).equals(da)

    arr = da.data
    ins = get_inspector(arr)
    snapshot = ins.deepcopy()
    assert ins.equals(snapshot)
    snapshot[0, 0, 0] = 1
    assert not ins.equals(snapshot)


def test_dask_comparison_cost_counts_eager_coordinates():
    ds = make_lazy_dataset()
    coord_bytes = sum(ds[name].nbytes for name in ("time", "lat", "lon"))
    assert get_inspector(ds).get_comparison_cost() == coord_bytes
    assert get_inspector(ds["sst"]).get_comparison_cost() == coord_bytes


def test_dask_equals_detects_in_place_edit_of_original():
    # The snapshot must not share dask Array objects with the original: dask's __setitem__
    # gives the array it is called on a new name, which a shared object would pass on to the
    # snapshot too.
    ds = make_lazy_dataset()
    snapshot = get_inspector(ds).deepcopy()
    ds["sst"][0, 0, 0] = 1
    assert not get_inspector(snapshot).equals(ds)

    da = make_lazy_dataset()["sst"]
    snapshot = get_inspector(da).deepcopy()
    da[0, 0, 0] = 1
    assert not get_inspector(snapshot).equals(da)


def test_indexes_row_truncates_long_name_lists():
    names = [f"dimension_{i:04d}" for i in range(5000)]
    ds = xr.Dataset(coords={name: [0] for name in names})
    text = get_inspector(get_inspector(ds).get_child("Indexes")).get_display_value()[0]
    assert text == _format_name_list(names)[0]
    assert len(text) < len(", ".join(names))


def make_mixed_lazy_dataset() -> xr.Dataset:
    """A dask-backed Dataset that also has an eager data variable and an eager coordinate."""
    ds = make_lazy_dataset()
    ds["mask"] = ("time", np.zeros(12))
    ds.coords["depth"] = ("lat", np.zeros(3))
    return ds


def test_dask_equals_detects_replaced_eager_data_variable():
    ds = make_mixed_lazy_dataset()
    snapshot = get_inspector(ds).deepcopy()
    assert get_inspector(snapshot).equals(ds)
    ds["mask"] = ("time", np.ones(12))
    assert not get_inspector(snapshot).equals(ds)


def test_dask_equals_detects_in_place_edit_of_eager_data_variable():
    ds = make_mixed_lazy_dataset()
    snapshot = get_inspector(ds).deepcopy()
    ds["mask"][0] = 5
    assert not get_inspector(snapshot).equals(ds)


def test_dask_equals_detects_in_place_edit_of_eager_coordinate():
    ds = make_mixed_lazy_dataset()
    snapshot = get_inspector(ds).deepcopy()
    ds["depth"][0] = 5
    assert not get_inspector(snapshot).equals(ds)

    da = ds["sst"]
    snapshot = get_inspector(da).deepcopy()
    da["depth"][0] = 7
    assert not get_inspector(snapshot).equals(da)


def test_dask_comparison_cost_counts_eager_data_variables():
    ds = make_mixed_lazy_dataset()
    eager = ("time", "lat", "lon", "depth", "mask")
    assert get_inspector(ds).get_comparison_cost() == sum(ds[name].nbytes for name in eager)


def test_dask_equals_detects_in_place_edit_of_eager_variable_named_none():
    # None is a valid Dataset variable name; it must still get its own copy in the snapshot.
    # make_lazy_dataset's dask variables fail the test if anything computes them.
    ds = make_lazy_dataset()
    ds[None] = ("time", np.zeros(12))
    snapshot = get_inspector(ds).deepcopy()
    assert get_inspector(snapshot).equals(ds)
    ds[None][0] = 5
    assert not get_inspector(snapshot).equals(ds)


#
# Child counts and self-referencing coordinates
#

_ARRAY_INSPECTORS = (
    XarrayDatasetInspector,
    XarrayDataArrayInspector,
    DaskArrayInspector,
    ArrayGroupInspector,
    MetadataRowInspector,
)


def _assert_lengths_match_children(value: Any, depth: int = 0) -> int:
    """
    Check that every array inspector under value reports its child count as its length.

    The Variables pane adds a "more values" row when a variable's length is larger than the
    number of children it receives.
    """
    ins = get_inspector(value)
    if type(ins) not in _ARRAY_INSPECTORS:
        return 0
    children = list(ins.get_children())
    assert ins.get_length() == len(children), (type(ins).__name__, children)
    assert ins.has_children() == (len(children) > 0)
    checked = 1
    if depth < 6:
        for key in children:
            checked += _assert_lengths_match_children(ins.get_child(key), depth + 1)
    return checked


@pytest.mark.parametrize(
    "make_value",
    [
        pytest.param(make_dataset, id="dataset"),
        pytest.param(make_lazy_dataset, id="dask-dataset"),
        pytest.param(lambda: make_dataset()["sst"], id="dataarray"),
        pytest.param(lambda: make_lazy_dataset()["sst"], id="dask-dataarray"),
        pytest.param(lambda: make_dataset()["time"], id="dimension-coordinate"),
        pytest.param(lambda: make_dataset()["sst"].isel(time=0), id="2d-slice"),
        pytest.param(lambda: make_dataset()["sst"].isel(time=0)["time"], id="scalar-coordinate"),
        pytest.param(lambda: make_dataset()["sst"].mean(), id="0d"),
        pytest.param(lambda: xr.DataArray(3.0), id="0d-bare"),
        pytest.param(lambda: make_lazy_dataset()["sst"].data, id="dask-array"),
        pytest.param(lambda: MetadataRow("time: 4"), id="metadata-row"),
    ],
)
def test_length_is_child_count(make_value: Any) -> None:
    assert _assert_lengths_match_children(make_value()) > 0


def test_length_is_child_count_for_groups() -> None:
    ds = make_lazy_dataset()
    for owner in (ds, ds["sst"]):
        for name in ("Coordinates", "Data variables", "Attributes", "Values", "Dask metadata"):
            if name == "Data variables" and owner is not ds:
                continue
            ins = get_inspector(ArrayGroup(name, owner))
            assert ins.get_length() == len(list(ins.get_children()))
    assert get_inspector(ArrayGroup("Values", make_dataset()["sst"])).get_length() == 0
    assert get_inspector(ArrayGroup("Dask metadata", ds["sst"].data)).get_length() == 3


def test_dimension_coordinate_does_not_list_itself():
    da = make_dataset()["time"]
    ins = get_inspector(da)
    assert list(ins.get_children()) == ["Attributes", "Values"]
    assert not ins.has_child("Coordinates")


def test_scalar_coordinate_does_not_list_itself():
    da = make_dataset()["sst"].isel(time=0)["time"]
    assert list(get_inspector(da).get_children()) == ["Attributes", "Values"]


def test_slice_keeps_other_coordinates():
    ins = get_inspector(make_dataset()["sst"].isel(time=0))
    assert list(ins.get_children()) == ["Coordinates", "Attributes", "Values"]
    coords = get_inspector(ins.get_child("Coordinates"))
    assert sorted(coords.get_children()) == ["lat", "lon", "time"]
    assert coords.get_length() == 3


def test_coordinate_with_other_coordinates_omits_itself():
    ds = make_dataset()
    ds.coords["doy"] = ("time", np.arange(4))
    for name, other in (("time", "doy"), ("doy", "time")):
        ins = get_inspector(ds[name])
        assert list(ins.get_children()) == ["Coordinates", "Attributes", "Values"]
        coords = get_inspector(ins.get_child("Coordinates"))
        assert list(coords.get_children()) == [other]
        assert coords.get_display_value()[0] == other
        assert not coords.has_child(name)
        assert _assert_lengths_match_children(ds[name]) > 0


def test_dataarray_without_coordinates_keeps_empty_group():
    da = xr.DataArray(np.arange(3), dims=("x",), name="x")
    ins = get_inspector(da)
    assert list(ins.get_children()) == ["Coordinates", "Attributes", "Values"]
    assert get_inspector(ins.get_child("Coordinates")).get_display_value()[0] == "(none)"


def test_same_named_distinct_coordinate_is_listed():
    # A coordinate can share the DataArray's name without being the DataArray itself.
    da = xr.DataArray([1, 2], dims="x", coords={"x": [10, 20]}, name="x")
    ins = get_inspector(da)
    assert list(ins.get_children()) == ["Coordinates", "Attributes", "Values"]
    coords = get_inspector(ins.get_child("Coordinates"))
    assert list(coords.get_children()) == ["x"]
    assert coords.has_child("x")
    assert coords.get_child("x").to_numpy().tolist() == [10, 20]
    assert _assert_lengths_match_children(da) > 0


def test_same_named_coordinate_sharing_data_is_listed():
    # A non-dimension coordinate that shares the DataArray's ndarray, with its own attrs, is
    # still a separate coordinate.
    base = xr.DataArray(np.arange(3.0), dims="y", name="x")
    coord = base.variable.copy(deep=False)
    coord.attrs["units"] = "m"
    da = base.assign_coords(x=coord)
    assert da.coords["x"].variable._data is da.variable._data  # noqa: SLF001
    ins = get_inspector(da)
    assert list(ins.get_children()) == ["Coordinates", "Attributes", "Values"]
    coords = get_inspector(ins.get_child("Coordinates"))
    assert list(coords.get_children()) == ["x"]
    assert coords.get_child("x").attrs == {"units": "m"}
    assert _assert_lengths_match_children(da) > 0


def test_sliced_dimension_coordinate_nesting_ends():
    # Slicing gives the DataArray and its coordinate separate variables, so the coordinate is
    # listed once; the coordinate's own DataArray then hides it.
    da = make_dataset()["time"].isel(time=slice(0, 2))
    coords = get_inspector(get_inspector(da).get_child("Coordinates"))
    assert list(coords.get_children()) == ["time"]
    inner = get_inspector(coords.get_child("time"))
    assert list(inner.get_children()) == ["Attributes", "Values"]


def test_dask_backed_coordinate_does_not_list_itself():
    from .utils import make_compute_bomb

    ds = make_lazy_dataset()
    # The bomb makes any computation fail the test.
    ds.coords["depth"] = (("time", "lat"), make_compute_bomb(np.zeros((12, 3)), chunks=(6, 3)))
    da = ds["depth"]
    ins = get_inspector(da)
    assert list(ins.get_children()) == ["Coordinates", "Attributes", "Dask metadata"]
    assert sorted(get_inspector(ins.get_child("Coordinates")).get_children()) == ["lat", "time"]
    assert _assert_lengths_match_children(da) > 0
    assert _walk(da) > 5


#
# Datetime values preview
#


def _values_preview(da: xr.DataArray) -> str:
    return get_inspector(get_inspector(da).get_child("Values")).get_display_value()[0]


def test_values_preview_datetime_uses_pandas_style():
    preview = _values_preview(make_dataset()["time"])
    assert preview == (
        "[2020-01-01 00:00:00, 2020-01-02 00:00:00, 2020-01-03 00:00:00, 2020-01-04 00:00:00]"
    )
    assert "T00:00:00.000000000" not in preview
    assert "'" not in preview


def test_values_preview_datetime_summarizes_long_arrays():
    preview = _values_preview(make_dataset(time=366)["time"])
    assert preview == (
        "[2020-01-01 00:00:00, 2020-01-02 00:00:00, 2020-01-03 00:00:00, ...,"
        " 2020-12-29 00:00:00, 2020-12-30 00:00:00, 2020-12-31 00:00:00]"
    )


def test_values_preview_datetime_nat_and_timedelta():
    times = np.array(["2020-01-01T06:30", "NaT"], dtype="datetime64[ns]")
    assert _values_preview(xr.DataArray(times, dims=("t",))) == "[2020-01-01 06:30:00, NaT]"
    deltas = np.array([1, 2], dtype="timedelta64[D]").astype("timedelta64[ns]")
    preview = _values_preview(xr.DataArray(deltas, dims=("t",)))
    assert preview == "[1 days 00:00:00, 2 days 00:00:00]"


def test_values_preview_datetime_without_pandas(monkeypatch: pytest.MonkeyPatch):
    import sys

    da = make_dataset()["time"]
    monkeypatch.setitem(sys.modules, "pandas", None)  # type: ignore
    preview = _values_preview(da)
    assert preview == " ".join(np.array2string(da.to_numpy(), threshold=100, edgeitems=3).split())


def test_values_preview_datetime_reads_only_edge_items():
    from xarray.core import indexing

    times = np.arange("2020-01-01", "2021-01-01", dtype="datetime64[D]").astype("datetime64[ns]")
    backend = _CountingBackendArray(times)
    da = xr.DataArray(xr.Variable(("time",), indexing.LazilyIndexedArray(backend)))
    preview = _values_preview(da)
    assert backend.elements_read <= 7
    assert preview.startswith("[2020-01-01 00:00:00, ")
    assert preview.endswith(" 2020-12-31 00:00:00]")
