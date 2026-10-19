#
# Copyright (C) 2026 Posit Software, PBC. All rights reserved.
# Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
#

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from positron.inspectors import (
    ArrayGroup,
    ArrayGroupInspector,
    MetadataRow,
    MetadataRowInspector,
    XarrayDataArrayInspector,
    XarrayDatasetInspector,
    _format_name_list,
    _format_nbytes,
    get_inspector,
)

if TYPE_CHECKING:
    import xarray as xr
else:
    xr = pytest.importorskip("xarray")


def make_dataset(time: int = 4, lat: int = 3, lon: int = 5) -> xr.Dataset:
    shape = (time, lat, lon)
    size = time * lat * lon
    coords = {
        "time": pd.date_range("2020-01-01", periods=time, freq="D"),
        "lat": np.linspace(-10, 10, lat),
        "lon": np.linspace(0, 40, lon),
    }
    dims = ("time", "lat", "lon")
    return xr.Dataset(
        {
            "sst": (dims, np.arange(size, dtype="float32").reshape(shape)),
            "anom": (dims, np.ones(shape, dtype="float32")),
        },
        coords=coords,
        attrs={"Conventions": "CF-1.5", "title": "NOAA OISST v2.1", "source": "synthetic"},
    )


def test_inspect_xarray_dataset():
    ds = make_dataset()
    ins = get_inspector(ds)
    assert type(ins) is XarrayDatasetInspector
    assert ins.get_display_value() == ("[3 dims x 2 vars] xarray.Dataset", True)
    assert ins.get_display_type() == "Dataset [3 dims x 2 vars]"
    assert ins.get_kind() == "table"
    assert ins.get_type_info() == "xarray.core.dataset.Dataset"
    assert ins.get_size() == ds.nbytes
    assert ins.get_length() == 2
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
    assert ins.get_length() == 4
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
    assert ins.get_display_type() == "DataArray [] float32"


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
    assert not ins.equals(pd.DataFrame({"a": [1]}))  # pyright: ignore[reportArgumentType]

    da = make_dataset()["sst"]
    ins = get_inspector(da)
    copy = ins.deepcopy()
    assert ins.equals(copy)
    copy.attrs["title"] = "x"
    assert not ins.equals(copy)
    copy = ins.deepcopy()
    copy[0, 0, 0] = 1
    assert not ins.equals(copy)
    assert not ins.equals(pd.DataFrame({"a": [1]}))  # pyright: ignore[reportArgumentType]


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
