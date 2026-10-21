#
# Copyright (C) 2026 Posit Software, PBC. All rights reserved.
# Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
#

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from positron.access_keys import encode_access_key
from positron.inspectors import (
    ArrayGroup,
    H5pyDatasetInspector,
    H5pyGroupInspector,
    MetadataRow,
    ZarrArrayInspector,
    ZarrGroupInspector,
    get_inspector,
)
from positron.variables import _resolve_value_from_path, _summarize_children

if TYPE_CHECKING:
    from pathlib import Path

    import h5py
    import zarr
else:
    zarr = pytest.importorskip("zarr")
    h5py = pytest.importorskip("h5py")

ZARR_3 = int(zarr.__version__.split(".")[0]) >= 3


def _walk(value: Any, depth: int = 0) -> int:
    """
    Summarize every row below value, as the Variables pane would, and count them.

    Checks that every row's length is its child count, so the pane adds no "more values" row.
    """
    ins = get_inspector(value)
    ins.get_display_value()
    ins.get_display_type()
    ins.get_size()
    if depth > 6 or not ins.has_children():
        return 1
    children = list(ins.get_children())
    assert ins.get_length() == len(children)
    return 1 + sum(_walk(ins.get_child(key), depth + 1) for key in children)


#
# zarr
#


def _create_array(group: Any, name: str, **kwargs: Any) -> Any:
    if ZARR_3:
        return group.create_array(name, **kwargs)
    kwargs.pop("dimension_names", None)
    return group.create_dataset(name, **kwargs)


@pytest.fixture
def zarr_group(tmp_path: Path) -> Any:
    root = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    root.attrs["title"] = "sea surface temperature"
    sst = _create_array(
        root,
        "sst",
        shape=(10, 4, 5),
        chunks=(6, 4, 5),
        dtype="f4",
        dimension_names=("time", "lat", "lon"),
    )
    sst[:] = np.arange(200, dtype="f4").reshape(10, 4, 5)
    sst.attrs["units"] = "degC"
    root.create_group("masks")
    return root


def test_inspect_zarr_group(zarr_group):
    ins = get_inspector(zarr_group)
    assert type(ins) is ZarrGroupInspector
    assert ins.get_display_value() == (
        f"[2 members] zarr.{type(zarr_group).__name__}",
        True,
    )
    assert ins.get_display_type() == f"{type(zarr_group).__name__} [2 members]"
    assert ins.get_kind() == "table"
    assert not ins.is_mutable()
    children = list(ins.get_children())
    assert children[0] == 0
    assert sorted(children[1:]) == ["masks", "sst"]
    assert [ins.get_display_name(key) for key in children[:1]] == ["Attributes"]
    assert ins.get_length() == 3

    attributes = ins.get_child(0)
    assert isinstance(attributes, ArrayGroup)
    assert get_inspector(attributes).get_display_value() == ("title", False)
    assert type(get_inspector(ins.get_child("masks"))) is ZarrGroupInspector
    assert type(get_inspector(ins.get_child("sst"))) is ZarrArrayInspector
    with pytest.raises(KeyError):
        ins.get_child("missing")


def test_zarr_group_member_named_attributes(zarr_group):
    # The Attributes row has an int key, so a member named "Attributes" stays reachable.
    zarr_group.create_group("Attributes")
    ins = get_inspector(zarr_group)
    assert type(get_inspector(ins.get_child("Attributes"))) is ZarrGroupInspector
    assert isinstance(ins.get_child(0), ArrayGroup)


def test_inspect_zarr_array(zarr_group):
    array = zarr_group["sst"]
    ins = get_inspector(array)
    assert type(ins) is ZarrArrayInspector
    dims = "time: 10 x lat: 4 x lon: 5" if ZARR_3 else "10 x 4 x 5"
    assert ins.get_display_value() == (f"[{dims}] zarr.Array", True)
    assert ins.get_display_type() == "Array [10x4x5] float32"
    assert ins.get_kind() == "table"
    assert ins.get_size() == 800
    assert not ins.is_mutable()
    assert list(ins.get_children()) == ["Attributes", "Chunks", "Compression", "Store"]
    assert get_inspector(ins.get_child("Attributes")).get_display_value() == ("units", False)

    chunks = ins.get_child("Chunks")
    assert isinstance(chunks, MetadataRow)
    assert chunks.text == "(6, 4, 5), 2 chunks"
    assert ins.get_child("Compression").text != ""
    assert ins.get_child("Store").text.endswith("store.zarr/sst")


def test_zarr_array_dims_from_xarray_attribute(zarr_group):
    # xarray names the dims of a zarr format 2 array in the _ARRAY_DIMENSIONS attribute.
    array = _create_array(zarr_group, "anom", shape=(3, 2), dtype="f4")
    array.attrs["_ARRAY_DIMENSIONS"] = ["lat", "lon"]
    assert get_inspector(array).get_display_value()[0] == "[lat: 3 x lon: 2] zarr.Array"


def test_zarr_array_without_dims(zarr_group):
    array = _create_array(zarr_group, "plain", shape=(3, 2), chunks=(3, 2), dtype="i8")
    ins = get_inspector(array)
    assert ins.get_display_value()[0] == "[3 x 2] zarr.Array"
    assert ins.get_child("Chunks").text == "(3, 2), 1 chunk"


def test_zarr_inspection_reads_metadata_only(zarr_group, tmp_path):
    # Overwrite every chunk with bytes that cannot be decoded, so that reading any value raises.
    store = tmp_path / "store.zarr" / "sst"
    chunk_files = [
        path
        for path in store.rglob("*")
        if path.is_file() and not path.name.startswith((".", "zarr.json"))
    ]
    assert chunk_files
    for path in chunk_files:
        path.write_bytes(b"not a chunk")
    with pytest.raises(Exception):  # noqa: B017
        zarr_group["sst"][:]

    assert _walk(zarr_group) > 10


def test_zarr_rows_resolve_by_access_key(zarr_group):
    is_known, attributes = _resolve_value_from_path(zarr_group, [encode_access_key(0)])
    assert is_known
    assert isinstance(attributes, ArrayGroup)
    is_known, sst = _resolve_value_from_path(zarr_group, [encode_access_key("sst")])
    assert is_known
    assert type(get_inspector(sst)) is ZarrArrayInspector
    names = [summary.display_name for summary in _summarize_children(zarr_group)]
    assert names[0] == "Attributes"
    assert sorted(names[1:]) == ["masks", "sst"]


#
# h5py
#


@pytest.fixture
def h5_file(tmp_path: Path) -> Any:
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as f:
        f.attrs["title"] = "sea surface temperature"
        time = f.create_dataset("time", data=np.arange(10.0))
        time.make_scale("time")
        sst = f.create_dataset(
            "sst",
            data=np.arange(200, dtype="f4").reshape(10, 4, 5),
            chunks=(6, 4, 5),
            compression="gzip",
            compression_opts=4,
            shuffle=True,
        )
        sst.attrs["units"] = "degC"
        sst.dims[0].attach_scale(time)
        sst.dims[1].label = "lat"
        sst.dims[2].label = "lon"
        f.create_dataset("flat", data=np.zeros((3, 2)))
        f.create_group("masks")
    f = h5py.File(path, "r")
    yield f
    f.close()


def test_inspect_h5py_file(h5_file):
    ins = get_inspector(h5_file)
    assert type(ins) is H5pyGroupInspector
    assert ins.get_display_value() == ("[4 members] h5py.File", True)
    assert ins.get_display_type() == "File [4 members]"
    assert ins.get_kind() == "table"
    assert list(ins.get_children()) == [0, "flat", "masks", "sst", "time"]
    assert ins.get_display_name(0) == "Attributes"
    assert get_inspector(ins.get_child(0)).get_display_value() == ("title", False)
    assert type(get_inspector(ins.get_child("masks"))) is H5pyGroupInspector
    assert get_inspector(ins.get_child("masks")).get_display_value() == (
        "[0 members] h5py.Group",
        True,
    )


def test_inspect_h5py_dataset(h5_file):
    ins = get_inspector(h5_file["sst"])
    assert type(ins) is H5pyDatasetInspector
    assert ins.get_display_value() == ("[time: 10 x lat: 4 x lon: 5] h5py.Dataset", True)
    assert ins.get_display_type() == "Dataset [10x4x5] float32"
    assert ins.get_size() == 800
    assert list(ins.get_children()) == ["Attributes", "Chunks", "Compression", "File"]
    assert [ins.get_child(key).text for key in ("Chunks", "Compression")] == [
        "(6, 4, 5), 2 chunks",
        "gzip (4), shuffle",
    ]
    assert ins.get_child("File").text == h5_file.filename


def test_inspect_h5py_contiguous_dataset(h5_file):
    ins = get_inspector(h5_file["flat"])
    assert ins.get_display_value()[0] == "[3 x 2] h5py.Dataset"
    assert [ins.get_child(key).text for key in ("Chunks", "Compression")] == [
        "none (contiguous)",
        "none",
    ]


def test_h5py_dimension_scale_names_its_dim(h5_file):
    assert get_inspector(h5_file["time"]).get_display_value()[0] == "[time: 10] h5py.Dataset"


def test_h5py_closed_file(tmp_path):
    path = tmp_path / "closed.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.zeros(3))
        dataset = f["x"]
    for value, text in ((f, "<Closed HDF5 file>"), (dataset, "<Closed HDF5 dataset>")):
        ins = get_inspector(value)
        assert ins.get_display_value() == (text, False)
        assert ins.get_length() == 0
        assert not ins.has_children()


def test_h5py_links(tmp_path):
    path = tmp_path / "links.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=np.zeros(3))
        f["soft"] = h5py.SoftLink("/x")
        f["broken"] = h5py.SoftLink("/missing")
        # The external file does not exist, so following the link would raise.
        f["external"] = h5py.ExternalLink("other.h5", "/y")
    with h5py.File(path, "r") as f:
        ins = get_inspector(f)
        assert type(get_inspector(ins.get_child("soft"))) is H5pyDatasetInspector
        assert ins.get_child("broken").text == "broken link to /missing"
        assert ins.get_child("external").text == "external link to other.h5:/y"
        assert _walk(f) > 4


def test_h5py_inspection_reads_metadata_only(h5_file, mocker):
    def fail(*args, **kwargs):  # noqa: ARG001
        raise AssertionError("read values")

    mocker.patch.object(h5py.Dataset, "__getitem__", fail)
    mocker.patch.object(h5py.Dataset, "read_direct", fail)
    mocker.patch.object(h5py.Dataset, "__array__", fail)
    assert _walk(h5_file) > 10


@pytest.mark.parametrize("kind", ["zarr", "h5py"])
def test_group_is_listed_once_per_summary(kind, zarr_group, h5_file, mocker):
    # Listing a remote zarr store without consolidated metadata takes a request.
    group = zarr_group if kind == "zarr" else h5_file
    keys = mocker.spy(type(group), "keys")
    summaries = _summarize_children(group)
    assert len(summaries) == len(list(get_inspector(group).get_children()))
    keys.reset_mock()
    _summarize_children(group)
    # Each child group lists its own members, for its member count.
    assert [call.args[0] for call in keys.call_args_list].count(group) == 1


def test_h5py_missing_member(h5_file):
    with pytest.raises(KeyError):
        get_inspector(h5_file).get_child("missing")
