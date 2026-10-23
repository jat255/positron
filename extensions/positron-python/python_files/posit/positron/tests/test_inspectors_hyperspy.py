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
    HyperspySignalInspector,
    HyperspyTreeInspector,
    MetadataRow,
    _hyperspy_axis_labels,
    get_inspector,
)
from positron.variables import _resolve_value_from_path, _summarize_children

if TYPE_CHECKING:
    import hyperspy.api as hs
else:
    hs = pytest.importorskip("hyperspy.api")


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


@pytest.fixture
def spectrum_image() -> Any:
    """A 4 x 3 spectrum image of 5 channels, with named and calibrated axes."""
    s = hs.signals.Signal1D(np.arange(60, dtype="f4").reshape(3, 4, 5))
    x, y = s.axes_manager.navigation_axes
    (energy,) = s.axes_manager.signal_axes
    x.name, y.name, energy.name = "x", "y", "Energy loss"
    x.units = y.units = "nm"
    energy.units = "eV"
    energy.scale = 0.1
    energy.offset = 100.2
    s.metadata.General.title = "EELS SI"
    return s


def _display_names(value: Any) -> list[str]:
    return [summary.display_name for summary in _summarize_children(value)]


def test_inspect_signal(spectrum_image):
    ins = get_inspector(spectrum_image)
    assert type(ins) is HyperspySignalInspector
    assert ins.get_display_value() == ("[x: 4 x y: 3 | Energy loss: 5] hyperspy.Signal1D", True)
    assert ins.get_display_type() == "Signal1D (4, 3|5) float32"
    assert ins.get_kind() == "table"
    assert ins.get_size() == 240
    assert _display_names(spectrum_image) == ["Title", "Axes", "Metadata", "Data"]
    assert ins.get_child("Title").text == "EELS SI"
    assert ins.get_child("Data") is spectrum_image.data
    assert _walk(spectrum_image) > 10


def test_signal_axes_rows(spectrum_image):
    axes = ArrayGroup(ArrayGroup.AXES, spectrum_image)
    ins = get_inspector(axes)
    assert ins.get_display_value() == ("x, y, Energy loss", False)
    assert _display_names(axes) == ["x", "y", "Energy loss"]
    assert [get_inspector(ins.get_child(key)).get_display_value()[0] for key in range(3)] == [
        "navigation, 4 points, 0.0 to 3.0 nm, step 1.0",
        "navigation, 3 points, 0.0 to 2.0 nm, step 1.0",
        "signal, 5 points, 100.2 to 100.6 eV, step 0.1",
    ]


def test_signal_with_unnamed_axes():
    s = hs.signals.BaseSignal(np.zeros((2, 3)))
    ins = get_inspector(s)
    # HyperSpy writes this shape as (|3, 2)
    assert ins.get_display_value()[0] == "[| 3 x 2] hyperspy.BaseSignal"
    assert ins.get_display_type() == "BaseSignal (|3, 2) float64"
    # Without a title or original metadata, those rows are left out.
    assert _display_names(s) == ["Axes", "Metadata", "Data"]
    assert _display_names(ArrayGroup(ArrayGroup.AXES, s)) == ["axis 0", "axis 1"]


def test_signal_subclasses_and_lazy_signals(spectrum_image):
    image = hs.signals.Signal2D(np.zeros((2, 3, 4)))
    assert type(get_inspector(image)) is HyperspySignalInspector
    assert get_inspector(image).get_display_value()[0] == "[2 | 4 x 3] hyperspy.Signal2D"

    lazy = spectrum_image.as_lazy()
    ins = get_inspector(lazy)
    assert type(ins) is HyperspySignalInspector
    assert ins.get_display_value()[0] == "[x: 4 x y: 3 | Energy loss: 5] hyperspy.LazySignal1D"
    # The size is read from the dask array, without computing it.
    assert ins.get_size() == 240


def test_inspect_metadata_tree(spectrum_image):
    spectrum_image.original_metadata.add_dictionary({"Acquisition": {"dwell_time": 0.01}})
    assert _display_names(spectrum_image) == [
        "Title",
        "Axes",
        "Metadata",
        "Original metadata",
        "Data",
    ]
    ins = get_inspector(spectrum_image.metadata)
    assert type(ins) is HyperspyTreeInspector
    # The internal _HyperSpy node is left out, as in HyperSpy's own listing.
    assert ins.get_display_value() == ("General, Signal", False)
    assert ins.get_kind() == "map"
    general = ins.get_child("General")
    assert get_inspector(general).get_child("title") == "EELS SI"
    with pytest.raises(KeyError):
        ins.get_child("_HyperSpy")


def test_signal_rows_resolve_by_access_key(spectrum_image):
    path = [encode_access_key("Axes"), encode_access_key(2)]
    is_known, row = _resolve_value_from_path(spectrum_image, path)
    assert is_known
    assert isinstance(row, MetadataRow)
    assert row.text.startswith("signal, 5 points")
    path = [encode_access_key("Metadata"), encode_access_key("General")]
    is_known, general = _resolve_value_from_path(spectrum_image, path)
    assert is_known
    assert type(get_inspector(general)) is HyperspyTreeInspector


def test_signal_viewer(spectrum_image):
    # The spectrum image has 3 axes; a line of it has 2.
    assert not get_inspector(spectrum_image).has_viewer()
    assert get_inspector(spectrum_image.inav[0]).has_viewer()
    assert get_inspector(spectrum_image.inav[0, 0]).has_viewer()


def test_signal_non_uniform_axis():
    axis = {"axis": np.array([1.0, 2.0, 4.0, 8.0]), "name": "Energy", "units": "eV"}
    s = hs.signals.Signal1D(np.zeros(4), axes=[axis])
    ins = get_inspector(ArrayGroup(ArrayGroup.AXES, s))
    assert get_inspector(ins.get_child(0)).get_display_value()[0] == (
        "signal, 4 points, 1.0 to 8.0 eV, non-uniform"
    )


def test_axis_labels_use_the_fewest_digits_that_stay_distinct():
    # Rounding error of offset + scale * index is hidden.
    assert _hyperspy_axis_labels(100.2 + 0.1 * np.arange(3)) == ["100.2", "100.3", "100.4"]
    # Values that differ only in the 8th digit keep 8 digits.
    assert _hyperspy_axis_labels([1.0000001, 1.0000002]) == ["1.0000001", "1.0000002"]


def test_non_uniform_axis_text_has_6_digits():
    axis = {"axis": np.array([1.59997158398, 2.5, 5.57244872918]), "name": "Energy"}
    s = hs.signals.Signal1D(np.zeros(3), axes=[axis])
    row = get_inspector(ArrayGroup(ArrayGroup.AXES, s)).get_child(0)
    assert row.text == "signal, 3 points, 1.59997 to 5.57245, non-uniform"


def test_axis_text_keeps_close_ends_distinct():
    s = hs.signals.Signal1D(np.zeros(2))
    (axis,) = s.axes_manager.signal_axes
    axis.offset = 100.0000001
    axis.scale = 1e-7
    row = get_inspector(ArrayGroup(ArrayGroup.AXES, s)).get_child(0)
    assert row.text == "signal, 2 points, 100.0000001 to 100.0000002, step 1e-07"
