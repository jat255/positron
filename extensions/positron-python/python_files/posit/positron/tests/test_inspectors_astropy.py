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
    AstropyNDDataInspector,
    AstropyQuantityInspector,
    FitsHDUInspector,
    FitsHDUListInspector,
    FitsHeaderInspector,
    FitsRecordsInspector,
    MetadataRow,
    get_inspector,
)
from positron.variables import _resolve_value_from_path, _summarize_children

from .utils import importorskip_astropy

# astropy defines its units at import, so a type checker cannot see them.
u: Any = importorskip_astropy("astropy.units")

if TYPE_CHECKING:
    from pathlib import Path

    from astropy.io import fits
    from astropy.nddata import CCDData, NDData, StdDevUncertainty
    from astropy.wcs import WCS
else:
    fits = importorskip_astropy("astropy.io.fits")
    from astropy.nddata import CCDData, NDData, StdDevUncertainty
    from astropy.wcs import WCS


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


def _display_names(value: Any) -> list[str]:
    return [summary.display_name for summary in _summarize_children(value)]


#
# Quantity
#


def test_inspect_quantity():
    q = np.arange(6.0).reshape(2, 3) * u.km / u.s
    ins = get_inspector(q)
    assert type(ins) is AstropyQuantityInspector
    assert ins.get_display_value() == ("[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]] km / s", True)
    assert ins.get_display_type() == "Quantity float64 (2, 3) [km / s]"
    assert ins.get_kind() == "table"
    assert ins.has_viewer()
    row = ins.get_child(0)
    assert get_inspector(row).get_display_value()[0] == "[0.0, 1.0, 2.0] km / s"


def test_inspect_quantity_scalar_dimensionless_and_subclass():
    from astropy.coordinates import Angle

    assert get_inspector(5 * u.m).get_display_value()[0] == "5.0 m"
    assert get_inspector(5 * u.m).get_display_type() == "Quantity float64 [m]"
    # A dimensionless quantity has no unit to show.
    assert get_inspector(np.array([1.0, 2.0]) * u.one).get_display_value()[0] == "[1.0, 2.0]"
    angle = Angle([1.0, 2.0], u.deg)
    assert type(get_inspector(angle)) is AstropyQuantityInspector
    assert get_inspector(angle).get_display_type() == "Angle float64 (2) [deg]"


def test_quantity_equality():
    q = np.arange(3.0) * u.m
    ins = get_inspector(q)
    assert ins.equals(ins.deepcopy())
    # The same values in another unit are a change.
    assert not ins.equals(np.arange(3.0) * u.km)


#
# NDData
#


@pytest.fixture
def ccd() -> Any:
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.cunit = ["deg", "deg"]
    return CCDData(
        np.arange(12.0).reshape(3, 4),
        unit="adu",
        uncertainty=StdDevUncertainty(np.ones((3, 4))),
        mask=np.eye(3, 4, dtype=bool),
        wcs=wcs,
        meta={"OBJECT": "M31"},
    )


def test_inspect_nddata(ccd):
    ins = get_inspector(ccd)
    assert type(ins) is AstropyNDDataInspector
    assert ins.get_display_value() == ("[3 x 4] astropy.CCDData adu", True)
    assert ins.get_display_type() == "CCDData [3x4] float64"
    assert ins.get_kind() == "table"
    assert ins.get_size() == 96
    assert _display_names(ccd) == [
        "Unit",
        "Uncertainty (StdDevUncertainty)",
        "Mask",
        "WCS",
        "Meta",
        "Data",
    ]
    assert ins.get_child("WCS").text == "WCS: pos.eq.ra (deg), pos.eq.dec (deg)"
    assert ins.get_child("Uncertainty") is ccd.uncertainty.array
    assert _walk(ccd) > 10


def test_inspect_plain_nddata():
    nddata = NDData(np.zeros((2, 3, 4)))
    assert type(get_inspector(nddata)) is AstropyNDDataInspector
    assert get_inspector(nddata).get_display_value()[0] == "[2 x 3 x 4] astropy.NDData"
    # Without a unit, uncertainty, mask, WCS, or metadata, only the data is listed.
    assert _display_names(nddata) == ["Data"]
    big_endian = CCDData(np.zeros(2, dtype=">i2"), unit="adu")
    assert get_inspector(big_endian).get_display_type() == "CCDData [2] int16"
    records = NDData(np.zeros(2, dtype=[("a", ">i4"), ("b", "<f8")]))
    assert get_inspector(records).get_display_type() == ("NDData [2] [('a', '<i4'), ('b', '<f8')]")
    assert not get_inspector(nddata).has_viewer()


def test_nddata_rows_resolve_by_access_key(ccd):
    is_known, wcs = _resolve_value_from_path(ccd, [encode_access_key("WCS")])
    assert is_known
    assert isinstance(wcs, MetadataRow)


#
# FITS
#


@pytest.fixture
def fits_file(tmp_path: Path) -> Any:
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["OBJECT"] = ("M31", "target name")
    header.add_history("made by a test")
    table = fits.BinTableHDU.from_columns(
        [fits.Column(name="flux", format="D", array=np.arange(3.0))], name="CAT"
    )
    hdus = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.arange(12, dtype="f4").reshape(3, 4), header=header, name="SCI"),
            table,
            fits.ImageHDU(np.zeros((2, 3, 4), dtype="i2"), name="CUBE"),
        ]
    )
    path = tmp_path / "data.fits"
    hdus.writeto(path)
    with fits.open(path) as hdul:
        yield hdul


def test_inspect_hdulist(fits_file):
    ins = get_inspector(fits_file)
    assert type(ins) is FitsHDUListInspector
    assert ins.get_display_value() == ("[4 HDUs] astropy.HDUList", True)
    assert ins.get_display_type() == "HDUList [4 HDUs]"
    assert _display_names(fits_file) == ["0: PRIMARY", "1: SCI", "2: CAT", "3: CUBE"]
    assert [get_inspector(hdu).get_display_value()[0] for hdu in fits_file] == [
        "[no data] astropy.PrimaryHDU",
        "[DEC--TAN: 3 x RA---TAN: 4] astropy.ImageHDU",
        "[3 rows x 1 column] astropy.BinTableHDU",
        "[2 x 3 x 4] astropy.ImageHDU",
    ]
    assert [type(get_inspector(hdu)) for hdu in fits_file] == [FitsHDUInspector] * 4


def test_fits_inspection_reads_headers_only(fits_file):
    # The rows of the file and of its HDUs, as shown before an HDU is expanded
    _summarize_children(fits_file)
    sci = get_inspector(fits_file[1])
    assert sci.get_display_type() == "ImageHDU [3x4] float32"
    assert sci.get_size() == 48
    assert sci.has_viewer()
    assert not get_inspector(fits_file["CUBE"]).has_viewer()
    assert not any(hdu._data_loaded for hdu in fits_file)  # noqa: SLF001
    # Expanding an HDU previews its data.
    assert _walk(fits_file) > 10


def test_inspect_fits_header(fits_file):
    header = fits_file[1].header
    ins = get_inspector(header)
    assert type(ins) is FitsHeaderInspector
    assert ins.get_kind() == "map"
    names = _display_names(header)
    assert names[:2] == ["XTENSION", "BITPIX"]
    texts = {name: ins.get_child(i).text for i, name in enumerate(names)}
    assert texts["OBJECT"] == "'M31' / target name"
    assert texts["BITPIX"] == "-32 / array data type"
    assert texts["HISTORY"] == "made by a test"


def test_fits_table_rows(fits_file):
    data = get_inspector(fits_file["CAT"]).get_child("Data")
    ins = get_inspector(data)
    assert type(ins) is FitsRecordsInspector
    assert ins.get_display_type() == "FITS_rec [3 rows] (flux)"
    assert not ins.has_children()


def test_fits_closed_file(tmp_path):
    path = tmp_path / "closed.fits"
    fits.HDUList([fits.PrimaryHDU(np.zeros((2, 2)))]).writeto(path)
    with fits.open(path) as hdul:
        pass
    # The data was not read before the file closed, so it cannot be read now.
    ins = get_inspector(hdul[0])
    assert ins.get_children() == ["Header"]
    assert not ins.has_viewer()


def test_fits_viewer_size_limit(tmp_path, mocker):
    import positron.data_explorer

    mocker.patch.object(positron.data_explorer, "ARRAY_VIEW_MAX_LAZY_BYTES", 100)
    path = tmp_path / "sizes.fits"
    scaled = fits.ImageHDU(np.zeros((4, 4), dtype="int16"), name="SCALED")
    scaled.header["BSCALE"] = 2.0
    fits.HDUList([fits.PrimaryHDU(np.zeros((10, 10), dtype="uint8")), scaled]).writeto(
        path, output_verify="silentfix"
    )
    with fits.open(path) as hdul:
        # 100 bytes of uint8 values, within the limit until read
        assert get_inspector(hdul[0]).has_viewer()
        # 32 bytes on disk, but its scaled values are read as a wider dtype.
        assert get_inspector(hdul["SCALED"]).get_display_type() == "ImageHDU [4x4]"
        assert not get_inspector(hdul["SCALED"]).has_viewer()
        # Data already in memory has no size limit.
        hdu: Any = hdul["SCALED"]
        _ = hdu.data
        assert get_inspector(hdu).has_viewer()
