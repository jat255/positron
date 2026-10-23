#
# Copyright (C) 2023-2024 Posit Software, PBC. All rights reserved.
# Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
#

from __future__ import annotations

import importlib
import os
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generator
from unittest.mock import Mock, patch

import pytest

if TYPE_CHECKING:
    import xarray as xr

    from positron._vendor.pydantic import BaseModel
    from positron.utils import JsonData, JsonRecord


def importorskip_astropy(modname: str) -> Any:
    """
    Import an astropy module, or skip the test when it cannot be imported.

    astropy 6.0, the last release for Python 3.9, raises AttributeError at import with numpy 2.
    """
    try:
        return importlib.import_module(modname)
    except (ImportError, AttributeError) as error:
        pytest.skip(f"could not import {modname}: {error}", allow_module_level=True)


def assert_pydantic_model_equal(actual: BaseModel, expected: BaseModel, exclude: set[str]) -> None:
    actual_dict = actual.dict(exclude=exclude)
    expected_dict = expected.dict(exclude=exclude)
    assert actual_dict == expected_dict


@contextmanager
def preserve_working_directory():
    """Reset the working directory after the context exits."""
    cwd = Path.cwd()
    try:
        yield
    finally:
        os.chdir(cwd)


def assert_register_table_called(
    mock_dataexplorer_service: Mock, obj: Any, title: str, variable_path: list[str] | None = None
) -> None:
    call_args_list = mock_dataexplorer_service.register_table.call_args_list
    assert len(call_args_list) == 1

    passed_table, passed_title = call_args_list[0].args
    assert passed_title == title
    assert passed_table is obj

    if variable_path is not None:
        call_args_kw = mock_dataexplorer_service.register_table.call_args.kwargs
        passed_variable_path = call_args_kw.get("variable_path", None)
        assert passed_variable_path is not None
        assert len(passed_variable_path) == len(variable_path)
        assert passed_variable_path[0] == variable_path[0]


def comm_message(
    data: JsonRecord | None = None,
) -> JsonRecord:
    if data is None:
        data = {}
    return {
        "data": data,
        "metadata": None,
        "buffers": None,
        "msg_type": "comm_msg",
    }


def comm_request(data: JsonRecord, **kwargs) -> JsonRecord:
    return {"content": {"data": data, **kwargs.pop("content", {})}, **kwargs}


def comm_open_message(target_name: str, data: JsonRecord | None = None) -> JsonRecord:
    return {
        **comm_message(data),
        "target_name": target_name,
        "target_module": None,
        "msg_type": "comm_open",
    }


def comm_close_message() -> JsonRecord:
    return {
        **comm_message(),
        "msg_type": "comm_close",
    }


def json_rpc_error(code: int, message: str) -> JsonRecord:
    return comm_message(
        {
            "jsonrpc": "2.0",
            "error": {
                "code": code,
                "message": message,
            },
        }
    )


def json_rpc_notification(method: str, params: JsonRecord | None = None) -> JsonRecord:
    return comm_message(
        {
            "jsonrpc": "2.0",
            "method": method,
            "params": {} if params is None else params,
        }
    )


def json_rpc_request(
    method: str,
    params: JsonRecord | None = None,
    **content: JsonData,
) -> JsonRecord:
    data = {"params": params} if params else {}
    return {
        "content": {
            "data": {
                "jsonrpc": "2.0",
                "id": "test-id",
                "method": method,
                **data,
            },
            **content,
        },
        "header": {},
    }


def json_rpc_response(result: JsonData) -> JsonRecord:
    return comm_message(
        {
            "jsonrpc": "2.0",
            "result": result,
        }
    )


# remove "<class '...'>" from value
def get_type_as_str(value: Any) -> str:
    return repr(type(value))[8:-2]


def percent_difference(actual: float, expected: float) -> float:
    return abs(actual - expected) / actual


def dummy_rpc_request(*args):
    return json_rpc_request(*args, comm_id="dummy_comm_id")


@contextmanager
def patch_positron_execute_request(positron: dict | None = None):
    """Patch the shell's get_parent to return a message with the given positron dict."""
    from positron.positron_ipkernel import PositronIPyKernel

    kernel = PositronIPyKernel.instance()
    positron = positron or {}
    parent = {"content": {"positron": positron}}
    with patch.object(kernel, "get_parent", return_value=parent):
        yield


def run_with_metadata(code: str, positron: dict | None = None):
    """Run a cell with the given positron metadata."""
    from positron.positron_ipkernel import PositronShell

    shell = PositronShell.instance()
    with patch_positron_execute_request(positron):
        return shell.run_cell(code).raise_error()


class CapturedError:
    """A captured kernel error message."""

    def __init__(self, ename: str, evalue: str, traceback: list[str]):
        self.ename = ename
        self.evalue = evalue
        self.traceback = traceback


@contextmanager
def capture_errors() -> Generator[list[CapturedError], None, None]:
    """Capture errors published by the kernel."""
    from positron.positron_ipkernel import PositronShell

    shell = PositronShell.instance()
    errors: list[CapturedError] = []
    session = shell.displayhook.session  # type: ignore
    original_send = session.send

    def send(stream, msg_or_type=None, content=None, *args, **kwargs):
        if msg_or_type == "error" and content is not None:
            error = CapturedError(
                ename=content["ename"],
                evalue=content["evalue"],
                traceback=content["traceback"],
            )
            errors.append(error)
        return original_send(stream, msg_or_type, content, *args, **kwargs)

    session.send = send
    try:
        yield errors
    finally:
        session.send = original_send


#
# xarray and dask fixtures. Each helper skips the calling test when a library it needs is not
# installed, so importing this module never requires xarray or dask.
#


def make_dataset(time: int = 4, lat: int = 3, lon: int = 5) -> xr.Dataset:
    """A small eager Dataset shaped like NOAA OISST: `sst` and `anom` over (time, lat, lon)."""
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    xr = pytest.importorskip("xarray")
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


def fail_on_compute(_block: Any) -> Any:
    """A dask block function that fails the test if dask ever computes it."""
    raise AssertionError("dask computation was triggered")


def make_compute_bomb(values: Any, chunks: Any) -> Any:
    """A dask array with the shape, dtype, and chunks of values that fails if computed."""
    np = pytest.importorskip("numpy")
    da_mod = pytest.importorskip("dask.array")
    lazy = da_mod.from_array(values, chunks=chunks)
    meta = np.empty((0,) * values.ndim, values.dtype)
    return lazy.map_blocks(fail_on_compute, dtype=values.dtype, meta=meta)


def make_lazy_dataset() -> xr.Dataset:
    """`make_dataset(time=12)` with `sst` and `anom` replaced by dask arrays that fail if computed."""
    ds = make_dataset(time=12, lat=3, lon=5)
    out = ds.copy()
    for name in ("sst", "anom"):
        bomb = make_compute_bomb(ds[name].to_numpy(), chunks=(6, 3, 5))
        out[name] = (ds[name].dims, bomb, ds[name].attrs)
    return out
