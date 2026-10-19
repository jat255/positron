#
# Copyright (C) 2023-2026 Posit Software, PBC. All rights reserved.
# Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
#

# flake8: ignore E203
# pyright: reportOptionalMemberAccess=false
from __future__ import annotations

import ast
import keyword
import logging
import math
import operator
import sys
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Tuple,
    cast,
)

import comm

from ._data_explorer_internal import (
    _get_histogram_method,
    _get_histogram_numpy,
    _get_histogram_polars,
)
from .access_keys import decode_access_key
from .convert import PandasConverter, PolarsConverter
from .data_explorer_comm import (
    ArraySelection,
    BackendState,
    CodeSyntaxName,
    ColumnDisplayType,
    ColumnFilter,
    ColumnFilterType,
    ColumnFilterTypeSupportStatus,
    ColumnFrequencyTable,
    ColumnFrequencyTableParams,
    ColumnHistogram,
    ColumnHistogramParams,
    ColumnProfileResult,
    ColumnProfileSpec,
    ColumnProfileType,
    ColumnProfileTypeSupportStatus,
    ColumnSchema,
    ColumnSelection,
    ColumnSortKey,
    ColumnSummaryStats,
    ColumnValue,
    ConvertedCode,
    ConvertToCodeFeatures,
    ConvertToCodeParams,
    DataExplorerBackendMessageContent,
    DataExplorerBackendRequest,
    DataExplorerFrontendEvent,
    DataSelectionCellIndices,
    DataSelectionCellRange,
    DataSelectionIndices,
    DataSelectionRange,
    DataSelectionSingleCell,
    ExportDataSelectionFeatures,
    ExportDataSelectionParams,
    ExportedData,
    ExportFormat,
    FilterBetween,
    FilterComparison,
    FilterComparisonOp,
    FilterMatchDataTypes,
    FilterResult,
    FilterSetMembership,
    FilterTextSearch,
    FormatOptions,
    GetColumnProfilesFeatures,
    GetColumnProfilesParams,
    GetDataValuesParams,
    GetRowLabelsParams,
    GetSchemaParams,
    RowFilter,
    RowFilterCondition,
    RowFilterType,
    RowFilterTypeSupportStatus,
    SearchSchemaFeatures,
    SearchSchemaParams,
    SearchSchemaResult,
    SearchSchemaSortOrder,
    SetColumnFiltersFeatures,
    SetColumnFiltersParams,
    SetRowFiltersFeatures,
    SetRowFiltersParams,
    SetSortColumnsFeatures,
    SetSortColumnsParams,
    SuggestCodeSyntaxRequest,
    SummaryStatsBoolean,
    SummaryStatsDate,
    SummaryStatsDatetime,
    SummaryStatsNumber,
    SummaryStatsOther,
    SummaryStatsString,
    SupportedFeatures,
    SupportStatus,
    TableSchema,
    TableSelectionKind,
    TableShape,
    TextSearchType,
)
from .inspectors import (
    _dask_nbytes,
    _dask_variables,
    _format_name_list,
    _format_nbytes,
    _is_dask_array,
)
from .positron_comm import CommMessage, PositronComm
from .third_party import is_ibis, is_pandas, is_polars
from .utils import BackgroundJobQueue, guid

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd
    import polars as pl

IBIS_LIMIT_ROWS = 1000
logger = logging.getLogger(__name__)


class DataExplorerWarning(UserWarning):
    """
    Warning raised when there are issues in the Data Explorer relevant to the user.

    This type of warning is shown once in the Console per session.
    """


PathKey = Tuple[str, ...]
SummarizerType = Callable[[Any, FormatOptions], ColumnSummaryStats]


def _summarize_not_implemented(col, options: FormatOptions):
    raise NotImplementedError


# For tables with fewer than this number of columns, when
# instantiating the table view, we compute and cache the list of
# column schemas, which should take well under 10ms.
SCHEMA_CACHE_THRESHOLD = 100


class DataExplorerState:
    name: str
    column_filters: list[ColumnFilter]
    row_filters: list[RowFilter]
    sort_keys: list[ColumnSortKey]

    # Maintain a mapping of column index to inferred dtype for any
    # object columns, to avoid recomputing. If the underlying
    # object is changed, this needs to be reset
    inferred_dtypes: dict[int, str]
    schema_cache: list[ColumnSchema] | None = None

    def __init__(
        self,
        name: str,
        *,
        column_filters=None,
        row_filters=None,
        sort_keys=None,
        inferred_dtypes=None,
        schema_cache=None,
    ):
        self.name = name
        self.column_filters = column_filters or []
        self.row_filters = row_filters or []
        self.sort_keys = sort_keys or []
        self.inferred_dtypes = inferred_dtypes or {}
        self.schema_cache = schema_cache


# Return type for get_updated_state below
StateUpdate = Tuple[bool, DataExplorerState]


class DataExplorerTableView:
    """
    A table interface.

    Interface providing a consistent wrapper around different data
    frame / table types for the data explorer for serving requests from
    the front end. This includes pandas.DataFrame, polars.DataFrame,
    pyarrow.Table, and any others.
    """

    def __init__(
        self,
        table,
        comm: PositronComm,
        state: DataExplorerState,
        job_queue: BackgroundJobQueue,
        sql_string: str | None,
    ):
        # Note: we must not ever modify the user's data
        self.table = table
        self.comm = comm
        self.state = state
        self.job_queue = job_queue
        self.sql_string = sql_string

        self.schema_memo = {}

        self._set_sort_keys(state.sort_keys)

        self._need_recompute = len(state.row_filters) > 0 or len(state.sort_keys) > 0

        # Array of selected ("true") indices using filters. If
        # there are also sort keys, we first filter the unsorted data,
        # and then sort the filtered data only, for the optimistic
        # case that a low-selectivity filters yields less data to sort
        self.filtered_indices = None

        # Array of selected AND reordered row indices
        # (e.g. including any sorting). If there are no sort keys and
        # only filters, then this should be the same as
        # self.filtered_indices
        self.row_view_indices = None

        # Array of selected column indices
        self.column_view_indices = None

        # We store a tuple of (last_filters, matches) here so that we
        # can support scrolling through the schema search results
        # without having to recompute the search. If the search term
        # changes, we discard the last search result. We might add an
        # LRU cache here or something if it helps performance.
        self._search_schema_last_result: tuple[list[ColumnFilter], list[int]] | None = None

        self._update_schema_cache()

    def _update_schema_cache(self):
        # If the number of columns is below the fixed threshold, we
        # compute and store the ColumnSchema objects up front so that
        # we can more easily determine if there has been an in-place
        # schema update. If the schema is large, then we don't cache
        # and where relevant we assume that the schema could have
        # changed.
        if self._should_cache_schema(self.table) and self.state.schema_cache is None:
            self.state.schema_cache = [
                self._get_single_column_schema(i) for i in range(self.table.shape[1])
            ]

    @property
    def _has_row_labels(self) -> bool:
        return False

    @classmethod
    def _should_cache_schema(cls, _table):
        return False

    def _set_sort_keys(self, sort_keys):
        self.state.sort_keys = sort_keys

        # We store the column schemas for each sort key to help with
        # eviction later during updates
        self._sort_key_schemas = [
            self._get_single_column_schema(key.column_index) for key in self.state.sort_keys
        ]

    def _recompute_if_needed(self) -> bool:
        if self._need_recompute:
            self._recompute()
            self._need_recompute = False
            return True
        else:
            return False

    def _update_row_view_indices(self):
        if len(self.state.sort_keys) == 0:
            self.row_view_indices = self.filtered_indices
        else:
            # If we have just applied a new filter, we now resort to
            # reflect the filtered_indices that have just been updated
            self._sort_data()

    # Gets the schema from a list of column indices.
    def get_schema(self, params: GetSchemaParams):
        # Loop over the sorted column indices to get the column schemas the user requested.
        column_schemas = []
        for column_index in sorted(params.column_indices):
            # Validate that the column index isn't negative.
            if column_index < 0:
                raise IndexError

            # Break when the column index is too large.
            if column_index >= len(self.table.columns):
                break

            # Add the column schema.
            column_schemas.append(self._get_single_column_schema(column_index))

        # Return the column schemas.
        return TableSchema(columns=column_schemas)

    def _get_single_column_schema(self, column_index: int) -> ColumnSchema:
        raise NotImplementedError

    def suggest_code_syntax(self, request: SuggestCodeSyntaxRequest):
        raise NotImplementedError

    def convert_to_code(self, request: ConvertToCodeParams):
        raise NotImplementedError

    def search_schema(self, params: SearchSchemaParams):
        filters = params.filters
        sort_order = params.sort_order
        if self._search_schema_last_result is not None:
            last_filters, matches = self._search_schema_last_result
            if last_filters != filters:
                matches = self._column_filter_get_matches(filters)
                self._search_schema_last_result = (filters, matches)
        else:
            matches = self._column_filter_get_matches(filters)
            self._search_schema_last_result = (filters, matches)

        # Apply sorting based on sort_order
        if sort_order == SearchSchemaSortOrder.AscendingName:
            # Sort by column name ascending
            matches = sorted(matches, key=lambda idx: self._get_column_name(idx).lower())
        elif sort_order == SearchSchemaSortOrder.DescendingName:
            # Sort by column name descending
            matches = sorted(
                matches, key=lambda idx: self._get_column_name(idx).lower(), reverse=True
            )
        elif sort_order == SearchSchemaSortOrder.AscendingType:
            # Sort by column type ascending (using lowercase type name)
            matches = sorted(
                matches, key=lambda idx: str(self._get_column_type_display(idx)).lower()
            )
        elif sort_order == SearchSchemaSortOrder.DescendingType:
            # Sort by column type descending (using lowercase type name)
            matches = sorted(
                matches,
                key=lambda idx: str(self._get_column_type_display(idx)).lower(),
                reverse=True,
            )
        # For SearchSchemaSortOrder.Original, keep original order (no sorting needed)

        return SearchSchemaResult(matches=matches)

    def _column_filter_get_matches(self, filters: list[ColumnFilter]):
        matchers = self._get_column_filter_functions(filters)

        return [
            column_index
            for column_index in range(self.table.shape[1])
            if all(matcher(column_index) for matcher in matchers)
        ]

    def _get_column_filter_functions(self, filters: list[ColumnFilter]):
        def _match_text_search(params: FilterTextSearch):
            term = params.term
            if not params.case_sensitive:
                term = term.lower()

                def matches(x):
                    return term in x.lower()

            else:

                def matches(x):
                    return term in x

            def matcher(index):
                return matches(self._get_column_name(index))

            return matcher

        def _match_display_types(params: FilterMatchDataTypes):
            def matcher(index):
                type_display = self._get_column_type_display(index)
                return type_display in params.display_types

            return matcher

        matchers = []
        for filt in filters:
            if filt.filter_type == ColumnFilterType.TextSearch:
                params = filt.params
                assert isinstance(params, FilterTextSearch)
                matchers.append(_match_text_search(params))
            elif filt.filter_type == ColumnFilterType.MatchDataTypes:
                params = filt.params
                assert isinstance(params, FilterMatchDataTypes)
                matchers.append(_match_display_types(params))

        return matchers

    def _get_column_name(self, column_index: int) -> str:
        raise NotImplementedError

    def _get_column_type_display(self, column_index: int) -> ColumnDisplayType:
        raise NotImplementedError

    def get_data_values(self, params: GetDataValuesParams):
        self._recompute_if_needed()
        return self._get_data_values(
            params.columns,
            params.format_options,
        )

    def get_row_labels(self, params: GetRowLabelsParams):
        self._recompute_if_needed()
        return self._get_row_labels(
            params.selection,
            params.format_options,
        )

    def _get_row_labels(self, _selection: ArraySelection, _format_options: FormatOptions):
        # By default, the table has no row labels, so this will only
        # be implemented for pandas
        return {"row_labels": []}

    def export_data_selection(self, params: ExportDataSelectionParams):
        self._recompute_if_needed()
        kind = params.selection.kind
        sel = params.selection.selection
        fmt = params.format
        if kind == TableSelectionKind.SingleCell:
            assert isinstance(sel, DataSelectionSingleCell)
            row_index = sel.row_index
            if self.row_view_indices is not None:
                row_index = self.row_view_indices[row_index]
            return self._export_cell(row_index, sel.column_index, fmt)
        elif kind == TableSelectionKind.CellRange:
            assert isinstance(sel, DataSelectionCellRange)
            return self._export_tabular(
                slice(sel.first_row_index, sel.last_row_index + 1),
                slice(sel.first_column_index, sel.last_column_index + 1),
                fmt,
            )
        elif kind == TableSelectionKind.CellIndices:
            assert isinstance(sel, DataSelectionCellIndices)
            return self._export_tabular(sel.row_indices, sel.column_indices, fmt)
        elif kind == TableSelectionKind.RowRange:
            assert isinstance(sel, DataSelectionRange)
            return self._export_tabular(
                slice(sel.first_index, sel.last_index + 1),
                slice(None),
                fmt,
            )
        elif kind == TableSelectionKind.ColumnRange:
            assert isinstance(sel, DataSelectionRange)
            return self._export_tabular(
                slice(None),
                slice(sel.first_index, sel.last_index + 1),
                fmt,
            )
        elif kind == TableSelectionKind.RowIndices:
            assert isinstance(sel, DataSelectionIndices)
            return self._export_tabular(sel.indices, slice(None), fmt)
        elif kind == TableSelectionKind.ColumnIndices:
            assert isinstance(sel, DataSelectionIndices)
            return self._export_tabular(slice(None), sel.indices, fmt)
        else:
            raise NotImplementedError(f"Unknown data export: {kind}")

    def _export_cell(self, row_index: int, column_index: int, fmt: ExportFormat):
        raise NotImplementedError

    def _export_tabular(self, row_selector, column_selector, fmt: ExportFormat):
        raise NotImplementedError

    def set_column_filters(self, params: SetColumnFiltersParams):
        return self._set_column_filters(params.filters)

    def _set_column_filters(self, filters: list[ColumnFilter]):
        raise NotImplementedError

    def set_row_filters(self, params: SetRowFiltersParams):
        return self._set_row_filters(params.filters)

    def _set_row_filters(self, filters: list[RowFilter]):
        self.state.row_filters = filters
        for filt in filters:
            # If is_valid isn't set, set it based on what is currently
            # supported
            if filt.is_valid is None:
                filt.is_valid = self._is_supported_filter(filt)

        if len(self.state.row_filters) == 0:
            # Simply reset if empty filter set passed
            self.filtered_indices = None
            self._update_row_view_indices()
            return FilterResult(selected_num_rows=len(self.table), had_errors=False)

        # Evaluate all the filters and combine them using the
        # indicated conditions
        combined_mask = None
        had_errors = False
        for filt in filters:
            if filt.is_valid is False:
                # If filter is invalid, do not evaluate it
                continue

            try:
                single_mask = self._eval_filter(filt)
            except Exception as e:
                had_errors = True

                # Filter fails: we capture the error message and mark
                # the filter as invalid
                filt.is_valid = False
                filt.error_message = str(e)

                # Perhaps use a different log level, but to help with
                # debugging for now.
                logger.warning(e, exc_info=True)
                continue

            if combined_mask is None:
                combined_mask = single_mask
            elif filt.condition == RowFilterCondition.And:
                combined_mask &= single_mask
            elif filt.condition == RowFilterCondition.Or:
                combined_mask |= single_mask

        self.filtered_indices = self._mask_to_indices(combined_mask)
        selected_num_rows = (
            len(self.table) if self.filtered_indices is None else len(self.filtered_indices)
        )

        # Update the view indices, re-sorting if needed
        self._update_row_view_indices()
        return FilterResult(selected_num_rows=selected_num_rows, had_errors=had_errors)

    def _mask_to_indices(self, mask):
        raise NotImplementedError

    def _eval_filter(self, filt: RowFilter):
        raise NotImplementedError

    def set_sort_columns(self, params: SetSortColumnsParams):
        self._set_sort_keys(params.sort_keys)

        if not self._recompute_if_needed():
            # If a re-filter is pending, then it will automatically
            # trigger a sort
            self._sort_data()

    def _sort_data(self):
        raise NotImplementedError

    def get_column_profiles(self, params: GetColumnProfilesParams):
        # Launch task to compute column profiles and return them
        # asynchronously, and return an empty result right away
        self.job_queue.submit(self._get_column_profiles_task, params)
        return {}

    def _get_column_profiles_task(self, params: GetColumnProfilesParams):
        self._recompute_if_needed()
        results = []

        for req in params.profiles:
            try:
                result = self._compute_profiles(
                    req.column_index,
                    req.profiles,
                    params.format_options,
                )
                results.append(result.dict())
            except Exception as e:  # noqa: PERF203
                # Error computing a profile -- don't swallow it and timeout
                logger.error(e, exc_info=True)
                # Append an empty result so the other profiles get computed
                results.append({})

        self.comm.send_event(
            DataExplorerFrontendEvent.ReturnColumnProfiles.value,
            {"callback_id": params.callback_id, "profiles": results},
        )

    def _compute_profiles(
        self,
        column_index: int,
        profiles: list[ColumnProfileSpec],
        format_options: FormatOptions,
    ):
        results = {}
        for spec in profiles:
            profile_type = spec.profile_type
            if profile_type == ColumnProfileType.NullCount:
                results["null_count"] = self._prof_null_count(column_index)
            elif profile_type == ColumnProfileType.SummaryStats:
                results["summary_stats"] = self._prof_summary_stats(column_index, format_options)
            elif profile_type == ColumnProfileType.SmallFrequencyTable:
                assert isinstance(spec.params, ColumnFrequencyTableParams)
                results["small_frequency_table"] = self._prof_freq_table(
                    column_index, spec.params, format_options
                )
            elif profile_type == ColumnProfileType.LargeFrequencyTable:
                assert isinstance(spec.params, ColumnFrequencyTableParams)
                results["large_frequency_table"] = self._prof_freq_table(
                    column_index, spec.params, format_options
                )
            elif profile_type == ColumnProfileType.SmallHistogram:
                assert isinstance(spec.params, ColumnHistogramParams)
                results["small_histogram"] = self._prof_histogram(
                    column_index, spec.params, format_options
                )
            elif profile_type == ColumnProfileType.LargeHistogram:
                assert isinstance(spec.params, ColumnHistogramParams)
                results["large_histogram"] = self._prof_histogram(
                    column_index, spec.params, format_options
                )
            else:
                raise NotImplementedError(profile_type)
        return ColumnProfileResult(**results)

    def get_state(self, _unused):
        self._recompute_if_needed()

        num_rows, num_columns = self.table.shape

        # Account for filters
        if self.row_view_indices is not None:
            filtered_num_rows = len(self.row_view_indices)
        else:
            filtered_num_rows = self.table.shape[0]

        if self.column_view_indices is not None:
            filtered_num_columns = len(self.column_view_indices)
        else:
            filtered_num_columns = self.table.shape[1]

        return BackendState(
            display_name=self.state.name,
            table_shape=TableShape(
                num_rows=filtered_num_rows,
                num_columns=filtered_num_columns,
            ),
            table_unfiltered_shape=TableShape(num_rows=num_rows, num_columns=num_columns),
            has_row_labels=self._has_row_labels,
            column_filters=self.state.column_filters,
            row_filters=self.state.row_filters,
            sort_keys=self.state.sort_keys,
            supported_features=self.FEATURES,
        )

    def _recompute(self):
        # Re-setting the column filters will trigger filtering AND
        # sorting
        self._set_row_filters(self.state.row_filters)

    def get_updated_state(self, new_table) -> StateUpdate:
        raise NotImplementedError

    def _get_adjusted_filters(
        self,
        new_columns,
        schema_changes,
        shifted_columns,
        deleted_columns,
        schema_getter,
    ):
        # Shared by implementations of get_updated_state
        new_filters = []
        for filt in self.state.row_filters:
            column_index = filt.column_schema.column_index
            column_name = filt.column_schema.column_name

            filt = filt.copy(deep=True)

            is_deleted = False
            if column_index in schema_changes:
                # A schema change is only valid if the column name is
                # the same, otherwise it's a deletion
                change = schema_changes[column_index]
                if column_name == change.column_name:
                    filt.column_schema = change.copy()
                else:
                    # Column may be deleted. We need to distinguish
                    # between the case of a deleted column that was
                    # filtered and a filter that was invalid and is
                    # now valid again as a result of changes in the
                    # data (e.g. deleting a column and then re-adding
                    # it right away)
                    if str(new_columns[column_index]) == column_name:
                        # Probably a new column that allows the old
                        # filter to become valid again if the type is
                        # compatible
                        filt.column_schema = schema_getter(
                            column_name,
                            column_index,
                        )
                    else:
                        is_deleted = True
            elif column_index in shifted_columns:
                filt.column_schema.column_index = shifted_columns[column_index]
            elif column_index in deleted_columns:
                # Column deleted
                is_deleted = True

            # If the column was not deleted, we always reset the
            # validity state of the filter in case something we
            # did to the data made a previous filter error (which
            # set the is_valid flag to False) to go away.
            if is_deleted:
                filt.is_valid = False
                filt.error_message = "Column was deleted"
            else:
                filt.is_valid = self._is_supported_filter(filt)
                if filt.is_valid:
                    filt.error_message = None
                else:
                    filt.error_message = "Unsupported column type for filter"

            new_filters.append(filt)

        return new_filters

    def _get_adjusted_sort_keys(
        self, new_columns, schema_changes, shifted_columns, deleted_columns
    ):
        # Shared by implementations of get_updated_state
        new_sort_keys = []
        for i, key in enumerate(self.state.sort_keys):
            column_index = key.column_index
            prior_name = self._sort_key_schemas[i].column_name

            key = key.copy()

            # Evict any sort key that was deleted
            if column_index in schema_changes:
                # A schema change is only valid if the column name is
                # the same, otherwise it's a deletion
                change = schema_changes[column_index]
                if prior_name != change.column_name:
                    # Column deleted
                    continue
            elif column_index in shifted_columns:
                key.column_index = shifted_columns[column_index]
            elif column_index in deleted_columns or prior_name != str(new_columns[column_index]):
                # Column deleted
                continue

            new_sort_keys.append(key)

        return new_sort_keys

    def _get_data_values(
        self,
        selections: list[ColumnSelection],
        format_options: FormatOptions,
    ) -> dict:
        raise NotImplementedError

    SUPPORTED_FILTERS = frozenset()

    def _is_supported_filter(self, filt: RowFilter) -> bool:
        if filt.filter_type not in self.SUPPORTED_FILTERS:
            return False

        display_type = filt.column_schema.type_display

        if filt.filter_type in [
            RowFilterType.IsEmpty,
            RowFilterType.NotEmpty,
            RowFilterType.Search,
        ]:
            # String-only filter types
            return display_type == ColumnDisplayType.String
        elif filt.filter_type == RowFilterType.Compare:
            params = filt.params
            assert isinstance(params, FilterComparison), str(params)
            if params.op in [
                FilterComparisonOp.Eq,
                FilterComparisonOp.NotEq,
            ]:
                return True
            else:
                return display_type in _FILTER_RANGE_COMPARE_SUPPORTED
        elif filt.filter_type in [
            RowFilterType.Between,
            RowFilterType.NotBetween,
        ]:
            return display_type in _FILTER_RANGE_COMPARE_SUPPORTED
        elif filt.filter_type in [
            RowFilterType.IsTrue,
            RowFilterType.IsFalse,
        ]:
            return display_type == ColumnDisplayType.Boolean
        else:
            # Filters always supported
            assert filt.filter_type in [
                RowFilterType.IsNull,
                RowFilterType.NotNull,
                RowFilterType.SetMembership,
            ]
            return True

    def _prof_null_count(self, column_index: int) -> int:
        raise NotImplementedError

    _SUMMARIZERS: MappingProxyType[str, SummarizerType] = MappingProxyType({})

    def _prof_summary_stats(self, column_index: int, options: FormatOptions) -> ColumnSummaryStats:
        col_schema = self._get_single_column_schema(column_index)
        col = self._get_summary_values(column_index)

        ui_type = col_schema.type_display
        handler = self._SUMMARIZERS.get(ui_type, _summarize_not_implemented)

        if handler is None:
            # Return nothing for types we don't yet know how to summarize
            return ColumnSummaryStats(type_display=ui_type)
        else:
            return handler(col, options)

    def _get_column(self, column_index: int):
        raise NotImplementedError

    def _get_summary_values(self, column_index: int):
        """Return the column values that the summarizers read."""
        return self._get_column(column_index)

    def _prof_freq_table(
        self,
        column_index: int,
        params: ColumnFrequencyTableParams,
        format_options: FormatOptions,
    ) -> ColumnFrequencyTable:
        raise NotImplementedError

    def _prof_histogram(
        self,
        column_index: int,
        params: ColumnHistogramParams,
        format_options: FormatOptions,
    ) -> ColumnHistogram:
        raise NotImplementedError

    FEATURES = SupportedFeatures(
        search_schema=SearchSchemaFeatures(
            support_status=SupportStatus.Unsupported,
            supported_types=[],
        ),
        set_column_filters=SetColumnFiltersFeatures(
            support_status=SupportStatus.Unsupported, supported_types=[]
        ),
        set_row_filters=SetRowFiltersFeatures(
            support_status=SupportStatus.Unsupported,
            supports_conditions=SupportStatus.Unsupported,
            supported_types=[],
        ),
        get_column_profiles=GetColumnProfilesFeatures(
            support_status=SupportStatus.Unsupported,
            supported_types=[],
        ),
        set_sort_columns=SetSortColumnsFeatures(support_status=SupportStatus.Unsupported),
        export_data_selection=ExportDataSelectionFeatures(
            support_status=SupportStatus.Unsupported,
            supported_formats=[],
        ),
        convert_to_code=ConvertToCodeFeatures(
            support_status=SupportStatus.Unsupported,
        ),
    )


def _box_number_stats(
    min_val, max_val, mean_val, median_val, std_val, display_type=ColumnDisplayType.Floating
):
    return ColumnSummaryStats(
        type_display=display_type,
        number_stats=SummaryStatsNumber(
            min_value=min_val,
            max_value=max_val,
            mean=mean_val,
            median=median_val,
            stdev=std_val,
        ),
    )


def _box_other_stats(num_unique, type_display=ColumnDisplayType.Object):
    return ColumnSummaryStats(
        type_display=type_display,
        other_stats=SummaryStatsOther(num_unique=int(num_unique)),
    )


def _box_string_stats(num_empty, num_unique):
    return ColumnSummaryStats(
        type_display=ColumnDisplayType.String,
        string_stats=SummaryStatsString(num_empty=int(num_empty), num_unique=int(num_unique)),
    )


def _box_boolean_stats(true_count, false_count):
    return ColumnSummaryStats(
        type_display=ColumnDisplayType.Boolean,
        boolean_stats=SummaryStatsBoolean(true_count=int(true_count), false_count=int(false_count)),
    )


def _box_date_stats(num_unique, min_date, mean_date, median_date, max_date):
    def format_date(x):
        return x.strftime("%Y-%m-%d")

    return ColumnSummaryStats(
        type_display=ColumnDisplayType.Date,
        date_stats=SummaryStatsDate(
            num_unique=int(num_unique),
            min_date=format_date(min_date),
            mean_date=format_date(mean_date),
            median_date=format_date(median_date),
            max_date=format_date(max_date),
        ),
    )


def _format_utc_offset(x):
    if x.tzinfo is None:
        return ""

    offset_seconds = x.utcoffset().total_seconds()
    sign = "+" if offset_seconds >= 0 else "-"

    offset_seconds = abs(offset_seconds)
    offset_hours = int(offset_seconds // 3600)
    offset_minutes = int((offset_seconds % 3600) // 60)
    return f"{sign}{offset_hours:02d}:{offset_minutes:02d}"


def _box_datetime_stats(
    time_unit, num_unique, min_date, mean_date, median_date, max_date, timezone
):
    def format_no_micros(x, utc_offset):
        return x.strftime("%Y-%m-%d %H:%M:%S") + utc_offset

    def format_micros(x, utc_offset):
        return x.strftime("%Y-%m-%d %H:%M:%S.%f") + utc_offset

    def format_date(x):
        if x is None:
            return None

        import pandas as pd

        # NaT has strftime but raises on it, so render it the way the "ns" branch does
        if pd.isna(x):
            return str(x)

        utc_offset = _format_utc_offset(x)
        if time_unit == "s":
            return format_no_micros(x, utc_offset)
        elif time_unit == "ms":
            if x.microsecond == 0:
                return format_no_micros(x, utc_offset)
            else:
                # Strip final 3 digits from microseconds, taking into account whether
                # there is a UTC offset
                formatted = format_micros(x, utc_offset)
                return formatted[:-9] + formatted[-6:] if utc_offset else formatted[:-3]
        elif time_unit == "us":
            return (
                format_no_micros(x, utc_offset)
                if x.microsecond == 0
                else format_micros(x, utc_offset)
            )
        else:
            # This will format the UTC offset for us
            return str(x)

    return ColumnSummaryStats(
        type_display=ColumnDisplayType.Datetime,
        datetime_stats=SummaryStatsDatetime(
            num_unique=num_unique,
            min_date=format_date(min_date),
            mean_date=format_date(mean_date),
            median_date=format_date(median_date),
            max_date=format_date(max_date),
            timezone=timezone,
        ),
    )


class UnsupportedView(DataExplorerTableView):
    def __init__(self, table, comm, state, job_queue, sql_string: str | None = None):
        super().__init__(table, comm, state, job_queue, sql_string)


# Special value codes for the protocol
_VALUE_NULL = 0
_VALUE_NA = 1
_VALUE_NAN = 2
_VALUE_NAT = 3
_VALUE_NONE = 4
_VALUE_INF = 10
_VALUE_NEGINF = 11


class NumPyMathHelper:
    def __init__(self):
        import numpy

        self.np = numpy

    def is_float_scalar(self, value):
        return isinstance(value, (float, self.np.floating))

    def isnan(self, value):
        if isinstance(value, Decimal):
            return False
        return self.np.isnan(value)

    def isinf(self, value):
        if isinstance(value, Decimal):
            return False
        return self.np.isinf(value)


def _builtin_is_float_scalar(value):
    return isinstance(value, float)


def _builtin_isnan(value):
    return value != value


def _builtin_isinf(value):
    return math.isinf(value)


def _get_value_formatter(
    options: FormatOptions,
    math_helper: NumPyMathHelper,
    is_nat: Callable[[Any], bool],
    is_na: Callable[[Any], bool],
) -> Callable[[Any], ColumnValue]:
    """
    Return a function that formats one cell value for the frontend.

    Floats use the float formatting options, and missing or non-finite values become the
    protocol's special value codes. `is_nat` and `is_na` recognize the backend's own
    not-a-time and missing-value markers.
    """
    float_format = _get_float_formatter(options)
    max_length = options.max_value_length

    def _format_value(x):
        if math_helper.is_float_scalar(x):
            if math_helper.isnan(x):
                return _VALUE_NAN
            elif math_helper.isinf(x):
                return _VALUE_INF if x > 0 else _VALUE_NEGINF
            else:
                return float_format(x)
        elif x is None:
            return _VALUE_NONE
        elif is_nat(x):
            return _VALUE_NAT
        elif is_na(x):
            return _VALUE_NA
        else:
            return _safe_stringify(x, max_length)

    return _format_value


def _get_float_formatter(options: FormatOptions) -> Callable:
    sci_format = f".{options.large_num_digits}E"
    medium_format = f".{options.large_num_digits}f"
    small_format = f".{options.small_num_digits}f"

    # The limit for large numbers before switching to scientific
    # notation
    upper_threshold = float("1" + "0" * options.max_integral_digits)

    # The limit for small numbers before switching to scientific
    # notation
    lower_threshold = float("0." + "0" * (options.small_num_digits - 1) + "1")

    thousands_sep = options.thousands_sep

    if thousands_sep is not None:
        # We format with comma then replace later
        medium_format = "," + medium_format

    def base_float_format(x) -> str:
        abs_x = abs(x)

        if abs_x >= 1:
            if abs_x < upper_threshold:
                # Has non-zero integral part but below
                return format(x, medium_format)
            else:
                return format(x, sci_format)
        elif abs_x == 0:
            # Special case 0 to align with other "medium" numbers
            return format(x, medium_format)
        else:
            if abs_x >= lower_threshold:
                # Less than 1 but above lower threshold
                return format(x, small_format)
            else:
                return format(x, sci_format)

    if thousands_sep is not None:
        if thousands_sep != ",":

            def float_format(x) -> str:
                base = base_float_format(x)
                return base.replace(",", thousands_sep)

            return float_format
        else:
            return base_float_format
    else:
        return base_float_format


_FILTER_RANGE_COMPARE_SUPPORTED = {
    ColumnDisplayType.Floating,
    ColumnDisplayType.Integer,
    ColumnDisplayType.Decimal,
    ColumnDisplayType.Date,
    ColumnDisplayType.Datetime,
    ColumnDisplayType.Time,
}


def _pandas_temporal_mapper(type_name):
    if "datetime64" in type_name:
        return "datetime"
    elif "timedelta64" in type_name:
        return "interval"
    return None


def _pandas_summarize_number(
    col: pd.Series, options: FormatOptions, display_type=ColumnDisplayType.Floating
):
    import numpy as np

    math_helper = NumPyMathHelper()

    float_format = _get_float_formatter(options)

    min_val = max_val = median_val = mean_val = std_val = None
    if "complex" in str(col.dtype):
        values = col.to_numpy()
        non_null_values = values[~np.isnan(values)]
        if len(non_null_values) > 0:
            median_val = float_format(np.median(non_null_values))
            mean_val = float_format(np.mean(non_null_values))
    else:
        non_null_values = col[col.notna()].to_numpy()  # type: ignore

        if len(non_null_values) > 0:
            min_val = non_null_values.min()
            max_val = non_null_values.max()

            if not math_helper.isinf(min_val) and not math_helper.isinf(max_val):
                # These stats are not defined when there is an
                # inf/-inf in the data
                mean_val = float_format(non_null_values.mean())
                median_val = float_format(np.median(non_null_values))
                std_val = float_format(non_null_values.std(ddof=1))

            if display_type == ColumnDisplayType.Floating:
                min_val = float_format(min_val)
                max_val = float_format(max_val)
            else:
                min_val = str(min_val)
                max_val = str(max_val)

    return _box_number_stats(
        min_val, max_val, mean_val, median_val, std_val, display_type=display_type
    )


def _pandas_summarize_floating(col: pd.Series, options: FormatOptions):
    return _pandas_summarize_number(col, options, display_type=ColumnDisplayType.Floating)


def _pandas_summarize_integer(col: pd.Series, options: FormatOptions):
    return _pandas_summarize_number(col, options, display_type=ColumnDisplayType.Integer)


def _pandas_summarize_string(col: pd.Series, _options: FormatOptions):
    num_empty = (col.str.len() == 0).sum()
    num_unique = col.nunique()
    return _box_string_stats(num_empty, num_unique)


def _pandas_summarize_object(
    col: pd.Series,
    _options: FormatOptions,
    type_display: ColumnDisplayType = ColumnDisplayType.Object,
):
    num_unique = col.nunique()
    return _box_other_stats(num_unique, type_display=type_display)


def _pandas_summarize_boolean(col: pd.Series, _options: FormatOptions):
    null_count = col.isna().sum()
    true_count = col.sum()
    false_count = len(col) - true_count - null_count
    return _box_boolean_stats(true_count, false_count)


def _pandas_summarize_date(col: pd.Series, _options: FormatOptions):
    import pandas as pd

    col_dttm = pd.to_datetime(col)
    min_date = col.min()
    mean_date = pd.to_datetime(col_dttm.mean()).date()
    median_date = _date_median(col_dttm)
    max_date = col.max()
    num_unique = col.nunique()
    return _box_date_stats(num_unique, min_date, mean_date, median_date, max_date)


def _pandas_summarize_datetime(col: pd.Series, _options: FormatOptions):
    # when there are mixed timezones in a single column, it's
    # possible that any of the operations below can
    # fail. specially if they mix timezone aware datetimes with
    # naive datetimes.

    # if an error happens we return `None` as the field value.
    min_date = _possibly(col.min)
    mean_date = _possibly(col.mean)
    median_date = _possibly(lambda: _date_median(col))
    max_date = _possibly(col.max)

    num_unique = _possibly(col.nunique)

    timezones = col.apply(lambda x: getattr(x, "tz", None)).unique()
    if len(timezones) == 1:
        timezone = str(timezones[0])
    else:
        timezone = [f"{x!s}" for x in timezones[:2]]
        timezone = ", ".join(timezone)
        if len(timezones) > 2:
            timezone = timezone + f", ... ({len(timezones) - 2} more)"

    # May have object dtype, so we only extract the time unit if
    # the .dt attribute is present. Also, older versions of pandas did not
    # support units other than nanos
    time_unit = getattr(col.dt, "unit", "ns") if hasattr(col, "dt") else None

    return _box_datetime_stats(
        time_unit, num_unique, min_date, mean_date, median_date, max_date, timezone
    )


def _safe_stringify(x, max_length: int):
    formatted = str(x)
    if len(formatted) > max_length:
        formatted = formatted[:max_length]
    return formatted


# If there are more than 10M data cells (num_rows x num_columns) then
# we do not do proactive schema caching, even if the number of columns
# is small
PANDAS_CACHE_CELLS_THRESHOLD = 10_000_000

# For long data frames, inferring an exact data type for dtype=object
# columns can significantly slow down get_schema requests. We make a
# trade-off between being exhaustive and returning an exactly correct
# result even in the most esoteric cases (e.g. consider a column of 10
# million null values except for a string in the last entry). In the
# event that we can't make a judgment after the limit here, we risk
# being wrong in these exceptional cases, but in the interest of much
# better performance in the > 99% of cases where the type can be
# inferred accurately by looking at this number of cells.
#
# This 1 million cell limit means that we are willing to spend in the
# ballpark of 10ms for each object column to determine an accurate
# data type. Depending on feedback we can further reduce this to
# improve performance if needed.
PANDAS_INFER_DTYPE_SIZE_LIMIT = 1_000_000


class PandasView(DataExplorerTableView):
    TYPE_NAME_MAPPING = MappingProxyType({"boolean": "bool"})

    def __init__(
        self,
        table: pd.DataFrame,
        comm: PositronComm,
        state: DataExplorerState,
        job_queue: BackgroundJobQueue,
        sql_string: str | None = None,
    ):
        self.was_series = False
        table = self._maybe_wrap(table)
        # For lazy importing NumPy
        self.math_helper = NumPyMathHelper()

        super().__init__(table, comm, state, job_queue, sql_string)

    @property
    def _has_row_labels(self):
        # pandas always has row labels
        return True

    @classmethod
    def _should_cache_schema(cls, table):
        num_rows, num_columns = table.shape
        num_cells = num_rows * num_columns
        return num_columns < SCHEMA_CACHE_THRESHOLD and num_cells < PANDAS_CACHE_CELLS_THRESHOLD

    def _maybe_wrap(self, value):
        import pandas as pd

        if isinstance(value, pd.Series):
            # track if the original value was a Series, so we can
            # convert filters appropriately for the real type
            self.was_series = True
            if value.name is None:
                return pd.DataFrame({"unnamed": value})
            else:
                return pd.DataFrame(value)
        else:
            return value

    def get_updated_state(self, new_table) -> StateUpdate:
        filtered_columns = {
            filt.column_schema.column_index: filt.column_schema for filt in self.state.row_filters
        }

        new_state = DataExplorerState(self.state.name)

        schema_updated = False

        # We go through the columns in the new table and see whether
        # there is a type change or whether a column name moved.
        #
        # TODO: duplicate column names are a can of worms here, and we
        # will need to return to make this logic robust to that
        old_columns = self.table.columns
        shifted_columns: dict[int, int] = {}
        schema_changes: dict[int, ColumnSchema] = {}

        # First, we look for detectable deleted columns
        deleted_columns: set[int] = set()
        if not self.table.columns.equals(new_table.columns):
            for old_index, column in enumerate(self.table.columns):
                if column not in new_table.columns:
                    deleted_columns.add(old_index)
                    schema_updated = True

        # When computing the new display type of a column requires
        # calling infer_dtype, we are careful below to only do it for
        # columns that are involved in a filter

        if new_table is self.table:
            if (
                # Schema was cached before
                self.state.schema_cache is not None
                # Number of columns has not changed
                and len(self.state.schema_cache) == len(self.table.columns)
                # Table is not too big to analyze
                and self._should_cache_schema(new_table)
            ):
                # Schema was previously cached, so we can use that for
                # change detection
                for i, column_name in enumerate(self.table.columns):
                    column = self.table.iloc[:, i]

                    new_schema = self._construct_schema(column, column_name, i, new_state)
                    old_schema = self.state.schema_cache[i]

                    if (
                        new_schema.column_name != old_schema.column_name
                        or new_schema.type_display != old_schema.type_display
                        or new_schema.type_name != old_schema.type_name
                    ):
                        schema_updated = True
                        schema_changes[i] = new_schema
            else:
                # Schema is large enough to not be cached, so we have
                # to assume the worst case of an in-place schema
                # update
                schema_updated = True

                for i, column_name in enumerate(self.table.columns):
                    column = self.table.iloc[:, i]

                    if i in filtered_columns and column.dtype != object:
                        old_schema = filtered_columns[i]
                        if filtered_columns[i].type_name == str(
                            column.dtype
                        ) and old_schema.column_name == str(column_name):
                            # For filtered, non-object dtype columns,
                            # if the type is the same there is no need
                            # for further analysis
                            continue

                    schema_changes[i] = self._construct_schema(column, column_name, i, new_state)
        else:
            # When computing the new display type of a column requires
            # calling infer_dtype, we are careful to only do it for
            # columns that are involved in a filter
            for new_index, column_name in enumerate(new_table.columns):
                # New table has more columns than the old table
                out_of_bounds = new_index >= len(old_columns)

                if out_of_bounds or old_columns[new_index] != column_name:
                    if column_name not in old_columns:
                        # New column
                        schema_updated = True
                        continue
                    # Column was shifted
                    old_index = old_columns.get_loc(column_name)
                    shifted_columns[old_index] = new_index
                else:
                    old_index = new_index

                new_column = new_table.iloc[:, new_index]

                # For object dtype columns, we refuse to make any
                # assumptions about whether the data type has changed
                # and will let re-filtering fail later if there is a
                # problem
                if new_column.dtype == object:
                    # The inferred type could be different
                    schema_updated = True
                else:
                    old_dtype = self.table.iloc[:, old_index].dtype
                    if new_column.dtype == old_dtype:
                        # Type is the same and not object dtype
                        continue

                # The type maybe changed
                schema_updated = True

                if old_index not in filtered_columns:
                    # This column index did not have a row filter
                    # attached to it, so doing further analysis is
                    # unnecessary
                    continue

                schema_changes[old_index] = self._construct_schema(
                    new_column, column_name, new_index, new_state
                )

        def schema_getter(column_name, column_index):
            return self._construct_schema(
                new_table.iloc[:, column_index],
                column_name,
                column_index,
                new_state,
            )

        new_state.row_filters = self._get_adjusted_filters(
            new_table.columns,
            schema_changes,
            shifted_columns,
            deleted_columns,
            schema_getter,
        )

        new_state.sort_keys = self._get_adjusted_sort_keys(
            new_table.columns, schema_changes, shifted_columns, deleted_columns
        )

        return schema_updated, new_state

    def suggest_code_syntax(self, request: SuggestCodeSyntaxRequest):  # noqa: ARG002
        """Returns the supported code types for exporting data."""
        return CodeSyntaxName(code_syntax_name="pandas").dict()

    def convert_to_code(self, request: ConvertToCodeParams):
        """Translates the current data view, including filters and sorts, into a code snippet."""
        converter = PandasConverter(
            self.table,
            self.state.name,
            request,
            was_series=self.was_series,
            sql_string=self.sql_string,
        )
        converted_code = converter.build_code()
        return ConvertedCode(converted_code=converted_code).dict()

    @classmethod
    def _construct_schema(
        cls, column, column_name, column_index: int, state: DataExplorerState
    ) -> ColumnSchema:
        type_name, type_display, timezone = cls._get_type(column, column_index, state)

        return ColumnSchema(
            column_name=str(column_name),
            column_index=column_index,
            type_name=type_name,
            type_display=ColumnDisplayType(type_display),
            timezone=timezone,
        )

    @classmethod
    def _get_inferred_dtype(cls, column, column_index: int, state: DataExplorerState):
        from pandas.api.types import infer_dtype

        if len(column) > PANDAS_INFER_DTYPE_SIZE_LIMIT:
            column = column.iloc[:PANDAS_INFER_DTYPE_SIZE_LIMIT]

        if column_index not in state.inferred_dtypes:
            state.inferred_dtypes[column_index] = infer_dtype(column)
        return state.inferred_dtypes[column_index]

    @classmethod
    def _get_type(
        cls, column, column_index, state: DataExplorerState
    ) -> tuple[str, ColumnDisplayType, str | None]:
        import pandas as pd

        # A helper function for returning the backend type_name and
        # the display type when returning schema results or analyzing
        # schema changes
        dtype = column.dtype

        try:
            tz_obj = column.dtype.tz
            timezone = str(tz_obj) if tz_obj is not None else None
        except AttributeError:
            timezone = None

        if dtype == object:  # noqa: E721
            type_name = cls._get_inferred_dtype(column, column_index, state)
            type_name = cls.TYPE_NAME_MAPPING.get(type_name, type_name)
            type_display = cls._get_type_display(type_name)

        elif isinstance(dtype, pd.CategoricalDtype):
            type_name = str(dtype)
            if dtype.categories.dtype == object:
                categories_type_name = cls._get_inferred_dtype(
                    dtype.categories, column_index, state
                )
                categories_type_name = cls.TYPE_NAME_MAPPING.get(
                    categories_type_name, categories_type_name
                )
                type_display = cls._get_type_display(categories_type_name)
            else:
                categories_type_name = str(dtype.categories.dtype)
                type_display = cls._get_type_display(categories_type_name)
        else:
            # TODO: more sophisticated type mapping
            type_name = str(dtype)
            type_display = cls._get_type_display(type_name)

        return type_name, type_display, timezone

    TYPE_DISPLAY_MAPPING = MappingProxyType(
        {
            "integer": "integer",
            "int8": "integer",
            "int16": "integer",
            "int32": "integer",
            "int64": "integer",
            "uint8": "integer",
            "uint16": "integer",
            "uint32": "integer",
            "uint64": "integer",
            "floating": "floating",
            "float16": "floating",
            "float32": "floating",
            "float64": "floating",
            "complex64": "floating",
            "complex128": "floating",
            "complex256": "floating",
            "mixed-integer": "object",
            "mixed-integer-float": "object",
            "mixed": "object",
            "decimal": "decimal",
            "complex": "floating",
            "bool": "boolean",
            "datetime64": "datetime",
            "datetime64[ns]": "datetime",
            "datetime": "datetime",
            "timedelta64[ns]": "interval",
            "timedelta": "interval",
            "date": "date",
            "time": "time",
            "bytes": "string",
            "empty": "unknown",
            # NA-enabled numeric data types
            "Int8": "integer",
            "Int16": "integer",
            "Int32": "integer",
            "Int64": "integer",
            "UInt8": "integer",
            "UInt16": "integer",
            "UInt32": "integer",
            "UInt64": "integer",
            "Float32": "floating",
            "Float64": "floating",
            # NA-enabled bool
            "boolean": "boolean",
            # NA-enabled string
            "string": "string",
            # pandas 3.0 default StringDtype
            "str": "string",
        }
    )

    TYPE_MAPPERS = (_pandas_temporal_mapper,)

    @classmethod
    def _get_type_display(cls, type_name) -> ColumnDisplayType:
        if type_name in cls.TYPE_DISPLAY_MAPPING:
            type_display = cls.TYPE_DISPLAY_MAPPING[type_name]
        else:
            type_display = None
            for mapper in cls.TYPE_MAPPERS:
                type_display = mapper(type_name)
                if type_display is not None:
                    break

            if type_display is None:
                type_display = "unknown"

        return ColumnDisplayType(type_display)

    def _get_single_column_schema(self, column_index: int):
        if self.state.schema_cache:
            return self.state.schema_cache[column_index]
        elif column_index in self.schema_memo:
            return self.schema_memo[column_index]
        else:
            col_schema = self._construct_schema(
                self.table.iloc[:, column_index],
                self.table.columns[column_index],
                column_index,
                self.state,
            )
            self.schema_memo[column_index] = col_schema
            return col_schema

    def _get_column_name(self, index: int):
        return str(self.table.columns[index])

    def _get_column_type_display(self, column_index: int) -> ColumnDisplayType:
        column = self.table.iloc[:, column_index]
        _, type_display, _ = self._get_type(column, column_index, self.state)
        return type_display

    def _get_data_values(
        self,
        selections: list[ColumnSelection],
        format_options: FormatOptions,
    ) -> dict:
        formatted_columns = []
        for selection in selections:
            col = self.table.iloc[:, selection.column_index]
            spec = selection.spec
            if isinstance(spec, DataSelectionRange):
                if self.row_view_indices is not None:
                    view_slice = self.row_view_indices[spec.first_index : spec.last_index + 1]
                    values = col.take(view_slice)
                else:
                    # No filtering or sorting, just slice directly
                    values = col.iloc[spec.first_index : spec.last_index + 1]
            else:
                if self.row_view_indices is not None:
                    values = col.take(self.row_view_indices.take(spec.indices))
                else:
                    # No filtering or sorting, just slice directly
                    values = col.take(spec.indices)

            formatted_columns.append(self._format_values(values, format_options))

        # Bypass pydantic model for speed
        return {"columns": formatted_columns}

    def _get_row_labels(self, selection: ArraySelection, _: FormatOptions):
        import pandas as pd

        if isinstance(selection, DataSelectionRange):
            if self.row_view_indices is not None:
                view_slice = self.row_view_indices[selection.first_index : selection.last_index + 1]
                indices = self.table.index.take(view_slice)
            else:
                indices = self.table.index[selection.first_index : selection.last_index + 1]
        else:
            if self.row_view_indices is not None:
                indices = self.table.index.take(self.row_view_indices.take(selection.indices))
            else:
                indices = self.table.index.take(selection.indices)

        # Currently, we format MultiIndex in its flat tuple
        # representation. In the future we will return multiple lists
        # of row labels to be formatted more nicely in the UI
        if isinstance(self.table.index, pd.MultiIndex):
            indices = indices.to_flat_index()
        row_labels = [[str(x) for x in indices]]
        return {"row_labels": row_labels}

    def _format_values(self, values, options: FormatOptions) -> list[ColumnValue]:
        import pandas as pd

        _format_value = _get_value_formatter(
            options,
            self.math_helper,
            is_nat=lambda x: x is pd.NaT,
            is_na=lambda x: x is pd.NA,
        )
        return [_format_value(x) for x in values]

    def _export_tabular(self, row_selector, column_selector, fmt: ExportFormat):
        from io import StringIO

        if self.row_view_indices is not None:
            row_selector = self.row_view_indices[row_selector]

        to_export = self.table.iloc[row_selector, column_selector]
        buf = StringIO()

        if fmt == ExportFormat.Csv:
            to_export.to_csv(buf, index=False, sep=",")
        elif fmt == ExportFormat.Tsv:
            to_export.to_csv(buf, sep="\t", index=False)
        elif fmt == ExportFormat.Html:
            to_export.to_html(buf, index=False)
        else:
            raise NotImplementedError(f"Unsupported export format {fmt}")

        result = buf.getvalue()

        # pandas will put a line break at the end of CSV data. If
        # present, remove it
        if result[-1] == "\n":
            result = result[:-1]

        return ExportedData(data=result, format=fmt)

    def _export_cell(self, row_index: int, column_index: int, fmt: ExportFormat):
        return ExportedData(data=str(self.table.iloc[row_index, column_index]), format=fmt)

    def _mask_to_indices(self, mask):
        if mask is not None:
            return mask.nonzero()[0]
        return None

    def _eval_filter(self, filt: RowFilter):
        import pandas as pd

        column_index = filt.column_schema.column_index
        col = self.table.iloc[:, column_index]

        dtype = col.dtype
        inferred_type = self._get_inferred_dtype(col, column_index, self.state)

        mask = None
        if filt.filter_type in (
            RowFilterType.Between,
            RowFilterType.NotBetween,
        ):
            params = filt.params
            assert isinstance(params, FilterBetween)
            left_value = self._coerce_value(params.left_value, dtype, inferred_type)
            right_value = self._coerce_value(params.right_value, dtype, inferred_type)
            if filt.filter_type == RowFilterType.Between:
                mask = (col >= left_value) & (col <= right_value)
            else:
                # NotBetween
                mask = (col < left_value) | (col > right_value)
        elif filt.filter_type == RowFilterType.Compare:
            params = filt.params
            assert isinstance(params, FilterComparison)

            if params.op not in COMPARE_OPS:
                raise ValueError(f"Unsupported filter type: {params.op}")
            op = COMPARE_OPS[params.op]
            # pandas comparison filters return False for null values
            mask = op(col, self._coerce_value(params.value, dtype, inferred_type))
        elif filt.filter_type == RowFilterType.IsEmpty:
            mask = col.str.len() == 0
        elif filt.filter_type == RowFilterType.IsNull:
            mask = col.isna()
        elif filt.filter_type == RowFilterType.NotEmpty:
            mask = col.str.len() != 0
        elif filt.filter_type == RowFilterType.NotNull:
            mask = col.notna()
        elif filt.filter_type == RowFilterType.IsTrue:
            mask = col == True  # noqa: E712
        elif filt.filter_type == RowFilterType.IsFalse:
            mask = col == False  # noqa: E712
        elif filt.filter_type == RowFilterType.SetMembership:
            params = filt.params
            assert isinstance(params, FilterSetMembership)

            boxed_values = pd.Series(
                [self._coerce_value(val, dtype, inferred_type) for val in params.values]  # noqa: PD011
            )
            # IN
            mask = col.isin(boxed_values)
            if not params.inclusive:
                # NOT-IN
                mask = ~mask
        elif filt.filter_type == RowFilterType.Search:
            params = filt.params
            assert isinstance(params, FilterTextSearch)

            if inferred_type != "string":
                col = col.astype(str)

            term = params.term

            if params.search_type == TextSearchType.RegexMatch:
                mask = col.str.match(term, case=params.case_sensitive)
            else:
                if not params.case_sensitive:
                    col = col.str.lower()
                    term = term.lower()
                if params.search_type == TextSearchType.Contains:
                    mask = col.str.contains(term)
                elif params.search_type == TextSearchType.NotContains:
                    mask = ~col.str.contains(term, na=True)
                elif params.search_type == TextSearchType.StartsWith:
                    mask = col.str.startswith(term)
                elif params.search_type == TextSearchType.EndsWith:
                    mask = col.str.endswith(term)

        assert mask is not None

        # Nulls are possible in the mask, so we just fill them if any
        if mask.dtype != bool:
            mask = mask.fillna(value=False).astype(bool)

        # pandas 3.0 CoW returns read-only arrays; copy to allow in-place combination
        return mask.to_numpy().copy()

    @staticmethod
    def _coerce_value(value, dtype, inferred_type):
        import pandas as pd
        import pandas.api.types as pat

        if pat.is_integer_dtype(dtype):
            # For integer types, try to coerce to integer, but if this
            # fails, allow a looser conversion to float
            try:
                return int(value)
            except ValueError as e1:
                try:
                    return pd.Series([value], dtype="float64").iloc[0]
                except ValueError:
                    raise e1 from None
        elif pat.is_bool_dtype(dtype):
            lvalue = value.lower()
            if lvalue == "true":
                return True
            elif lvalue == "false":
                return False
            else:
                raise ValueError(f"Unable to convert {value} to boolean")
        elif "datetime" in inferred_type:
            return _parse_iso8601_like(value, tz=getattr(dtype, "tz", None))
        else:
            # As a fallback, let Series.astype do the coercion
            dummy = pd.Series([value])
            if dummy.dtype != dtype:
                dummy = dummy.astype(dtype)
            return dummy.iloc[0]

    def _sort_data(self) -> None:
        from pandas.core.sorting import lexsort_indexer, nargsort

        if len(self.state.sort_keys) == 1:
            key = self.state.sort_keys[0]
            column = self.table.iloc[:, key.column_index]
            if self.filtered_indices is not None:
                # pandas's univariate null-friendly argsort (computes
                # the sorting indices). Mergesort is needed to make it
                # stable
                sort_indexer = nargsort(
                    column.take(self.filtered_indices),
                    kind="mergesort",
                    ascending=key.ascending,
                )
                # Reorder the filtered_indices to provide the
                # filtered, sorted virtual view for future data
                # requests
                self.row_view_indices = self.filtered_indices.take(sort_indexer)
            else:
                # Data is not filtered
                self.row_view_indices = nargsort(column, kind="mergesort", ascending=key.ascending)
        elif len(self.state.sort_keys) > 1:
            # Multiple sorting keys
            cols_to_sort = []
            directions = []
            for key in self.state.sort_keys:
                col = self._get_column(key.column_index)
                cols_to_sort.append(col)
                directions.append(key.ascending)

            # lexsort_indexer uses np.lexsort and so is always stable
            sort_indexer = lexsort_indexer(cols_to_sort, directions)
            if self.filtered_indices is not None:
                # Create the filtered, sorted virtual view indices
                self.row_view_indices = self.filtered_indices.take(sort_indexer)
            else:
                self.row_view_indices = sort_indexer
        else:
            # This will be None if the data is unfiltered
            self.row_view_indices = self.filtered_indices

    def _get_column(self, column_index: int) -> pd.Series:
        column = self.table.iloc[:, column_index]
        if self.filtered_indices is not None:
            column = column.take(self.filtered_indices)
        return column

    def _prof_null_count(self, column_index: int) -> int:
        return int(self._get_column(column_index).isna().sum())

    _SUMMARIZERS = MappingProxyType(
        {
            ColumnDisplayType.Boolean: _pandas_summarize_boolean,
            ColumnDisplayType.Floating: _pandas_summarize_floating,
            ColumnDisplayType.Integer: _pandas_summarize_integer,
            ColumnDisplayType.Decimal: _pandas_summarize_number,
            ColumnDisplayType.String: _pandas_summarize_string,
            ColumnDisplayType.Date: _pandas_summarize_date,
            ColumnDisplayType.Datetime: _pandas_summarize_datetime,
            ColumnDisplayType.Object: _pandas_summarize_object,
        }
    )

    def _prof_freq_table(
        self,
        column_index: int,
        params: ColumnFrequencyTableParams,
        format_options: FormatOptions,
    ) -> ColumnFrequencyTable:
        col = self._get_column(column_index)
        counts = col.value_counts()

        top_counts = counts.iloc[: params.limit]
        other_group = counts.iloc[params.limit :]

        formatted_groups = self._format_values(top_counts.index, format_options)

        return ColumnFrequencyTable(
            values=formatted_groups,
            counts=[int(x) for x in top_counts],
            other_count=int(other_group.sum()),
        )

    def _prof_histogram(
        self,
        column_index: int,
        params: ColumnHistogramParams,
        format_options: FormatOptions,
    ) -> ColumnHistogram:
        import numpy as np
        import pandas as pd

        col = self._get_column(column_index)

        # TODO: why does this type error?
        data = col[col.notna()].to_numpy()  # type: ignore

        dtype = data.dtype
        is_datetime64 = np.issubdtype(dtype, np.datetime64)

        if is_datetime64:
            data = data.view(np.int64)

        method = _get_histogram_method(params.method)

        bin_counts, bin_edges = _get_histogram_numpy(data, params.num_bins, method=method)

        if is_datetime64:
            # A bit hacky for now, but will replace this with
            # something better soon
            bin_edges = np.floor(bin_edges).astype(np.int64).view(dtype)
            bin_edges = pd.Series(bin_edges)

        formatted_edges = self._format_values(bin_edges, format_options)

        # TODO: formatted_edges should not contain any special values, but we should
        # probably check more carefully.

        return ColumnHistogram(
            bin_edges=[str(x) for x in formatted_edges],
            bin_counts=[int(x) for x in bin_counts],
            quantiles=[],
        )

    SUPPORTED_FILTERS = frozenset(
        {
            RowFilterType.Between,
            RowFilterType.Compare,
            RowFilterType.IsEmpty,
            RowFilterType.IsFalse,
            RowFilterType.IsNull,
            RowFilterType.IsTrue,
            RowFilterType.NotBetween,
            RowFilterType.NotEmpty,
            RowFilterType.NotNull,
            RowFilterType.Search,
            RowFilterType.SetMembership,
        }
    )

    FEATURES = SupportedFeatures(
        search_schema=SearchSchemaFeatures(
            support_status=SupportStatus.Supported,
            supported_types=[
                ColumnFilterTypeSupportStatus(
                    column_filter_type=ColumnFilterType.TextSearch,
                    support_status=SupportStatus.Supported,
                ),
                ColumnFilterTypeSupportStatus(
                    column_filter_type=ColumnFilterType.MatchDataTypes,
                    support_status=SupportStatus.Supported,
                ),
            ],
        ),
        set_column_filters=SetColumnFiltersFeatures(
            support_status=SupportStatus.Unsupported, supported_types=[]
        ),
        set_row_filters=SetRowFiltersFeatures(
            support_status=SupportStatus.Supported,
            # Temporarily disabled for https://github.com/posit-dev/positron/issues/3489 on
            # 6/11/2024. This will be enabled again when the UI has been reworked to support
            # grouping.
            supports_conditions=SupportStatus.Unsupported,
            supported_types=[
                RowFilterTypeSupportStatus(
                    row_filter_type=x, support_status=SupportStatus.Supported
                )
                for x in SUPPORTED_FILTERS
            ],
        ),
        get_column_profiles=GetColumnProfilesFeatures(
            support_status=SupportStatus.Supported,
            supported_types=[
                ColumnProfileTypeSupportStatus(
                    profile_type=profile_type,
                    support_status=SupportStatus.Supported,
                )
                for profile_type in ColumnProfileType
            ],
        ),
        set_sort_columns=SetSortColumnsFeatures(support_status=SupportStatus.Supported),
        export_data_selection=ExportDataSelectionFeatures(
            support_status=SupportStatus.Supported,
            supported_formats=[
                ExportFormat.Csv,
                ExportFormat.Tsv,
                ExportFormat.Html,
            ],
        ),
        convert_to_code=ConvertToCodeFeatures(
            support_status=SupportStatus.Supported,
            code_syntaxes=[CodeSyntaxName(code_syntax_name="pandas")],
        ),
    )


COMPARE_OPS = {
    FilterComparisonOp.Gt: operator.gt,
    FilterComparisonOp.GtEq: operator.ge,
    FilterComparisonOp.Lt: operator.lt,
    FilterComparisonOp.LtEq: operator.le,
    FilterComparisonOp.Eq: operator.eq,
    FilterComparisonOp.NotEq: operator.ne,
}


def _date_median(x):
    """
    Computes the median of a date or datetime series.

    It converts to the integer representation of the datetime,
    then computes the median, and then converts back to a datetime
    """
    import numpy as np
    import pandas as pd

    # the np.array calls are required to please pyright
    median_value = np.int64(np.median(pd.to_numeric(x).to_numpy()))  # type: ignore

    if isinstance(x.dtype, pd.DatetimeTZDtype):
        # pandas has been buggy with datetimetz dtype other than nanosecond,
        # so we convert to nanoseconds and then back to the original dtype
        if x.dtype.unit == "s":
            median_value = median_value * 1_000_000_000
        elif x.dtype.unit == "ms":
            median_value = median_value * 1_000_000
        elif x.dtype.unit == "us":
            median_value = median_value * 1_000

        median_value = pd.Series([median_value], dtype=pd.DatetimeTZDtype(unit="ns", tz=x.dtype.tz))
    else:
        median_value = pd.Series(np.array([median_value], dtype=x.dtype))

    return median_value[0]


def _possibly(f, otherwise=None):
    """Executes a function an if an error occurs, returns `otherwise`."""
    try:
        return f()
    except Exception:
        return otherwise


_ISO_8601_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
]


def _parse_iso8601_like(x, tz=None):
    for fmt in _ISO_8601_FORMATS:
        try:
            result = datetime.strptime(x, fmt)  # noqa: DTZ007

            # Localize tz-naive datetime if needed to avoid TypeError
            if tz is not None:
                if isinstance(tz, str):
                    import pytz

                    tz = pytz.timezone(tz)
                # pytz uses .localize() for DST-aware localization;
                # zoneinfo.ZoneInfo (pandas 3.0) uses .replace(tzinfo=tz)
                if hasattr(tz, "localize"):
                    result = tz.localize(result)
                else:
                    result = result.replace(tzinfo=tz)

            return result
        except ValueError:  # noqa: PERF203
            continue

    raise ValueError(f'"{x}" not ISO8601 YYYY-MM-DD HH:MM:SS format')


# ----------------------------------------------------------------------
# polars Data Explorer RPC implementations


def _polars_summarize_number(
    col: pl.Series, options: FormatOptions, display_type=ColumnDisplayType.Floating
):
    float_format = _get_float_formatter(options)
    min_val = max_val = median_val = mean_val = std_val = None

    is_empty = col.null_count() == len(col)

    if not is_empty:
        min_val = col.min()
        max_val = col.max()

        if not _builtin_isinf(min_val) and not _builtin_isinf(max_val):
            # These stats are not defined when there is an
            # inf/-inf in the data
            mean_val = float_format(col.mean())
            median_val = float_format(col.median())
            std_val = float_format(col.std())

        if display_type == ColumnDisplayType.Integer:
            min_val = str(min_val)
            max_val = str(max_val)
        else:
            min_val = float_format(min_val)
            max_val = float_format(max_val)

    return _box_number_stats(
        min_val,
        max_val,
        mean_val,
        median_val,
        std_val,
        display_type=display_type,
    )


def _polars_summarize_integer(col: pl.Series, options: FormatOptions):
    return _polars_summarize_number(col, options, display_type=ColumnDisplayType.Integer)


def _polars_summarize_string(col: pl.Series, _):
    num_empty = (col.str.len_chars() == 0).sum()
    num_unique = col.n_unique()
    return _box_string_stats(num_empty, num_unique)


def _polars_summarize_object(
    col: pl.Series,
    _format_options: FormatOptions,
    type_display: ColumnDisplayType = ColumnDisplayType.Object,
):
    num_unique = col.n_unique()
    return _box_other_stats(num_unique, type_display=type_display)


def _polars_summarize_boolean(col: pl.Series, _):
    null_count = int(col.is_null().sum())
    true_count = int(col.sum())
    false_count = len(col) - true_count - null_count
    return _box_boolean_stats(true_count, false_count)


def _polars_summarize_date(col: pl.Series, _):
    import polars as pl

    min_date = col.min()
    max_date = col.max()
    num_unique = col.n_unique()

    as_int32 = col.cast(pl.Int32)
    mean_date = _polars_box_value(
        int(as_int32.mean()),  # type: ignore
        col.dtype,
    )
    median_date = _polars_box_value(
        int(as_int32.median()),  # type: ignore
        col.dtype,
    )
    return _box_date_stats(num_unique, min_date, mean_date, median_date, max_date)


def _polars_box_value(val, dtype):
    from polars import Series

    return Series([val], dtype=dtype)[0]


def _polars_summarize_datetime(col: pl.Series, _):
    import polars as pl

    as_int64 = col.cast(pl.Int64)
    mean_date = _polars_box_value(
        int(as_int64.mean()),  # type: ignore
        col.dtype,
    )
    median_date = _polars_box_value(
        int(as_int64.median()),  # type: ignore
        col.dtype,
    )

    # polars Datetime dtypes can either have a static time zone or no time zone
    timezone = str(getattr(col.dtype, "time_zone", None))

    return _box_datetime_stats(
        getattr(col.dtype, "time_unit", None),
        col.n_unique(),
        col.min(),
        mean_date,
        median_date,
        col.max(),
        timezone,
    )


class PolarsView(DataExplorerTableView):
    def __init__(
        self,
        table: pl.DataFrame,
        comm: PositronComm,
        state: DataExplorerState,
        job_queue: BackgroundJobQueue,
        sql_string: str | None = None,
    ):
        super().__init__(table, comm, state, job_queue, sql_string)

    @classmethod
    def _should_cache_schema(cls, table):
        return table.shape[1] < SCHEMA_CACHE_THRESHOLD

    def get_updated_state(self, new_table) -> StateUpdate:
        new_state = DataExplorerState(
            self.state.name,
            row_filters=self.state.row_filters,
            sort_keys=self.state.sort_keys,
        )

        # As of June 2024, polars seems to be really slow for
        # inspecting the metadata of data frames with a large amount
        # of columns. DataFrame.columns, DataFrame.schema, etc. are
        # fairly expensive. Until this changes (or we add some methods
        # in polars to do the metadata comparisons that we need in
        # Rust), we have to be pretty conservative about how much
        # metadata we touch.
        #
        # Note also the syntax "column_name in df" is fairly slow

        if new_table.shape[1] > SCHEMA_CACHE_THRESHOLD:
            # We always say the schema was updated. We'll let filter
            # and sort keys get invalidated by downstream checking
            # rather than proactive invalidation
            return True, new_state

        assert self.state.schema_cache is not None

        schema_updated = False

        # We go through the columns in the new table and see whether
        # there is a type change or whether a column name moved.
        shifted_columns: dict[int, int] = {}
        deleted_columns: set[int] = set()
        schema_changes: dict[int, ColumnSchema] = {}

        new_columns = new_table.columns
        new_columns_set = {c: i for i, c in enumerate(new_columns)}

        old_columns = [c.column_name for c in self.state.schema_cache]
        old_columns_set = {c: i for i, c in enumerate(old_columns)}

        for old_index, column in enumerate(old_columns):
            if column not in new_columns_set:
                deleted_columns.add(old_index)
                schema_updated = True

        for new_index, column_name in enumerate(new_columns):
            # New table has more columns than the old table
            out_of_bounds = new_index >= len(old_columns)

            if out_of_bounds or old_columns[new_index] != column_name:
                if column_name not in old_columns_set:
                    # New column
                    schema_updated = True
                    continue
                # Column was shifted
                old_index = old_columns_set[column_name]
                shifted_columns[old_index] = new_index
            else:
                old_index = new_index

            new_column = new_table[:, new_index]
            old_schema = self.state.schema_cache[old_index]

            if str(new_column.dtype) == old_schema.type_name:
                # dtype is unchanged
                continue

            # The type changed
            schema_updated = True
            schema_changes[old_index] = self._construct_schema(new_column, column_name, new_index)

        def schema_getter(column_name, column_index):
            return self._construct_schema(
                new_table[:, column_index],
                column_name,
                column_index,
            )

        new_state.row_filters = self._get_adjusted_filters(
            new_columns,
            schema_changes,
            shifted_columns,
            deleted_columns,
            schema_getter,
        )

        new_state.sort_keys = self._get_adjusted_sort_keys(
            new_columns, schema_changes, shifted_columns, deleted_columns
        )

        return schema_updated, new_state

    def suggest_code_syntax(self, request: SuggestCodeSyntaxRequest):  # noqa: ARG002
        """Returns the supported code types for exporting data."""
        return CodeSyntaxName(code_syntax_name="polars").dict()

    def convert_to_code(self, request: ConvertToCodeParams):
        """Translates the current data view, including filters and sorts, into a code snippet."""
        converter = PolarsConverter(self.table, self.state.name, request)
        converted_code = converter.build_code()
        return ConvertedCode(converted_code=converted_code).dict()

    def _get_single_column_schema(self, column_index: int):
        if self.state.schema_cache:
            return self.state.schema_cache[column_index]
        elif column_index in self.schema_memo:
            return self.schema_memo[column_index]
        else:
            column = self.table[:, column_index]
            col_schema = self._construct_schema(column, column.name, column_index)
            self.schema_memo[column_index] = col_schema
            return col_schema

    def _get_column_name(self, column_index: int) -> str:
        return self.table[:, column_index].name

    def _get_column_type_display(self, column_index: int) -> ColumnDisplayType:
        column = self.table[:, column_index]
        return self._get_type_display(column.dtype)

    @classmethod
    def _construct_schema(
        cls,
        column: pl.Series,
        column_name: str,
        column_index: int,
    ):
        import polars as pl

        if isinstance(column.dtype, pl.Categorical):
            # Categorical is always string in polars
            type_display = "string"
            # For Categorical types, we just use "Categorical" for the type name
            # for simplicity
            type_name = "Categorical"
        else:
            type_display = cls._get_type_display(column.dtype)
            type_name = str(column.dtype)

        return ColumnSchema(
            column_name=column_name,
            column_index=column_index,
            type_name=type_name,
            type_display=ColumnDisplayType(type_display),
            timezone=getattr(column.dtype, "time_zone", None),
        )

    TYPE_DISPLAY_MAPPING = MappingProxyType(
        {
            "Boolean": "boolean",
            "Int8": "integer",
            "Int16": "integer",
            "Int32": "integer",
            "Int64": "integer",
            "UInt8": "integer",
            "UInt16": "integer",
            "UInt32": "integer",
            "UInt64": "integer",
            "Float32": "floating",
            "Float64": "floating",
            "Binary": "string",
            "String": "string",
            "Date": "date",
            "Datetime": "datetime",
            "Time": "time",
            "Duration": "interval",
            "Decimal": "decimal",
            "Object": "object",
            "List": "array",
            "Struct": "struct",
            "Categorical": "string",
            "Enum": "unknown",
            "Null": "unknown",  # Not yet implemented
            "Unknown": "unknown",
        }
    )

    @classmethod
    def _get_type_display(cls, dtype: pl.DataType):
        key = str(dtype.base_type())
        type_display = cls.TYPE_DISPLAY_MAPPING.get(key, "unknown")
        return ColumnDisplayType(type_display)

    def _search_schema(
        self, filters: list[ColumnFilter], start_index: int, max_results: int
    ) -> SearchSchemaResult:
        raise NotImplementedError

    def _get_data_values(
        self,
        selections: list[ColumnSelection],
        format_options: FormatOptions,
    ) -> dict:
        formatted_columns = []
        for selection in selections:
            col = self.table[:, selection.column_index]
            spec = selection.spec
            if isinstance(spec, DataSelectionRange):
                if self.row_view_indices is not None:
                    view_slice = self.row_view_indices[spec.first_index : spec.last_index + 1]
                    values = col.gather(view_slice)
                else:
                    # No filtering or sorting, just slice
                    values = col[spec.first_index : spec.last_index + 1]
            else:
                if self.row_view_indices is not None:
                    values = col.gather(self.row_view_indices.gather(spec.indices))
                else:
                    values = col.gather(spec.indices)
            formatted_columns.append(self._format_values(values, format_options))

        # Bypass pydantic model for speed
        return {"columns": formatted_columns}

    @classmethod
    def _format_values(cls, values, options: FormatOptions) -> list[ColumnValue]:
        import polars as pl

        float_format = _get_float_formatter(options)
        max_length = options.max_value_length

        def _format_scalar(x):
            if _builtin_is_float_scalar(x):
                if _builtin_isnan(x):
                    return _VALUE_NAN
                else:
                    return float_format(x)
            else:
                return _safe_stringify(x, max_length)

        def _format_series(s):
            result = []
            is_valid_mask = s.is_not_null()
            if s.dtype.base_type() is pl.List:
                # Special recursive formatting for List types
                for i, v in enumerate(s):
                    if is_valid_mask[i]:
                        inner_values = _format_series(v)
                        result.append(
                            "[" + ", ".join("null" if v == 0 else v for v in inner_values) + "]"
                        )
                    else:
                        result.append(_VALUE_NULL)
            else:
                for i, v in enumerate(s):
                    if is_valid_mask[i]:
                        result.append(_format_scalar(v))
                    else:
                        result.append(_VALUE_NULL)
            return result

        return _format_series(values)

    def _export_tabular(self, row_selector, column_selector, fmt: ExportFormat):
        if self.row_view_indices is not None:
            row_selector = self.row_view_indices[row_selector]

        to_export = self.table[row_selector, column_selector]

        if fmt == ExportFormat.Csv:
            result = to_export.write_csv()
        elif fmt == ExportFormat.Tsv:
            result = to_export.write_csv(separator="\t")
        elif fmt == ExportFormat.Html:
            raise NotImplementedError(f"Unsupported export format {fmt}")

        return ExportedData(data=result, format=fmt)

    def _export_cell(self, row_index: int, column_index: int, fmt: ExportFormat):
        return ExportedData(data=str(self.table[row_index, column_index]), format=fmt)

    SUPPORTED_FILTERS = frozenset(
        {
            RowFilterType.Between,
            RowFilterType.Compare,
            RowFilterType.NotBetween,
            RowFilterType.IsNull,
            RowFilterType.NotNull,
            RowFilterType.IsEmpty,
            RowFilterType.NotEmpty,
            RowFilterType.IsTrue,
            RowFilterType.IsFalse,
            RowFilterType.Search,
            RowFilterType.SetMembership,
        }
    )

    def _mask_to_indices(self, mask):
        # Boolean array -> int32 array of true indices
        if mask is not None:
            return mask.arg_true()
        return None

    def _eval_filter(self, filt: RowFilter):
        import polars as pl

        column_index = filt.column_schema.column_index
        col = self.table[:, column_index]

        dtype = col.dtype
        display_type = self._get_type_display(dtype)

        mask = None
        if filt.filter_type in (
            RowFilterType.Between,
            RowFilterType.NotBetween,
        ):
            params = filt.params
            assert isinstance(params, FilterBetween)
            left_value = self._coerce_value(params.left_value, dtype, display_type)
            right_value = self._coerce_value(params.right_value, dtype, display_type)
            mask = col.is_between(left_value, right_value)
            if filt.filter_type == RowFilterType.NotBetween:
                mask = ~mask
        elif filt.filter_type == RowFilterType.Compare:
            params = filt.params
            assert isinstance(params, FilterComparison)

            if params.op not in COMPARE_OPS:
                raise ValueError(f"Unsupported filter type: {params.op}")
            op = COMPARE_OPS[params.op]
            # pandas comparison filters return False for null values
            mask = op(col, self._coerce_value(params.value, dtype, display_type))
        elif filt.filter_type == RowFilterType.IsEmpty:
            if col.dtype.is_(pl.String):
                mask = col.str.len_chars() == 0
            elif col.dtype.is_(pl.Binary):
                # col == b"" segfaults in polars
                mask = col.bin.encode("hex").str.len_chars() == 0
            else:
                raise TypeError(col.dtype)
        elif filt.filter_type == RowFilterType.NotEmpty:
            if col.dtype.is_(pl.String):
                mask = col.str.len_chars() != 0
            elif col.dtype.is_(pl.Binary):
                # col == b"" segfaults in polars
                mask = col.bin.encode("hex").str.len_chars() != 0
            else:
                raise TypeError(col.dtype)
        elif filt.filter_type == RowFilterType.IsNull:
            mask = col.is_null()
        elif filt.filter_type == RowFilterType.NotNull:
            mask = ~col.is_null()
        elif filt.filter_type == RowFilterType.IsTrue:
            mask = col == True  # noqa: E712
        elif filt.filter_type == RowFilterType.IsFalse:
            mask = col == False  # noqa: E712
        elif filt.filter_type == RowFilterType.SetMembership:
            params = filt.params
            assert isinstance(params, FilterSetMembership)

            # Per https://github.com/pola-rs/polars/issues/17771, we
            # have to be really careful here because this can fail on
            # polars 1.x or 0.x
            coerced_values = [self._coerce_value(val, dtype, display_type) for val in params.values]  # noqa: PD011
            try:
                boxed_values = pl.Series(coerced_values, dtype=col.dtype)
            except TypeError:
                boxed_values = pl.Series(
                    coerced_values,
                    dtype=_polars_dtype_from_display(display_type),
                )
            mask = col.is_in(boxed_values)
            if not params.inclusive:
                # NOT-IN
                mask = ~mask
        elif filt.filter_type == RowFilterType.Search:
            params = filt.params
            assert isinstance(params, FilterTextSearch)

            if not col.dtype.is_(pl.String):
                col = col.cast(str)

            term = params.term

            if params.search_type == TextSearchType.RegexMatch:
                if not params.case_sensitive:
                    term = "(?i)" + term
                mask = col.str.contains(term)
            else:
                if not params.case_sensitive:
                    col = col.str.to_lowercase()
                    term = term.lower()
                if params.search_type == TextSearchType.Contains:
                    mask = col.str.contains(term)
                elif params.search_type == TextSearchType.StartsWith:
                    mask = col.str.starts_with(term)
                elif params.search_type == TextSearchType.EndsWith:
                    mask = col.str.ends_with(term)

        assert mask is not None

        # Nulls are possible in the mask, so we just fill them if any
        if mask.null_count() > 0:
            mask[mask.is_null()] = False

        return mask

    @staticmethod
    def _coerce_value(value, dtype, display_type):
        import polars as pl

        if dtype.is_integer():
            # For integer types, try to coerce to integer, but if this
            # fails, allow a looser conversion to float
            try:
                return int(value)
            except ValueError as e:
                try:
                    return pl.Series([value]).cast(pl.Float64)[0]
                except ValueError:
                    raise e from None
        elif dtype.is_(pl.Boolean):
            lvalue = value.lower()
            if lvalue == "true":
                return True
            elif lvalue == "false":
                return False
            else:
                raise ValueError(f"Unable to convert {value} to boolean")
        elif display_type == "datetime":
            return _parse_iso8601_like(value, tz=dtype.time_zone)
        else:
            # As a fallback, let polars.Series.cast do the coercion
            dummy = pl.Series([value])
            if dummy.dtype != dtype:
                dummy = dummy.cast(dtype)
            return dummy[0]

    def _sort_data(self) -> None:
        import polars as pl

        if len(self.state.sort_keys) > 0:
            cols_to_sort = []
            directions = []
            for key in self.state.sort_keys:
                col = self._get_column(key.column_index)
                cols_to_sort.append(col)
                directions.append(not key.ascending)

            indexer_name = guid()

            if self.filtered_indices is not None:
                num_rows = len(self.filtered_indices)
            else:
                num_rows = len(self.table)

            # Do a stable sort of the indices using the columns as sort keys
            to_sort = pl.DataFrame([pl.arange(num_rows, eager=True).alias(indexer_name)])

            try:
                to_sort = to_sort.select(
                    pl.all().sort_by(
                        cols_to_sort,
                        descending=directions,
                        maintain_order=True,
                    )
                )
            except TypeError:
                # Older versions of polars do not have maintain_order
                to_sort = to_sort.select(pl.all().sort_by(cols_to_sort, descending=directions))

            sort_indexer = to_sort[indexer_name]
            if self.filtered_indices is not None:
                # Create the filtered, sorted virtual view indices
                self.row_view_indices = self.filtered_indices.gather(sort_indexer)
            else:
                self.row_view_indices = sort_indexer
        else:
            # No sort keys. This will be None if the data is
            # unfiltered
            self.row_view_indices = self.filtered_indices

    def _prof_null_count(self, column_index: int) -> int:
        return self._get_column(column_index).null_count()

    def _get_column(self, column_index: int) -> pl.Series:
        column = self.table[:, column_index]
        if self.filtered_indices is not None:
            column = column.gather(self.filtered_indices)
        return column

    _SUMMARIZERS = MappingProxyType(
        {
            ColumnDisplayType.Boolean: _polars_summarize_boolean,
            ColumnDisplayType.Floating: _polars_summarize_number,
            ColumnDisplayType.Integer: _polars_summarize_integer,
            ColumnDisplayType.Decimal: _polars_summarize_number,
            ColumnDisplayType.String: _polars_summarize_string,
            ColumnDisplayType.Object: _polars_summarize_object,
            ColumnDisplayType.Date: _polars_summarize_date,
            ColumnDisplayType.Datetime: _polars_summarize_datetime,
        }
    )

    def _prof_freq_table(
        self,
        column_index: int,
        params: ColumnFrequencyTableParams,
        format_options: FormatOptions,
    ) -> ColumnFrequencyTable:
        col = self._get_column(column_index).alias("values")

        col = col.filter(col.is_not_null())
        counts = col.value_counts().sort(by=["count", "values"], descending=[True, False])

        top_counts = counts[: params.limit]
        other_count = int(counts[params.limit :, 1].sum())

        formatted_groups = self._format_values(top_counts[:, 0], format_options)

        return ColumnFrequencyTable(
            values=formatted_groups,
            counts=[int(x) for x in top_counts[:, 1]],
            other_count=other_count,
        )

    def _prof_histogram(
        self,
        column_index: int,
        params: ColumnHistogramParams,
        format_options: FormatOptions,
    ) -> ColumnHistogram:
        import polars as pl

        col = self._get_column(column_index)

        # remove nulls
        data = col.filter(col.is_not_null())
        dtype = data.dtype

        if isinstance(dtype, (pl.Datetime, pl.Time)):
            data = data.cast(pl.Int64)
            cast_bin_edges = True
        elif isinstance(dtype, pl.Date):
            data = data.cast(pl.Int32)
            cast_bin_edges = True
        else:
            cast_bin_edges = False

        method = _get_histogram_method(params.method)

        # Always use the Polars implementation for PolarsView
        bin_counts, bin_edges = _get_histogram_polars(data, params.num_bins, method=method)
        bin_edges = pl.Series(bin_edges)

        if cast_bin_edges:
            bin_edges = bin_edges.cast(dtype)

        formatted_edges = self._format_values(bin_edges, format_options)

        # TODO: make sure that formatted_edges has no special values

        return ColumnHistogram(
            bin_edges=[str(x) for x in formatted_edges],
            bin_counts=[int(x) for x in bin_counts],
            quantiles=[],
        )

    FEATURES = SupportedFeatures(
        search_schema=SearchSchemaFeatures(
            support_status=SupportStatus.Unsupported, supported_types=[]
        ),
        set_column_filters=SetColumnFiltersFeatures(
            support_status=SupportStatus.Unsupported, supported_types=[]
        ),
        set_row_filters=SetRowFiltersFeatures(
            support_status=SupportStatus.Supported,
            supports_conditions=SupportStatus.Unsupported,
            supported_types=[
                RowFilterTypeSupportStatus(
                    row_filter_type=x, support_status=SupportStatus.Supported
                )
                for x in SUPPORTED_FILTERS
            ],
        ),
        get_column_profiles=GetColumnProfilesFeatures(
            support_status=SupportStatus.Supported,
            supported_types=[
                ColumnProfileTypeSupportStatus(
                    profile_type=profile_type,
                    support_status=SupportStatus.Supported,
                )
                for profile_type in ColumnProfileType
            ],
        ),
        export_data_selection=ExportDataSelectionFeatures(
            support_status=SupportStatus.Supported,
            supported_formats=[ExportFormat.Csv, ExportFormat.Tsv],
        ),
        set_sort_columns=SetSortColumnsFeatures(support_status=SupportStatus.Supported),
        convert_to_code=ConvertToCodeFeatures(
            support_status=SupportStatus.Supported,
            code_syntaxes=[
                CodeSyntaxName(code_syntax_name="polars"),
                CodeSyntaxName(code_syntax_name="pandas"),
            ],
        ),
    )


def _polars_dtype_from_display(display_type):
    import polars as pl

    return {
        ColumnDisplayType.Floating: pl.Float64,
        ColumnDisplayType.Integer: pl.Int64,
        ColumnDisplayType.Decimal: pl.Float64,  # Polars doesn't have a separate decimal type
    }.get(display_type)


class PyArrowView(DataExplorerTableView):
    pass


# ----------------------------------------------------------------------
# ibis Data Explorer RPC implementations


class IbisView(PandasView):
    """DataExplorer view implementation for Ibis tables."""

    def __init__(
        self,
        table,
        comm: PositronComm,
        state: DataExplorerState,
        job_queue: BackgroundJobQueue,
        sql_string: str | None = None,
    ):
        self._ibis_table = table
        # Convert to pandas dataframe (limit to 1000 rows for performance)
        pandas_df = table.limit(IBIS_LIMIT_ROWS).execute()

        super().__init__(pandas_df, comm, state, job_queue, sql_string)

    def get_state(self, _unused):
        state = super().get_state(_unused)
        state.display_name = self.state.name + " (preview)"

        return state

    # Override convert_to_code to be unsupported for now, tracked in #9514
    FEATURES = SupportedFeatures(
        search_schema=PandasView.FEATURES.search_schema,
        set_column_filters=PandasView.FEATURES.set_column_filters,
        set_row_filters=PandasView.FEATURES.set_row_filters,
        get_column_profiles=PandasView.FEATURES.get_column_profiles,
        set_sort_columns=PandasView.FEATURES.set_sort_columns,
        export_data_selection=PandasView.FEATURES.export_data_selection,
        convert_to_code=ConvertToCodeFeatures(
            support_status=SupportStatus.Unsupported,
            code_syntaxes=[],
        ),
    )


# ----------------------------------------------------------------------
# Array Data Explorer RPC implementations for 1-D and 2-D numpy, xarray, and dask arrays

# Opening an array view loads its values into memory. That computes a dask-backed array and reads
# a lazily indexed one (for example a variable opened from a file without chunks). Larger arrays
# that are not already in memory have no Data Explorer view.
ARRAY_VIEW_MAX_LAZY_BYTES = 100 * 1024 * 1024


def _is_array(value: Any) -> bool:
    """Whether value is a numpy array, xarray DataArray, or dask array."""
    np_module = sys.modules.get("numpy")
    xr_module = sys.modules.get("xarray")
    return (
        (np_module is not None and isinstance(value, np_module.ndarray))
        or (xr_module is not None and isinstance(value, xr_module.DataArray))
        or _is_dask_array(value)
    )


def _array_lazy_nbytes(value: Any) -> int | None:
    """
    The bytes that opening an array view would load into memory, read from metadata only.

    Returns 0 for an array that is already in memory, and None when the size of a dim is not
    known until the array is computed.
    """
    np_module = sys.modules.get("numpy")
    if np_module is not None and isinstance(value, np_module.ndarray):
        return 0
    dask_sizes = [_dask_nbytes(array) for _, array in _dask_variables(value)]
    if dask_sizes:
        if None in dask_sizes:
            return None
        return sum(cast("list[int]", dask_sizes))
    # A DataArray that is not dask-backed. xarray has no public check for whether its data is
    # loaded, so read Variable._in_memory, and treat the data as lazy if it is missing.
    if getattr(value.variable, "_in_memory", False):
        return 0
    return int(value.nbytes)


def _is_viewable_array(value: Any) -> bool:
    """Whether value is a 1-D or 2-D numpy array, xarray DataArray, or dask array to view."""
    if not _is_array(value) or value.ndim not in (1, 2):
        return False
    if value.dtype.names is not None:
        # Structured dtypes have no single column type
        return False
    nbytes = _array_lazy_nbytes(value)
    return nbytes is not None and nbytes <= ARRAY_VIEW_MAX_LAZY_BYTES


def _array_view_unavailable_message(value: Any, expr: str) -> str | None:
    """
    Explain why the Data Explorer cannot show value, which expr evaluates to.

    Returns None when value is not an array or Dataset, or when the Data Explorer can show it.
    The message names expr as typed. Each suggestion is built from the parsed expr, so it is
    valid code whatever expr's spacing, comments, or operators; when no valid suggestion exists,
    the message gives plain guidance instead.
    """
    node = _parse_expr(expr)
    xr_module = sys.modules.get("xarray")
    if xr_module is not None and isinstance(value, xr_module.Dataset):
        names = list(value.data_vars)
        if not names:
            return f"{expr} is an xarray Dataset with no data variables to view"
        key = _literal_node(names[0])
        command = None
        if node is not None and key is not None:
            command = _view_command(ast.Subscript(value=node, slice=key, ctx=ast.Load()))
        if command is None:
            return (
                f"{expr} is an xarray Dataset; view one of its data variables: "
                f"{_format_name_list(names)[0]}"
            )
        return f"{expr} is an xarray Dataset; view one of its variables, e.g. {command}"
    if not _is_array(value):
        return None
    if value.ndim > 2:
        example = None if node is None else _slice_example(value, node)
        suggestion = "." if example is None else f", e.g. {example}"
        return (
            f"{expr} has {value.ndim} dimensions; the Data Explorer shows 1-D and 2-D arrays. "
            f"Select a slice first{suggestion}"
        )
    if value.ndim in (1, 2):
        nbytes = _array_lazy_nbytes(value)
        if nbytes is None:
            return (
                f"{expr} has a shape that is not known until it is computed. Call .compute() first."
            )
        if nbytes > ARRAY_VIEW_MAX_LAZY_BYTES:
            return (
                f"{expr} is {_format_nbytes(nbytes)} (not in memory), over the "
                f"{_format_nbytes(ARRAY_VIEW_MAX_LAZY_BYTES)} limit for the Data Explorer. "
                "Select a smaller slice or call .compute() first."
            )
    return None


def _parse_expr(expr: str) -> ast.expr | None:
    """The parsed expression expr, or None when it is not a single Python expression."""
    try:
        return ast.parse(expr.strip(), mode="eval").body
    except (SyntaxError, ValueError):
        return None


def _literal_node(value: Any) -> ast.expr | None:
    """A literal expression that evaluates to value, or None when value has no literal form."""
    try:
        # A str subclass such as numpy.str_ has a repr that is not a literal
        text = repr(str(value)) if isinstance(value, str) else repr(value)
        if not _objects_equal(ast.literal_eval(text), value):
            return None
        return ast.parse(text, mode="eval").body
    except Exception:
        return None


def _slice_example(value: Any, node: ast.expr) -> str | None:
    """Code that selects the first slice along the first dim of value, which node evaluates to."""
    xr_module = sys.modules.get("xarray")
    if xr_module is None or not isinstance(value, xr_module.DataArray):
        return ast.unparse(ast.Subscript(value=node, slice=ast.Constant(0), ctx=ast.Load()))
    dim = value.dims[0]
    isel = ast.Attribute(value=node, attr="isel", ctx=ast.Load())
    if isinstance(dim, str) and dim.isidentifier() and not keyword.iskeyword(dim):
        call = ast.Call(func=isel, args=[], keywords=[ast.keyword(arg=dim, value=ast.Constant(0))])
    else:
        key = _literal_node(dim)
        if key is None:
            return None
        call = ast.Call(
            func=isel, args=[ast.Dict(keys=[key], values=[ast.Constant(0)])], keywords=[]
        )
    return ast.unparse(call)


def _view_command(node: ast.expr) -> str | None:
    """
    A %view command that opens what node evaluates to, or None when no command can.

    ast.unparse adds the parentheses that operator precedence needs. IPython replaces `{expr}`
    in a magic's arguments with the value of expr, so code with braces cannot be a %view
    argument. %view splits its arguments on whitespace and strips one pair of quotes around each.
    On Windows only double quotes quote an argument, and a double quote inside an argument is
    removed, so code that has a double quote cannot be a %view argument either. Other code is
    quoted with double quotes when it has whitespace or starts with a single quote.
    """
    code = ast.unparse(node)
    if any(c in code for c in '"{}'):
        return None
    if any(c.isspace() for c in code) or code.startswith("'"):
        return f'%view "{code}"'
    return f"%view {code}"


class _ArrayTable:
    """
    A 1-D or 2-D array in the table shape that DataExplorerTableView reads.

    `values` is always a 2-D numpy array: a 1-D array becomes a single column.
    """

    def __init__(
        self,
        values: np.ndarray,
        columns: list[str],
        row_labels: list[str] | None,
        dims: tuple[str, ...],
        name: str | None,
        mask: np.ndarray | None = None,
    ):
        self.values = values
        # 2-D boolean array, True where a cell is masked (missing), or None
        self.mask = mask
        self.shape: tuple[int, int] = (values.shape[0], values.shape[1])
        self.dtype = values.dtype
        self.columns = columns
        self.row_labels = row_labels
        self.dims = dims
        self.name = name

    def __len__(self) -> int:
        return self.shape[0]

    @classmethod
    def from_value(cls, value: Any) -> _ArrayTable:
        if isinstance(value, _ArrayTable):
            return value
        if not _is_viewable_array(value):
            raise TypeError(f"Cannot view value of type {type(value)} as an array table")

        import numpy as np

        mask = None
        xr_module = sys.modules.get("xarray")
        if xr_module is not None and isinstance(value, xr_module.DataArray):
            # .values computes a dask-backed DataArray
            values = np.asarray(value.values)
            dims = tuple(str(dim) for dim in value.dims)
            name = None if value.name is None else str(value.name)
            dim_labels = [cls._dim_labels(value, dim) for dim in value.dims]
            row_labels = dim_labels[0]
            columns = ["0" if name is None else name] if value.ndim == 1 else dim_labels[1]
        else:
            if isinstance(value, np.ma.MaskedArray):
                # np.asarray would drop the mask
                mask = np.ma.getmaskarray(value)
                value = np.ma.getdata(value)
            # np.asarray computes a dask array
            values = np.asarray(value)
            dims = ()
            name = None
            row_labels = None
            columns = ["0"] if values.ndim == 1 else [str(i) for i in range(values.shape[1])]

        if values.ndim == 1:
            values = values.reshape(-1, 1)
            if mask is not None:
                mask = mask.reshape(-1, 1)
        return cls(values, columns, row_labels, dims, name, mask)

    @staticmethod
    def _dim_labels(data_array: Any, dim: Any) -> list[str]:
        # Use the dim's coordinate values when it has one, else index positions
        if dim in data_array.coords:
            labels = data_array.coords[dim].to_numpy()
            # Datetimes read the same as the cells of a datetime column
            to_str = _datetime64_formatter() if labels.dtype.kind == "M" else str
            return [to_str(label) for label in labels]
        return [str(i) for i in range(data_array.sizes[dim])]


_ARRAY_DTYPE_KIND_DISPLAY = MappingProxyType(
    {
        "b": ColumnDisplayType.Boolean,
        "i": ColumnDisplayType.Integer,
        "u": ColumnDisplayType.Integer,
        "f": ColumnDisplayType.Floating,
        "M": ColumnDisplayType.Datetime,
        "U": ColumnDisplayType.String,
        "S": ColumnDisplayType.String,
    }
)


def _datetime64_formatter() -> Callable[[Any], str]:
    """Return a function that formats a numpy datetime64 scalar the way pandas does."""
    try:
        import pandas as pd
    except ImportError:
        return lambda x: str(x).replace("T", " ")
    return lambda x: str(pd.Timestamp(x))


def _array_summarize_number(
    values: np.ndarray, options: FormatOptions, display_type=ColumnDisplayType.Floating
):
    import numpy as np

    float_format = _get_float_formatter(options)
    math_helper = NumPyMathHelper()

    min_val = max_val = median_val = mean_val = std_val = None
    if len(values) > 0:
        min_val = values.min()
        max_val = values.max()

        if not math_helper.isinf(min_val) and not math_helper.isinf(max_val):
            # These stats are not defined when there is an inf/-inf in the data
            mean_val = float_format(values.mean())
            median_val = float_format(np.median(values))
            # Matches pandas, which yields NaN for a single value
            std_val = float_format(values.std(ddof=1)) if len(values) > 1 else float_format(np.nan)

        if display_type == ColumnDisplayType.Floating:
            min_val = float_format(min_val)
            max_val = float_format(max_val)
        else:
            min_val = str(min_val)
            max_val = str(max_val)

    return _box_number_stats(
        min_val, max_val, mean_val, median_val, std_val, display_type=display_type
    )


def _array_summarize_floating(values: np.ndarray, options: FormatOptions):
    return _array_summarize_number(values, options, display_type=ColumnDisplayType.Floating)


def _array_summarize_integer(values: np.ndarray, options: FormatOptions):
    return _array_summarize_number(values, options, display_type=ColumnDisplayType.Integer)


def _array_summarize_boolean(values: np.ndarray, _options: FormatOptions):
    true_count = int(values.sum())
    return _box_boolean_stats(true_count, len(values) - true_count)


def _array_summarize_string(values: np.ndarray, _options: FormatOptions):
    import numpy as np

    num_empty = int(np.sum(np.char.str_len(values) == 0))
    return _box_string_stats(num_empty, len(np.unique(values)))


def _count_objects(values: Iterable[Any]) -> tuple[list[Any], list[int]]:
    """
    Count the distinct values of an object column by value equality, in order of first appearance.

    Values do not need to be orderable or hashable. Equal values count as one even when one is
    hashable and the other is not, e.g. {1} and frozenset({1}). A comparison that raises or does
    not return a bool counts as not equal.
    """
    # Each distinct value, with its count, in order of first appearance
    uniques: list[Any] = []
    counts: list[int] = []
    # Index into uniques of each hashable value seen, for the fast path
    hashable_index: dict[Any, int] = {}
    # Indices into uniques of the values that could not go in hashable_index
    unhashable_indices: list[int] = []
    for value in values:
        try:
            index = hashable_index.get(value)
            hashable = True
        except Exception:
            # Unhashable, or a hash collision whose comparison raised
            index = None
            hashable = False
        if index is None:
            # Hashable values only need comparing with the values outside the hash table; other
            # values need comparing with every distinct value
            candidates = unhashable_indices if hashable else range(len(uniques))
            match = next((i for i in candidates if _objects_equal(uniques[i], value)), None)
            if match is None:
                index = len(uniques)
                uniques.append(value)
                counts.append(0)
            else:
                index = match
            # Later equal hashable values then take the fast path
            cached = hashable and _try_set_item(hashable_index, value, index)
            if match is None and not cached:
                unhashable_indices.append(index)
        counts[index] += 1
    return uniques, counts


def _try_set_item(mapping: dict[Any, int], key: Any, value: int) -> bool:
    """Set mapping[key] = value, returning False when a comparison of keys raises."""
    try:
        mapping[key] = value
    except Exception:
        return False
    return True


def _objects_equal(a: Any, b: Any) -> bool:
    """Whether a == b is True, treating a comparison that raises or is not a bool as unequal."""
    try:
        result = a == b
    except Exception:
        return False
    return result is True


def _array_summarize_object(values: np.ndarray, _options: FormatOptions):
    return _box_other_stats(len(_count_objects(values)[0]), type_display=ColumnDisplayType.Object)


def _array_summarize_datetime(values: np.ndarray, options: FormatOptions):
    # Reuse the pandas implementation so that formatting matches PandasView. pandas is optional,
    # so without it the summary has the column type and no stats.
    try:
        import pandas as pd
    except ImportError:
        return ColumnSummaryStats(type_display=ColumnDisplayType.Datetime)

    return _pandas_summarize_datetime(pd.Series(values), options)


class ArrayView(DataExplorerTableView):
    """DataExplorer view implementation for 1-D and 2-D numpy, xarray, and dask arrays."""

    table: _ArrayTable

    def __init__(
        self,
        table,
        comm: PositronComm,
        state: DataExplorerState,
        job_queue: BackgroundJobQueue,
        sql_string: str | None = None,
    ):
        # For lazy importing NumPy
        self.math_helper = NumPyMathHelper()
        super().__init__(_ArrayTable.from_value(table), comm, state, job_queue, sql_string)

    @property
    def _values(self) -> np.ndarray:
        return self.table.values  # noqa: PD011

    @property
    def _has_row_labels(self) -> bool:
        return self.table.row_labels is not None

    def _get_single_column_schema(self, column_index: int) -> ColumnSchema:
        return ColumnSchema(
            column_name=self._get_column_name(column_index),
            column_index=column_index,
            type_name=str(self._values.dtype),
            type_display=self._get_column_type_display(column_index),
        )

    def _get_column_name(self, column_index: int) -> str:
        return self.table.columns[column_index]

    def _get_column_type_display(self, column_index: int) -> ColumnDisplayType:  # noqa: ARG002
        # Every column of an array has the same dtype
        kind = self._values.dtype.kind
        return _ARRAY_DTYPE_KIND_DISPLAY.get(kind, ColumnDisplayType.Object)

    def _get_data_values(
        self,
        selections: list[ColumnSelection],
        format_options: FormatOptions,
    ) -> dict:
        np = self.math_helper.np

        kind = self._values.dtype.kind
        is_nat: Callable[[Any], bool] = np.isnat if kind in "mM" else (lambda _x: False)
        _format_value = _get_value_formatter(
            format_options, self.math_helper, is_nat=is_nat, is_na=lambda _x: False
        )
        if kind == "M":
            # Render datetimes the way pandas does, e.g. "2020-01-01 00:00:00"
            _format_scalar = _format_value
            _to_timestamp = _datetime64_formatter()
            _format_value = lambda x: (  # noqa: E731
                _format_scalar(x) if is_nat(x) else _to_timestamp(x)
            )

        formatted_columns = []
        for selection in selections:
            col = self._values[:, selection.column_index]
            mask = self._get_mask(selection.column_index)
            spec = selection.spec
            if isinstance(spec, DataSelectionRange):
                rows = slice(spec.first_index, spec.last_index + 1)
            else:
                rows = np.asarray(spec.indices, dtype=np.intp)
            values = col[rows]
            masked = None if mask is None else mask[rows]
            formatted = [_format_value(x) for x in values]
            if masked is not None:
                formatted = [_VALUE_NA if m else v for v, m in zip(formatted, masked)]
            formatted_columns.append(formatted)

        # Bypass pydantic model for speed
        return {"columns": formatted_columns}

    def _get_row_labels(self, selection: ArraySelection, _format_options: FormatOptions):
        labels = self.table.row_labels
        if labels is None:
            return {"row_labels": []}
        if isinstance(selection, DataSelectionRange):
            selected = labels[selection.first_index : selection.last_index + 1]
        else:
            selected = [labels[i] for i in selection.indices]
        return {"row_labels": [selected]}

    def get_state(self, _unused):
        state = super().get_state(_unused)
        if self.table.dims:
            state.display_name = f"{self.state.name} ({', '.join(self.table.dims)})"
        return state

    def get_updated_state(self, new_table) -> StateUpdate:
        # Arrays have no sort keys or filters to carry over. The schema is unchanged when the
        # shape, dtype, column names, row labels, and dims (which the title names) all match, so
        # only the data needs a refresh.
        new_table = _ArrayTable.from_value(new_table)
        old_table = self.table
        schema_updated = (
            new_table.shape != old_table.shape
            or new_table.dtype != old_table.dtype
            or new_table.columns != old_table.columns
            or new_table.row_labels != old_table.row_labels
            or new_table.dims != old_table.dims
        )
        return schema_updated, DataExplorerState(self.state.name)

    def _get_mask(self, column_index: int) -> np.ndarray | None:
        mask = self.table.mask
        return None if mask is None else mask[:, column_index]

    def _get_column(self, column_index: int) -> np.ndarray:
        """Return one column as a 1-D array, a masked array when the table has masked cells."""
        col = self._values[:, column_index]
        mask = self._get_mask(column_index)
        if mask is None:
            return col
        return self.math_helper.np.ma.array(col, mask=mask)

    def _get_valid_values(self, column_index: int) -> np.ndarray:
        """Return a column's values without masked, NaN, NaT, or None cells."""
        np = self.math_helper.np
        col = self._get_column(column_index)
        if isinstance(col, np.ma.MaskedArray):
            col = col.compressed()
        if col.dtype.kind in "fcmM":
            col = col[~np.isnan(col) if col.dtype.kind in "fc" else ~np.isnat(col)]
        elif col.dtype.kind == "O":
            # Cells render None and float NaN as missing, so count them as nulls
            is_float = self.math_helper.is_float_scalar
            isnan = self.math_helper.isnan
            valid = [not (x is None or (is_float(x) and isnan(x))) for x in col]
            col = col[np.asarray(valid, dtype=bool)]
        return col

    def _prof_null_count(self, column_index: int) -> int:
        return len(self._values) - len(self._get_valid_values(column_index))

    def _get_summary_values(self, column_index: int) -> np.ndarray:
        return self._get_valid_values(column_index)

    _SUMMARIZERS = MappingProxyType(
        {
            ColumnDisplayType.Boolean: _array_summarize_boolean,
            ColumnDisplayType.Floating: _array_summarize_floating,
            ColumnDisplayType.Integer: _array_summarize_integer,
            ColumnDisplayType.String: _array_summarize_string,
            ColumnDisplayType.Datetime: _array_summarize_datetime,
            ColumnDisplayType.Object: _array_summarize_object,
        }
    )

    def _format_values(self, values, options: FormatOptions) -> list[ColumnValue]:
        # Profile values never contain NaT or masked cells
        if self._values.dtype.kind == "M":
            _to_timestamp = _datetime64_formatter()
            return [_to_timestamp(x) for x in values]
        _format_value = _get_value_formatter(
            options, self.math_helper, is_nat=lambda _x: False, is_na=lambda _x: False
        )
        return [_format_value(x) for x in values]

    def _prof_freq_table(
        self,
        column_index: int,
        params: ColumnFrequencyTableParams,
        format_options: FormatOptions,
    ) -> ColumnFrequencyTable:
        np = self.math_helper.np
        valid = self._get_valid_values(column_index)
        if valid.dtype.kind == "O":
            # Objects may not be orderable, and np.unique sorts. Ties keep first appearance.
            distinct, object_counts = _count_objects(valid)
            values = np.empty(len(distinct), dtype=object)
            values[:] = distinct
            counts = np.asarray(object_counts, dtype=np.int64)
        else:
            # Ties in ascending value order
            values, counts = np.unique(valid, return_counts=True)
        # Most frequent first
        order = np.argsort(-counts, kind="stable")
        top = order[: params.limit]
        other = order[params.limit :]

        return ColumnFrequencyTable(
            values=self._format_values(values[top], format_options),
            counts=[int(x) for x in counts[top]],
            other_count=int(counts[other].sum()),
        )

    def _prof_histogram(
        self,
        column_index: int,
        params: ColumnHistogramParams,
        format_options: FormatOptions,
    ) -> ColumnHistogram:
        np = self.math_helper.np

        data = self._get_valid_values(column_index)
        dtype = data.dtype
        is_datetime64 = dtype.kind == "M"
        if is_datetime64:
            data = data.view(np.int64)

        method = _get_histogram_method(params.method)
        bin_counts, bin_edges = _get_histogram_numpy(data, params.num_bins, method=method)

        if is_datetime64:
            bin_edges = np.floor(bin_edges).astype(np.int64).view(dtype)

        formatted_edges = self._format_values(bin_edges, format_options)

        return ColumnHistogram(
            bin_edges=[str(x) for x in formatted_edges],
            bin_counts=[int(x) for x in bin_counts],
            quantiles=[],
        )

    # Sorting, filtering, search, export, and code conversion are not supported for arrays
    FEATURES = SupportedFeatures(
        search_schema=DataExplorerTableView.FEATURES.search_schema,
        set_column_filters=DataExplorerTableView.FEATURES.set_column_filters,
        set_row_filters=DataExplorerTableView.FEATURES.set_row_filters,
        get_column_profiles=GetColumnProfilesFeatures(
            support_status=SupportStatus.Supported,
            supported_types=[
                ColumnProfileTypeSupportStatus(
                    profile_type=profile_type,
                    support_status=SupportStatus.Supported,
                )
                for profile_type in (
                    ColumnProfileType.NullCount,
                    ColumnProfileType.SummaryStats,
                    ColumnProfileType.SmallHistogram,
                    ColumnProfileType.SmallFrequencyTable,
                )
            ],
        ),
        set_sort_columns=DataExplorerTableView.FEATURES.set_sort_columns,
        export_data_selection=DataExplorerTableView.FEATURES.export_data_selection,
        convert_to_code=DataExplorerTableView.FEATURES.convert_to_code,
    )


def _get_table_view(
    table,
    comm: PositronComm,
    state: DataExplorerState,
    job_queue: BackgroundJobQueue,
    *,
    sql_string: str | None = None,
):
    state.name = state.name or guid()

    if is_pandas(table):
        return PandasView(table, comm, state, job_queue, sql_string)
    elif is_polars(table):
        return PolarsView(table, comm, state, job_queue)
    elif is_ibis(table):
        return IbisView(table, comm, state, job_queue, sql_string)
    elif isinstance(table, _ArrayTable) or _is_viewable_array(table):
        return ArrayView(table, comm, state, job_queue, sql_string)
    else:
        return UnsupportedView(table, comm, state, job_queue)


def _value_type_is_supported(value):
    if is_pandas(value):
        return True
    if is_polars(value):
        return True
    if is_ibis(value):
        return True
    # An _ArrayTable is the table of an open ArrayView, which can open a new explorer for it
    return isinstance(value, _ArrayTable) or _is_viewable_array(value)


class DataExplorerService:
    def __init__(self, comm_target: str, job_queue: BackgroundJobQueue) -> None:
        self.comm_target = comm_target
        self.job_queue = job_queue

        # Maps comm_id for each dataset being viewed to PositronComm
        self.comms: dict[str, PositronComm] = {}
        self.table_views: dict[str, DataExplorerTableView] = {}

        # Maps from variable path to set of comm_ids serving DE
        # requests. The user could have multiple DE windows open
        # referencing the same dataset.
        self.path_to_comm_ids: dict[PathKey, set[str]] = {}

        # Mapping from comm_id to the corresponding variable path, if any
        self.comm_id_to_path: dict[str, PathKey] = {}

        # Called when comm closure is initiated from the backend
        self._close_callback = None

    def shutdown(self) -> None:
        for comm_id in list(self.comms.keys()):
            self._close_explorer(comm_id)
        self.path_to_comm_ids.clear()
        self.comm_id_to_path.clear()

    def is_supported(self, value) -> bool:
        return value is not None and _value_type_is_supported(value)

    def register_table(
        self,
        table,
        title,
        variable_path: list[str] | None = None,
        comm_id=None,
        *,
        sql_string: str | None = None,
        inline_only: bool = False,
    ):
        """
        Set up a new comm and data explorer table query wrapper to handle requests and manage state.

        Parameters
        ----------
        table : table-like object
        title : str
            Display name in UI
        variable_path : List[str], default None
            If the data explorer references an assigned variable in
            the user namespace, we track it so that namespace changes
            (variable deletions or assignments) can reflect the
            appropriate change on active data explorer tabs and make
            sure e.g. that we do not hold onto memory inappropriately.
        comm_id : str, default None
            A specific comm identifier to use, otherwise generate a
            random uuid.
        sql_string : str, default None
            If the data explorer was opened from a SQL query result,
            this is the SQL string that was executed to produce the
            result. This is used for code generation.
        inline_only : bool, default False
            If True, the data explorer is for inline display only and
            should not automatically open the full data explorer panel.

        Returns
        -------
        comm_id : str
            The associated (generated or passed in) comm_id
        """
        if not _value_type_is_supported(table):
            raise TypeError(type(table))

        if comm_id is None:
            comm_id = guid()

        comm_open_data = {"title": title, "inline_only": inline_only}
        if variable_path is not None:
            comm_open_data["variable_path"] = variable_path

        base_comm = comm.create_comm(
            target_name=self.comm_target,
            comm_id=comm_id,
            data=comm_open_data,
        )

        def close_callback(_):
            # Notify via callback that the comm_id has closed
            if self._close_callback:
                self._close_callback(comm_id)

            self._close_explorer(comm_id)

        base_comm.on_close(close_callback)
        wrapped_comm = PositronComm(base_comm)
        wrapped_comm.on_msg(self.handle_msg, DataExplorerBackendMessageContent)

        self.table_views[comm_id] = _get_table_view(
            table, wrapped_comm, DataExplorerState(title), self.job_queue, sql_string=sql_string
        )

        if variable_path is not None:
            if not isinstance(variable_path, list):
                raise ValueError(variable_path)

            key = tuple(variable_path)
            self.comm_id_to_path[comm_id] = key

            if key in self.path_to_comm_ids:
                self.path_to_comm_ids[key].add(comm_id)
            else:
                self.path_to_comm_ids[key] = {comm_id}

        self.comms[comm_id] = wrapped_comm
        return comm_id

    def _close_explorer(self, comm_id: str):
        try:
            # This is idempotent, so if the comm is already closed, we
            # can call this again. This will also notify the UI with
            # the comm_close event
            self.comms[comm_id].close()
        except Exception as err:
            logger.warning(err, exc_info=True)

        del self.comms[comm_id]
        del self.table_views[comm_id]

        if comm_id in self.comm_id_to_path:
            path = self.comm_id_to_path[comm_id]
            self.path_to_comm_ids[path].remove(comm_id)
            del self.comm_id_to_path[comm_id]

    def on_comm_closed(self, callback: Callable[[str], None]):
        """Register a callback to invoke when a comm was closed in the backend."""
        self._close_callback = callback

    def variable_has_active_explorers(self, variable_name):
        # Check if any data explorer has been opened with the indicated
        # variable as a path prefix
        return len(self.get_paths_for_variable(variable_name)) > 0

    def get_paths_for_variable(self, variable_name):
        result = []
        for path, comm_ids in self.path_to_comm_ids.items():
            key = decode_access_key(path[0])
            if key == variable_name and len(comm_ids) > 0:
                # An active data explorer shares a path prefix
                result.append(path)
                continue
        return result

    def handle_variable_deleted(self, variable_name):
        """
        Clean up.

        If a variable with active data explorers is deleted, we must
        shut down and delete unneeded state and object references
        stored here.
        """
        affected_paths = self.get_paths_for_variable(variable_name)
        for path in affected_paths:
            for comm_id in list(self.path_to_comm_ids[path]):
                self._close_explorer(comm_id)

    def handle_variable_updated(self, variable_name, new_variable):
        affected_paths = self.get_paths_for_variable(variable_name)
        for path in affected_paths:
            for comm_id in list(self.path_to_comm_ids[path]):
                self._update_explorer_for_comm(comm_id, path, new_variable)

    def _update_explorer_for_comm(self, comm_id: str, path: PathKey, new_variable):
        """
        If a variable is updated, we have to handle the different scenarios.

        * The variable type is the same and the schema is the same,
          but the data is possibly different (e.g. if the object is
          mutable and large, this will happen every time the user
          performs an action). Depending on whether the object
          reference has changed, we can reason about what state needs
          to be invalidated on a case by case basis (for example:
          sort/filter indices will need to be recomputed generally).
        * The variable type is the same and the schema is
          different. Depending on whether the schema or column names
          are different, we may signal the UI to do a "soft" update
          (leaving the cursor position and UI state as is) or a hard
          update (resetting everything to its initial state). We will
          have to do some work to decide whether to preserve filters
          and sorts (if the sorts and filters are still valid after
          the schema change).
        * The variable type is different but still supported in the
          data explorer.
        * The variable type is different and NOT supported in the data
          explorer.
        """
        from .variables import _resolve_value_from_path

        comm = self.comms[comm_id]
        table_view = self.table_views[comm_id]

        full_title = ", ".join([str(decode_access_key(k)) for k in path])

        # When detecting namespace assignments or changes, the first
        # level of the path has already been resolved. If there is a
        # data explorer open for a nested value, then we need to use
        # the same variables inspection logic to resolve it here.
        if len(path) > 1:
            is_found, new_table = _resolve_value_from_path(new_variable, path[1:])
            if not is_found:
                raise KeyError(f"Path {full_title} not found in value")
        else:
            new_table = new_variable

        if not _value_type_is_supported(new_table):
            # If a variable has been assigned a type that is not
            # supported in the existing data explorer tab, we should
            # tear down everything here and let the comm_closed event
            # signal the UI to make the explorer that the user may be
            # looking at invalid.
            return self._close_explorer(comm_id)

        if isinstance(table_view, ArrayView) and _is_viewable_array(new_table):
            # The open view's table is an _ArrayTable, never the user's array type. Load the new
            # array once, for both the schema comparison and the new view.
            new_table = _ArrayTable.from_value(new_table)
            schema_updated, new_state = table_view.get_updated_state(new_table)
        elif not isinstance(new_table, type(table_view.table)):
            # Data structure type has changed. For now, we drop the
            # entire state: sorting keys, filters, etc. and start
            # over. At some point we can return here and selectively
            # preserve state if we can confidently do so.
            schema_updated = True
            new_state = DataExplorerState(table_view.state.name)
        else:
            schema_updated, new_state = table_view.get_updated_state(new_table)

        self.table_views[comm_id] = _get_table_view(
            new_table, table_view.comm, new_state, self.job_queue, sql_string=table_view.sql_string
        )

        if schema_updated:
            comm.send_event(DataExplorerFrontendEvent.SchemaUpdate.value, {})
            return None
        else:
            comm.send_event(DataExplorerFrontendEvent.DataUpdate.value, {})
            return None

    def handle_msg(self, msg: CommMessage[DataExplorerBackendMessageContent], _raw_msg):
        """Handle messages received from the client via the positron.data_explorer comm."""
        comm_id = msg.content.comm_id
        request = msg.content.data

        comm = self.comms[comm_id]
        table = self.table_views[comm_id]

        # open_data_explorer is a service-level operation (creates a new comm),
        # not a table-view operation, so handle it before the generic dispatch.
        if request.method == DataExplorerBackendRequest.OpenDataExplorer:
            self._open_data_explorer(comm_id)
            comm.send_result(None)
            return

        # GetState is the only method that doesn't have params
        result = getattr(table, request.method.value)(getattr(request, "params", None))

        # To help remember to convert pydantic types to dicts
        if result is not None:
            # Convert pydantic types to dict
            if not isinstance(result, dict):
                result = result.dict()
            if isinstance(result, list):
                for x in result:
                    assert isinstance(x, dict)
            else:
                assert isinstance(result, dict)

        comm.send_result(result)

    def _open_data_explorer(self, source_comm_id: str) -> None:
        """Open a new, independent data explorer for the same underlying data."""
        table_view = self.table_views.get(source_comm_id)
        if table_view is None:
            raise KeyError(f"No table view found for comm_id: {source_comm_id}")

        # Preserve the variable_path from the source inline explorer
        source_path = self.comm_id_to_path.get(source_comm_id)
        variable_path = list(source_path) if source_path is not None else None

        self.register_table(
            table_view.table,
            table_view.state.name,
            variable_path=variable_path,
            inline_only=False,
        )


def _get_column_profiles(table_view, schema, query_types, format_options):
    """Generate column profiles for a table view."""
    profiles = []
    skipped_columns = []

    for i, column in enumerate(schema.columns):
        summary_stats = None
        if "summary_stats" in query_types:
            try:
                summary_stats = table_view._prof_summary_stats(i, format_options)  # noqa: SLF001
            except Exception as e:
                # Collect failed columns for later logging
                skipped_columns.append((i, column.column_name, e))
                continue

        profiles.append(
            {
                "column_name": column.column_name,
                "type_display": column.type_display,
                "summary_stats": summary_stats.dict() if summary_stats else None,
            }
        )

    return profiles, skipped_columns
