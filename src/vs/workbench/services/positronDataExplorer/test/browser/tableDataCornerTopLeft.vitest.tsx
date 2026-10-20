/*---------------------------------------------------------------------------------------------
 *  Copyright (C) 2026 Posit Software, PBC. All rights reserved.
 *  Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
 *--------------------------------------------------------------------------------------------*/

/// <reference types="vitest/globals" />

// Testing libraries.
import { screen } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';

// Other dependencies.
import { Emitter } from '../../../../../base/common/event.js';
import { IConfigurationService } from '../../../../../platform/configuration/common/configuration.js';
import { TestConfigurationService } from '../../../../../platform/configuration/test/common/testConfigurationService.js';
import { IHoverService } from '../../../../../platform/hover/browser/hover.js';
import { NullHoverService } from '../../../../../platform/hover/test/browser/nullHoverService.js';
import { PositronReactServices } from '../../../../../base/browser/positronReactServices.js';
import { createTestContainer } from '../../../../../test/vitest/positronTestContainer.js';
import { DataGridCornerTopLeft } from '../../../../browser/positronDataGrid/components/dataGridCornerTopLeft.js';
import { PositronDataGridContextProvider } from '../../../../browser/positronDataGrid/positronDataGridContext.js';
import { DataExplorerClientInstance } from '../../../languageRuntime/common/languageRuntimeDataExplorerClient.js';
import { BackendState, ColumnDisplayType, ColumnSelection, SchemaUpdateEvent, SupportStatus, TableData, TableRowLabels, TableSchema } from '../../../languageRuntime/common/positronDataExplorerComm.js';
import { TableDataDataGridInstance } from '../../browser/tableDataDataGridInstance.js';
import { TableDataCache } from '../../common/tableDataCache.js';
import { getColumnSchema } from '../../common/positronDataExplorerMocks.js';
import { setupRTLRenderer } from '../../../../../test/vitest/reactTestingLibrary.js';
import { stubInterface } from '../../../../../test/vitest/stubInterface.js';
import { PositronActionBarHoverManager } from '../../../../../platform/positronActionBar/browser/positronActionBarHoverManager.js';
import { TableDataCornerTopLeft } from '../../browser/components/tableDataCornerTopLeft.js';

describe('TableDataCornerTopLeft', () => {
	const rtl = setupRTLRenderer();

	function newHoverManager() {
		return stubInterface<PositronActionBarHoverManager>({ showHover: vi.fn(), hideHover: vi.fn() });
	}

	it('names both axes, with the full names in the hover', async () => {
		const user = userEvent.setup();
		const hoverManager = newHoverManager();
		rtl.render(<TableDataCornerTopLeft columnAxisName='lon' hoverManager={hoverManager} rowAxisName='lat' />);

		await user.hover(screen.getByText('lat'));

		// Narrow row headers truncate the names, so the hover spells out which is which.
		expect(screen.getByText('lon')).toBeInTheDocument();
		expect(hoverManager.showHover).toHaveBeenCalledWith(expect.any(HTMLElement), 'Rows: lat, Columns: lon');
	});

	it('names only the row axis when there is no column axis', async () => {
		const user = userEvent.setup();
		const hoverManager = newHoverManager();
		rtl.render(<TableDataCornerTopLeft hoverManager={hoverManager} rowAxisName='time' />);

		await user.hover(screen.getByText('time'));

		expect(hoverManager.showHover).toHaveBeenCalledWith(expect.any(HTMLElement), 'Rows: time');
	});
});

/**
 * Builds the backend state of a 2-D xarray DataArray with dims (lat, lon). The instance reads the
 * table shape, the supported features, and the axis names from it, so the rest is filler.
 */
function dataArrayBackendState(): BackendState {
	return {
		display_name: 'sst (lat, lon)',
		table_shape: { num_rows: 3, num_columns: 2 },
		table_unfiltered_shape: { num_rows: 3, num_columns: 2 },
		has_row_labels: true,
		column_filters: [],
		row_filters: [],
		sort_keys: [],
		supported_features: {
			search_schema: { support_status: SupportStatus.Unsupported, supported_types: [] },
			set_column_filters: { support_status: SupportStatus.Unsupported, supported_types: [] },
			set_row_filters: { support_status: SupportStatus.Unsupported, supports_conditions: SupportStatus.Unsupported, supported_types: [] },
			get_column_profiles: { support_status: SupportStatus.Unsupported, supported_types: [] },
			export_data_selection: { support_status: SupportStatus.Unsupported, supported_formats: [] },
			set_sort_columns: { support_status: SupportStatus.Unsupported },
			convert_to_code: { support_status: SupportStatus.Unsupported }
		},
		row_axis_name: 'lat',
		column_axis_name: 'lon'
	};
}

describe('DataGridCornerTopLeft with the table data corner content', () => {
	const ctx = createTestContainer()
		.withReactServices()
		.stub(IConfigurationService, new TestConfigurationService())
		.stub(IHoverService, NullHoverService)
		.build();
	const rtl = setupRTLRenderer(() => ctx.reactServices);

	let cache: TableDataCache;
	let instance: TableDataDataGridInstance;

	beforeEach(async () => {
		// TableDataDataGridInstance reads PositronReactServices.services (static singleton) in its
		// constructor. Bridge the builder-configured DI container to the singleton.
		PositronReactServices.services = ctx.reactServices;

		const client: Partial<DataExplorerClientInstance> = {
			cachedBackendState: dataArrayBackendState(),
			onDidClose: new Emitter<void>().event,
			onDidSchemaUpdate: new Emitter<SchemaUpdateEvent>().event,
			onDidDataUpdate: new Emitter<void>().event,
			onDidUpdateBackendState: new Emitter<BackendState>().event,
			getBackendState: vi.fn().mockResolvedValue(dataArrayBackendState()),
			getSchema: vi.fn(async (indices: number[]): Promise<TableSchema> => ({
				columns: indices.map(i => getColumnSchema(`${i}`, i, 'float64', ColumnDisplayType.Floating))
			})),
			getDataValues: vi.fn(async (columns: ColumnSelection[]): Promise<TableData> => ({
				columns: columns.map(() => ['1'])
			})),
			getRowLabels: vi.fn(async (): Promise<TableRowLabels> => ({ row_labels: [['10.0']] })),
			getSupportedFeatures: vi.fn().mockReturnValue(dataArrayBackendState().supported_features),
		};
		cache = new TableDataCache(client as DataExplorerClientInstance);
		instance = new TableDataDataGridInstance(client as DataExplorerClientInstance, cache);
		await instance.setVisible(true);
	});

	afterEach(() => {
		instance.dispose();
		cache.dispose();
		PositronReactServices.services = undefined!;
	});

	it('shows the axis names hover, not the scroll hover, over the axis names', async () => {
		const user = userEvent.setup();
		const showHover = vi.spyOn(instance.hoverManager, 'showHover');
		rtl.render(
			<PositronDataGridContextProvider instance={instance}>
				<DataGridCornerTopLeft onClick={vi.fn()} />
			</PositronDataGridContextProvider>
		);

		await user.hover(screen.getByText('lon'));

		expect(showHover.mock.calls.map(([, content]) => content)).toEqual(['Rows: lat, Columns: lon']);
	});

	it('still scrolls to the top left when the axis names are clicked', async () => {
		const user = userEvent.setup();
		const onClick = vi.fn();
		rtl.render(
			<PositronDataGridContextProvider instance={instance}>
				<DataGridCornerTopLeft onClick={onClick} />
			</PositronDataGridContextProvider>
		);

		await user.click(screen.getByText('lat'));

		expect(onClick).toHaveBeenCalledTimes(1);
	});
});
