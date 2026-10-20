/*---------------------------------------------------------------------------------------------
 *  Copyright (C) 2026 Posit Software, PBC. All rights reserved.
 *  Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
 *--------------------------------------------------------------------------------------------*/

/// <reference types="vitest/globals" />

// Testing libraries.
import { screen } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';

// Other dependencies.
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

	it('keeps its hover from reaching the enclosing corner', async () => {
		const user = userEvent.setup();
		const hoverManager = newHoverManager();
		const cornerMouseOver = vi.fn();
		rtl.render(
			<div role='presentation' onMouseOver={cornerMouseOver}>
				<TableDataCornerTopLeft columnAxisName='lon' hoverManager={hoverManager} rowAxisName='lat' />
			</div>
		);

		await user.hover(screen.getByText('lon'));

		// The corner shows "Scroll to top-left" on mouseover, which would replace the axis names.
		expect(cornerMouseOver).not.toHaveBeenCalled();
	});
});
