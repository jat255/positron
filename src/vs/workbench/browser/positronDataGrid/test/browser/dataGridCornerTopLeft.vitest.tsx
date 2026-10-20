/*---------------------------------------------------------------------------------------------
 *  Copyright (C) 2026 Posit Software, PBC. All rights reserved.
 *  Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
 *--------------------------------------------------------------------------------------------*/

/// <reference types="vitest/globals" />

// React.
import { JSX } from 'react';

// Testing libraries.
import { screen } from '@testing-library/react';
import { userEvent } from '@testing-library/user-event';

// Other dependencies.
import { setupRTLRenderer } from '../../../../../test/vitest/reactTestingLibrary.js';
import { createTestContainer } from '../../../../../test/vitest/positronTestContainer.js';
import { DataGridInstance } from '../../classes/dataGridInstance.js';
import { PositronDataGridContextProvider } from '../../positronDataGridContext.js';
import { DataGridCornerTopLeft } from '../../components/dataGridCornerTopLeft.js';

/**
 * A minimal concrete data grid instance whose corner content is supplied by the test.
 */
class TestDataGridInstance extends DataGridInstance {
	constructor(private readonly _cornerContent?: JSX.Element) {
		super({
			columnHeaders: true,
			columnHeadersHeight: 30,
			rowHeaders: true,
			rowHeadersWidth: 50,
			rowHeadersResize: true,
			defaultColumnWidth: 100,
			defaultRowHeight: 20,
			columnResize: false,
			rowResize: false,
			columnPinning: false,
			rowPinning: false,
			horizontalScrollbar: false,
			verticalScrollbar: false,
			useEditorFont: false,
			automaticLayout: true,
			cellBorders: false,
			internalCursor: false,
		});
	}

	get columns() {
		return 0;
	}

	get rows() {
		return 0;
	}

	override cornerTopLeft() {
		return this._cornerContent;
	}

	cell() {
		return undefined;
	}
}

describe('DataGridCornerTopLeft', () => {
	const ctx = createTestContainer().withReactServices().build();
	const rtl = setupRTLRenderer(() => ctx.reactServices);

	function renderCorner(cornerContent: JSX.Element | undefined, onClick = vi.fn()) {
		const instance = new TestDataGridInstance(cornerContent);
		ctx.disposables.add(instance);
		rtl.render(
			<PositronDataGridContextProvider instance={instance}>
				<DataGridCornerTopLeft onClick={onClick} />
			</PositronDataGridContextProvider>
		);
		return onClick;
	}

	it('renders the corner content the instance supplies, alongside the splitter', () => {
		renderCorner(<span>lat by lon</span>);

		// The content takes the corner's content column; the splitter that resizes the row headers
		// must still be there next to it.
		expect(screen.getByText('lat by lon')).toBeInTheDocument();
		expect(screen.getByTestId('vertical-splitter')).toBeInTheDocument();
	});

	it('scrolls to the top left when the corner content is clicked', async () => {
		const user = userEvent.setup();
		const onClick = renderCorner(<span>lat by lon</span>);

		await user.click(screen.getByText('lat by lon'));

		expect(onClick).toHaveBeenCalledTimes(1);
	});

	it('renders only the splitter when the instance supplies no corner content', () => {
		renderCorner(undefined);

		expect(screen.queryByTestId('data-grid-corner-top-left-content')).not.toBeInTheDocument();
		expect(screen.getByTestId('vertical-splitter')).toBeInTheDocument();
	});

	it('offers the scroll tooltip only on an empty corner', () => {
		// Corner content shows its own hover; an empty corner keeps the "Scroll to top-left" one.
		renderCorner(undefined);

		expect(screen.getByTitle('Scroll to top-left')).toBeInTheDocument();
	});

	it('leaves the hover to the corner content when there is some', () => {
		renderCorner(<span>lat by lon</span>);

		expect(screen.queryByTitle('Scroll to top-left')).not.toBeInTheDocument();
	});
});
