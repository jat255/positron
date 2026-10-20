/*---------------------------------------------------------------------------------------------
 *  Copyright (C) 2026 Posit Software, PBC. All rights reserved.
 *  Licensed under the Elastic License 2.0. See LICENSE.txt for license information.
 *--------------------------------------------------------------------------------------------*/

// CSS.
import './tableDataCornerTopLeft.css';

// React.
import { useEffect, useRef } from 'react';

// Other dependencies.
import { localize } from '../../../../../nls.js';
import { PositronActionBarHoverManager } from '../../../../../platform/positronActionBar/browser/positronActionBarHoverManager.js';

/**
 * TableDataCornerTopLeftProps interface.
 */
interface TableDataCornerTopLeftProps {
	rowAxisName?: string;
	columnAxisName?: string;
	hoverManager: PositronActionBarHoverManager;
}

/**
 * Builds the hover text that spells out which axis each name belongs to.
 * @param rowAxisName The row axis name, if any.
 * @param columnAxisName The column axis name, if any.
 * @returns The hover text.
 */
const axisNamesHoverText = (rowAxisName?: string, columnAxisName?: string) => {
	if (rowAxisName !== undefined && columnAxisName !== undefined) {
		return localize(
			'positron.dataExplorer.axisNames.rowsAndColumns',
			"Rows: {0}, Columns: {1}",
			rowAxisName,
			columnAxisName
		);
	} else if (rowAxisName !== undefined) {
		return localize('positron.dataExplorer.axisNames.rows', "Rows: {0}", rowAxisName);
	} else {
		return localize('positron.dataExplorer.axisNames.columns', "Columns: {0}", columnAxisName);
	}
};

/**
 * TableDataCornerTopLeft component. Names the dimensions the grid's rows and columns are indexed
 * by, crosstab style: the column axis name at the top right, the row axis name at the bottom left.
 * @param props A TableDataCornerTopLeftProps that contains the component properties.
 * @returns The rendered component.
 */
export const TableDataCornerTopLeft = (props: TableDataCornerTopLeftProps) => {
	// Reference hooks.
	const ref = useRef<HTMLDivElement>(undefined!);

	// The names truncate when the row headers are narrow, so the hover gives them in full.
	const hoverText = axisNamesHoverText(props.rowAxisName, props.columnAxisName);

	// Show the hover while the mouse is over the names. The mouseover listener stops propagation
	// so the corner's own "Scroll to top-left" hover does not replace this one.
	useEffect(() => {
		const element = ref.current;
		const hoverManager = props.hoverManager;
		const showHover = (e: MouseEvent) => {
			e.stopPropagation();
			hoverManager.showHover(element, hoverText);
		};
		const hideHover = () => hoverManager.hideHover();
		element.addEventListener('mouseover', showHover);
		element.addEventListener('mouseleave', hideHover);
		return () => {
			element.removeEventListener('mouseover', showHover);
			element.removeEventListener('mouseleave', hideHover);
		};
	}, [hoverText, props.hoverManager]);

	// Render.
	return (
		<div ref={ref} className='table-data-corner-top-left'>
			{props.columnAxisName !== undefined &&
				<div className='axis-name column-axis-name'>
					<span className='name'>{props.columnAxisName}</span>
					<span aria-hidden='true' className='codicon codicon-arrow-right' />
				</div>
			}
			{props.rowAxisName !== undefined &&
				<div className='axis-name row-axis-name'>
					<span className='name'>{props.rowAxisName}</span>
					<span aria-hidden='true' className='codicon codicon-arrow-down' />
				</div>
			}
		</div>
	);
};
