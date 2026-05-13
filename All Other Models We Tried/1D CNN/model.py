<!DOCTYPE html>

<html lang="en">
<head><meta charset="utf-8"/>
<meta content="width=device-width, initial-scale=1.0" name="viewport"/>
<title>1DCNN</title><script src="https://cdnjs.cloudflare.com/ajax/libs/require.js/2.1.10/require.min.js"></script>
<style type="text/css">
    pre { line-height: 125%; }
td.linenos .normal { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
span.linenos { color: inherit; background-color: transparent; padding-left: 5px; padding-right: 5px; }
td.linenos .special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
span.linenos.special { color: #000000; background-color: #ffffc0; padding-left: 5px; padding-right: 5px; }
.highlight .hll { background-color: var(--jp-cell-editor-active-background) }
.highlight { background: var(--jp-cell-editor-background); color: var(--jp-mirror-editor-variable-color) }
.highlight .c { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment */
.highlight .err { color: var(--jp-mirror-editor-error-color) } /* Error */
.highlight .k { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword */
.highlight .o { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator */
.highlight .p { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation */
.highlight .ch { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Hashbang */
.highlight .cm { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Multiline */
.highlight .cp { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Preproc */
.highlight .cpf { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.PreprocFile */
.highlight .c1 { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Single */
.highlight .cs { color: var(--jp-mirror-editor-comment-color); font-style: italic } /* Comment.Special */
.highlight .kc { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Constant */
.highlight .kd { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Declaration */
.highlight .kn { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Namespace */
.highlight .kp { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Pseudo */
.highlight .kr { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Reserved */
.highlight .kt { color: var(--jp-mirror-editor-keyword-color); font-weight: bold } /* Keyword.Type */
.highlight .m { color: var(--jp-mirror-editor-number-color) } /* Literal.Number */
.highlight .s { color: var(--jp-mirror-editor-string-color) } /* Literal.String */
.highlight .ow { color: var(--jp-mirror-editor-operator-color); font-weight: bold } /* Operator.Word */
.highlight .pm { color: var(--jp-mirror-editor-punctuation-color) } /* Punctuation.Marker */
.highlight .w { color: var(--jp-mirror-editor-variable-color) } /* Text.Whitespace */
.highlight .mb { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Bin */
.highlight .mf { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Float */
.highlight .mh { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Hex */
.highlight .mi { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer */
.highlight .mo { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Oct */
.highlight .sa { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Affix */
.highlight .sb { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Backtick */
.highlight .sc { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Char */
.highlight .dl { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Delimiter */
.highlight .sd { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Doc */
.highlight .s2 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Double */
.highlight .se { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Escape */
.highlight .sh { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Heredoc */
.highlight .si { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Interpol */
.highlight .sx { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Other */
.highlight .sr { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Regex */
.highlight .s1 { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Single */
.highlight .ss { color: var(--jp-mirror-editor-string-color) } /* Literal.String.Symbol */
.highlight .il { color: var(--jp-mirror-editor-number-color) } /* Literal.Number.Integer.Long */
  </style>
<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
 * Mozilla scrollbar styling
 */

/* use standard opaque scrollbars for most nodes */
[data-jp-theme-scrollbars='true'] {
  scrollbar-color: rgb(var(--jp-scrollbar-thumb-color))
    var(--jp-scrollbar-background-color);
}

/* for code nodes, use a transparent style of scrollbar. These selectors
 * will match lower in the tree, and so will override the above */
[data-jp-theme-scrollbars='true'] .CodeMirror-hscrollbar,
[data-jp-theme-scrollbars='true'] .CodeMirror-vscrollbar {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
}

/* tiny scrollbar */

.jp-scrollbar-tiny {
  scrollbar-color: rgba(var(--jp-scrollbar-thumb-color), 0.5) transparent;
  scrollbar-width: thin;
}

/* tiny scrollbar */

.jp-scrollbar-tiny::-webkit-scrollbar,
.jp-scrollbar-tiny::-webkit-scrollbar-corner {
  background-color: transparent;
  height: 4px;
  width: 4px;
}

.jp-scrollbar-tiny::-webkit-scrollbar-thumb {
  background: rgba(var(--jp-scrollbar-thumb-color), 0.5);
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:horizontal {
  border-left: 0 solid transparent;
  border-right: 0 solid transparent;
}

.jp-scrollbar-tiny::-webkit-scrollbar-track:vertical {
  border-top: 0 solid transparent;
  border-bottom: 0 solid transparent;
}

/*
 * Lumino
 */

.lm-ScrollBar[data-orientation='horizontal'] {
  min-height: 16px;
  max-height: 16px;
  min-width: 45px;
  border-top: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] {
  min-width: 16px;
  max-width: 16px;
  min-height: 45px;
  border-left: 1px solid #a0a0a0;
}

.lm-ScrollBar-button {
  background-color: #f0f0f0;
  background-position: center center;
  min-height: 15px;
  max-height: 15px;
  min-width: 15px;
  max-width: 15px;
}

.lm-ScrollBar-button:hover {
  background-color: #dadada;
}

.lm-ScrollBar-button.lm-mod-active {
  background-color: #cdcdcd;
}

.lm-ScrollBar-track {
  background: #f0f0f0;
}

.lm-ScrollBar-thumb {
  background: #cdcdcd;
}

.lm-ScrollBar-thumb:hover {
  background: #bababa;
}

.lm-ScrollBar-thumb.lm-mod-active {
  background: #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal'] .lm-ScrollBar-thumb {
  height: 100%;
  min-width: 15px;
  border-left: 1px solid #a0a0a0;
  border-right: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='vertical'] .lm-ScrollBar-thumb {
  width: 100%;
  min-height: 15px;
  border-top: 1px solid #a0a0a0;
  border-bottom: 1px solid #a0a0a0;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-left);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='horizontal']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-right);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='decrement'] {
  background-image: var(--jp-icon-caret-up);
  background-size: 17px;
}

.lm-ScrollBar[data-orientation='vertical']
  .lm-ScrollBar-button[data-action='increment'] {
  background-image: var(--jp-icon-caret-down);
  background-size: 17px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Widget {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
}

.lm-Widget.lm-mod-hidden {
  display: none !important;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.lm-AccordionPanel[data-orientation='horizontal'] > .lm-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  display: flex;
  flex-direction: column;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-CommandPalette-search {
  flex: 0 0 auto;
}

.lm-CommandPalette-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  min-height: 0;
  overflow: auto;
  list-style-type: none;
}

.lm-CommandPalette-header {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-CommandPalette-item {
  display: flex;
  flex-direction: row;
}

.lm-CommandPalette-itemIcon {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemContent {
  flex: 1 1 auto;
  overflow: hidden;
}

.lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemLabel {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.lm-close-icon {
  border: 1px solid transparent;
  background-color: transparent;
  position: absolute;
  z-index: 1;
  right: 3%;
  top: 0;
  bottom: 0;
  margin: auto;
  padding: 7px 0;
  display: none;
  vertical-align: middle;
  outline: 0;
  cursor: pointer;
}
.lm-close-icon:after {
  content: 'X';
  display: block;
  width: 15px;
  height: 15px;
  text-align: center;
  color: #000;
  font-weight: normal;
  font-size: 12px;
  cursor: pointer;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-DockPanel {
  z-index: 0;
}

.lm-DockPanel-widget {
  z-index: 0;
}

.lm-DockPanel-tabBar {
  z-index: 1;
}

.lm-DockPanel-handle {
  z-index: 2;
}

.lm-DockPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-DockPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-DockPanel-handle[data-orientation='horizontal'] {
  cursor: ew-resize;
}

.lm-DockPanel-handle[data-orientation='vertical'] {
  cursor: ns-resize;
}

.lm-DockPanel-handle[data-orientation='horizontal']:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-DockPanel-handle[data-orientation='vertical']:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

.lm-DockPanel-overlay {
  z-index: 3;
  box-sizing: border-box;
  pointer-events: none;
}

.lm-DockPanel-overlay.lm-mod-hidden {
  display: none !important;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-Menu {
  z-index: 10000;
  position: absolute;
  white-space: nowrap;
  overflow-x: hidden;
  overflow-y: auto;
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-Menu-content {
  margin: 0;
  padding: 0;
  display: table;
  list-style-type: none;
}

.lm-Menu-item {
  display: table-row;
}

.lm-Menu-item.lm-mod-hidden,
.lm-Menu-item.lm-mod-collapsed {
  display: none !important;
}

.lm-Menu-itemIcon,
.lm-Menu-itemSubmenuIcon {
  display: table-cell;
  text-align: center;
}

.lm-Menu-itemLabel {
  display: table-cell;
  text-align: left;
}

.lm-Menu-itemShortcut {
  display: table-cell;
  text-align: right;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-MenuBar {
  outline: none;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-MenuBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex-direction: row;
  list-style-type: none;
}

.lm-MenuBar-item {
  box-sizing: border-box;
}

.lm-MenuBar-itemIcon,
.lm-MenuBar-itemLabel {
  display: inline-block;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-ScrollBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-ScrollBar[data-orientation='horizontal'] {
  flex-direction: row;
}

.lm-ScrollBar[data-orientation='vertical'] {
  flex-direction: column;
}

.lm-ScrollBar-button {
  box-sizing: border-box;
  flex: 0 0 auto;
}

.lm-ScrollBar-track {
  box-sizing: border-box;
  position: relative;
  overflow: hidden;
  flex: 1 1 auto;
}

.lm-ScrollBar-thumb {
  box-sizing: border-box;
  position: absolute;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-SplitPanel-child {
  z-index: 0;
}

.lm-SplitPanel-handle {
  z-index: 1;
}

.lm-SplitPanel-handle.lm-mod-hidden {
  display: none !important;
}

.lm-SplitPanel-handle:after {
  position: absolute;
  top: 0;
  left: 0;
  width: 100%;
  height: 100%;
  content: '';
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle {
  cursor: ew-resize;
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle {
  cursor: ns-resize;
}

.lm-SplitPanel[data-orientation='horizontal'] > .lm-SplitPanel-handle:after {
  left: 50%;
  min-width: 8px;
  transform: translateX(-50%);
}

.lm-SplitPanel[data-orientation='vertical'] > .lm-SplitPanel-handle:after {
  top: 50%;
  min-height: 8px;
  transform: translateY(-50%);
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabBar {
  display: flex;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.lm-TabBar[data-orientation='horizontal'] {
  flex-direction: row;
  align-items: flex-end;
}

.lm-TabBar[data-orientation='vertical'] {
  flex-direction: column;
  align-items: flex-end;
}

.lm-TabBar-content {
  margin: 0;
  padding: 0;
  display: flex;
  flex: 1 1 auto;
  list-style-type: none;
}

.lm-TabBar[data-orientation='horizontal'] > .lm-TabBar-content {
  flex-direction: row;
}

.lm-TabBar[data-orientation='vertical'] > .lm-TabBar-content {
  flex-direction: column;
}

.lm-TabBar-tab {
  display: flex;
  flex-direction: row;
  box-sizing: border-box;
  overflow: hidden;
  touch-action: none; /* Disable native Drag/Drop */
}

.lm-TabBar-tabIcon,
.lm-TabBar-tabCloseIcon {
  flex: 0 0 auto;
}

.lm-TabBar-tabLabel {
  flex: 1 1 auto;
  overflow: hidden;
  white-space: nowrap;
}

.lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
}

.lm-TabBar-tab.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar-addButton.lm-mod-hidden {
  display: none !important;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab {
  position: relative;
}

.lm-TabBar.lm-mod-dragging[data-orientation='horizontal'] .lm-TabBar-tab {
  left: 0;
  transition: left 150ms ease;
}

.lm-TabBar.lm-mod-dragging[data-orientation='vertical'] .lm-TabBar-tab {
  top: 0;
  transition: top 150ms ease;
}

.lm-TabBar.lm-mod-dragging .lm-TabBar-tab.lm-mod-dragging {
  transition: none;
}

.lm-TabBar-tabLabel .lm-TabBar-tabInput {
  user-select: all;
  width: 100%;
  box-sizing: border-box;
  background: inherit;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-TabPanel-tabBar {
  z-index: 1;
}

.lm-TabPanel-stackedPanel {
  z-index: 0;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapse {
  display: flex;
  flex-direction: column;
  align-items: stretch;
}

.jp-Collapse-header {
  padding: 1px 12px;
  background-color: var(--jp-layout-color1);
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  align-items: center;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  text-transform: uppercase;
  user-select: none;
}

.jp-Collapser-icon {
  height: 16px;
}

.jp-Collapse-header-collapsed .jp-Collapser-icon {
  transform: rotate(-90deg);
  margin: auto 0;
}

.jp-Collapser-title {
  line-height: 25px;
}

.jp-Collapse-contents {
  padding: 0 12px;
  background-color: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* This file was auto-generated by ensureUiComponents() in @jupyterlab/buildutils */

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

/* Icons urls */

:root {
  --jp-icon-add-above: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzEzN18xOTQ5MikiPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGQ9Ik00Ljc1IDQuOTMwNjZINi42MjVWNi44MDU2NkM2LjYyNSA3LjAxMTkxIDYuNzkzNzUgNy4xODA2NiA3IDcuMTgwNjZDNy4yMDYyNSA3LjE4MDY2IDcuMzc1IDcuMDExOTEgNy4zNzUgNi44MDU2NlY0LjkzMDY2SDkuMjVDOS40NTYyNSA0LjkzMDY2IDkuNjI1IDQuNzYxOTEgOS42MjUgNC41NTU2NkM5LjYyNSA0LjM0OTQxIDkuNDU2MjUgNC4xODA2NiA5LjI1IDQuMTgwNjZINy4zNzVWMi4zMDU2NkM3LjM3NSAyLjA5OTQxIDcuMjA2MjUgMS45MzA2NiA3IDEuOTMwNjZDNi43OTM3NSAxLjkzMDY2IDYuNjI1IDIuMDk5NDEgNi42MjUgMi4zMDU2NlY0LjE4MDY2SDQuNzVDNC41NDM3NSA0LjE4MDY2IDQuMzc1IDQuMzQ5NDEgNC4zNzUgNC41NTU2NkM0LjM3NSA0Ljc2MTkxIDQuNTQzNzUgNC45MzA2NiA0Ljc1IDQuOTMwNjZaIiBmaWxsPSIjNjE2MTYxIiBzdHJva2U9IiM2MTYxNjEiIHN0cm9rZS13aWR0aD0iMC43Ii8+CjwvZz4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTExLjUgOS41VjExLjVMMi41IDExLjVWOS41TDExLjUgOS41Wk0xMiA4QzEyLjU1MjMgOCAxMyA4LjQ0NzcyIDEzIDlWMTJDMTMgMTIuNTUyMyAxMi41NTIzIDEzIDEyIDEzTDIgMTNDMS40NDc3MiAxMyAxIDEyLjU1MjMgMSAxMlY5QzEgOC40NDc3MiAxLjQ0NzcxIDggMiA4TDEyIDhaIiBmaWxsPSIjNjE2MTYxIi8+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzEzN18xOTQ5MiI+CjxyZWN0IGNsYXNzPSJqcC1pY29uMyIgd2lkdGg9IjYiIGhlaWdodD0iNiIgZmlsbD0id2hpdGUiIHRyYW5zZm9ybT0ibWF0cml4KC0xIDAgMCAxIDEwIDEuNTU1NjYpIi8+CjwvY2xpcFBhdGg+CjwvZGVmcz4KPC9zdmc+Cg==);
  --jp-icon-add-below: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPGcgY2xpcC1wYXRoPSJ1cmwoI2NsaXAwXzEzN18xOTQ5OCkiPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGQ9Ik05LjI1IDEwLjA2OTNMNy4zNzUgMTAuMDY5M0w3LjM3NSA4LjE5NDM0QzcuMzc1IDcuOTg4MDkgNy4yMDYyNSA3LjgxOTM0IDcgNy44MTkzNEM2Ljc5Mzc1IDcuODE5MzQgNi42MjUgNy45ODgwOSA2LjYyNSA4LjE5NDM0TDYuNjI1IDEwLjA2OTNMNC43NSAxMC4wNjkzQzQuNTQzNzUgMTAuMDY5MyA0LjM3NSAxMC4yMzgxIDQuMzc1IDEwLjQ0NDNDNC4zNzUgMTAuNjUwNiA0LjU0Mzc1IDEwLjgxOTMgNC43NSAxMC44MTkzTDYuNjI1IDEwLjgxOTNMNi42MjUgMTIuNjk0M0M2LjYyNSAxMi45MDA2IDYuNzkzNzUgMTMuMDY5MyA3IDEzLjA2OTNDNy4yMDYyNSAxMy4wNjkzIDcuMzc1IDEyLjkwMDYgNy4zNzUgMTIuNjk0M0w3LjM3NSAxMC44MTkzTDkuMjUgMTAuODE5M0M5LjQ1NjI1IDEwLjgxOTMgOS42MjUgMTAuNjUwNiA5LjYyNSAxMC40NDQzQzkuNjI1IDEwLjIzODEgOS40NTYyNSAxMC4wNjkzIDkuMjUgMTAuMDY5M1oiIGZpbGw9IiM2MTYxNjEiIHN0cm9rZT0iIzYxNjE2MSIgc3Ryb2tlLXdpZHRoPSIwLjciLz4KPC9nPgo8cGF0aCBjbGFzcz0ianAtaWNvbjMiIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMi41IDUuNUwyLjUgMy41TDExLjUgMy41TDExLjUgNS41TDIuNSA1LjVaTTIgN0MxLjQ0NzcyIDcgMSA2LjU1MjI4IDEgNkwxIDNDMSAyLjQ0NzcyIDEuNDQ3NzIgMiAyIDJMMTIgMkMxMi41NTIzIDIgMTMgMi40NDc3MiAxMyAzTDEzIDZDMTMgNi41NTIyOSAxMi41NTIzIDcgMTIgN0wyIDdaIiBmaWxsPSIjNjE2MTYxIi8+CjxkZWZzPgo8Y2xpcFBhdGggaWQ9ImNsaXAwXzEzN18xOTQ5OCI+CjxyZWN0IGNsYXNzPSJqcC1pY29uMyIgd2lkdGg9IjYiIGhlaWdodD0iNiIgZmlsbD0id2hpdGUiIHRyYW5zZm9ybT0ibWF0cml4KDEgMS43NDg0NmUtMDcgMS43NDg0NmUtMDcgLTEgNCAxMy40NDQzKSIvPgo8L2NsaXBQYXRoPgo8L2RlZnM+Cjwvc3ZnPgo=);
  --jp-icon-add: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDEzaC02djZoLTJ2LTZINXYtMmg2VjVoMnY2aDZ2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bell: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE2IDE2IiB2ZXJzaW9uPSIxLjEiPgogICA8cGF0aCBjbGFzcz0ianAtaWNvbjIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMzMzMzMzIgogICAgICBkPSJtOCAwLjI5Yy0xLjQgMC0yLjcgMC43My0zLjYgMS44LTEuMiAxLjUtMS40IDMuNC0xLjUgNS4yLTAuMTggMi4yLTAuNDQgNC0yLjMgNS4zbDAuMjggMS4zaDVjMC4wMjYgMC42NiAwLjMyIDEuMSAwLjcxIDEuNSAwLjg0IDAuNjEgMiAwLjYxIDIuOCAwIDAuNTItMC40IDAuNi0xIDAuNzEtMS41aDVsMC4yOC0xLjNjLTEuOS0wLjk3LTIuMi0zLjMtMi4zLTUuMy0wLjEzLTEuOC0wLjI2LTMuNy0xLjUtNS4yLTAuODUtMS0yLjItMS44LTMuNi0xLjh6bTAgMS40YzAuODggMCAxLjkgMC41NSAyLjUgMS4zIDAuODggMS4xIDEuMSAyLjcgMS4yIDQuNCAwLjEzIDEuNyAwLjIzIDMuNiAxLjMgNS4yaC0xMGMxLjEtMS42IDEuMi0zLjQgMS4zLTUuMiAwLjEzLTEuNyAwLjMtMy4zIDEuMi00LjQgMC41OS0wLjcyIDEuNi0xLjMgMi41LTEuM3ptLTAuNzQgMTJoMS41Yy0wLjAwMTUgMC4yOCAwLjAxNSAwLjc5LTAuNzQgMC43OS0wLjczIDAuMDAxNi0wLjcyLTAuNTMtMC43NC0wLjc5eiIgLz4KPC9zdmc+Cg==);
  --jp-icon-bug-dot: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiPgogICAgICAgIDxwYXRoIGZpbGwtcnVsZT0iZXZlbm9kZCIgY2xpcC1ydWxlPSJldmVub2RkIiBkPSJNMTcuMTkgOEgyMFYxMEgxNy45MUMxNy45NiAxMC4zMyAxOCAxMC42NiAxOCAxMVYxMkgyMFYxNEgxOC41SDE4VjE0LjAyNzVDMTUuNzUgMTQuMjc2MiAxNCAxNi4xODM3IDE0IDE4LjVDMTQgMTkuMjA4IDE0LjE2MzUgMTkuODc3OSAxNC40NTQ5IDIwLjQ3MzlDMTMuNzA2MyAyMC44MTE3IDEyLjg3NTcgMjEgMTIgMjFDOS43OCAyMSA3Ljg1IDE5Ljc5IDYuODEgMThINFYxNkg2LjA5QzYuMDQgMTUuNjcgNiAxNS4zNCA2IDE1VjE0SDRWMTJINlYxMUM2IDEwLjY2IDYuMDQgMTAuMzMgNi4wOSAxMEg0VjhINi44MUM3LjI2IDcuMjIgNy44OCA2LjU1IDguNjIgNi4wNEw3IDQuNDFMOC40MSAzTDEwLjU5IDUuMTdDMTEuMDQgNS4wNiAxMS41MSA1IDEyIDVDMTIuNDkgNSAxMi45NiA1LjA2IDEzLjQyIDUuMTdMMTUuNTkgM0wxNyA0LjQxTDE1LjM3IDYuMDRDMTYuMTIgNi41NSAxNi43NCA3LjIyIDE3LjE5IDhaTTEwIDE2SDE0VjE0SDEwVjE2Wk0xMCAxMkgxNFYxMEgxMFYxMloiIGZpbGw9IiM2MTYxNjEiLz4KICAgICAgICA8cGF0aCBkPSJNMjIgMTguNUMyMiAyMC40MzMgMjAuNDMzIDIyIDE4LjUgMjJDMTYuNTY3IDIyIDE1IDIwLjQzMyAxNSAxOC41QzE1IDE2LjU2NyAxNi41NjcgMTUgMTguNSAxNUMyMC40MzMgMTUgMjIgMTYuNTY3IDIyIDE4LjVaIiBmaWxsPSIjNjE2MTYxIi8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-bug: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yMCA4aC0yLjgxYy0uNDUtLjc4LTEuMDctMS40NS0xLjgyLTEuOTZMMTcgNC40MSAxNS41OSAzbC0yLjE3IDIuMTdDMTIuOTYgNS4wNiAxMi40OSA1IDEyIDVjLS40OSAwLS45Ni4wNi0xLjQxLjE3TDguNDEgMyA3IDQuNDFsMS42MiAxLjYzQzcuODggNi41NSA3LjI2IDcuMjIgNi44MSA4SDR2MmgyLjA5Yy0uMDUuMzMtLjA5LjY2LS4wOSAxdjFINHYyaDJ2MWMwIC4zNC4wNC42Ny4wOSAxSDR2MmgyLjgxYzEuMDQgMS43OSAyLjk3IDMgNS4xOSAzczQuMTUtMS4yMSA1LjE5LTNIMjB2LTJoLTIuMDljLjA1LS4zMy4wOS0uNjYuMDktMXYtMWgydi0yaC0ydi0xYzAtLjM0LS4wNC0uNjctLjA5LTFIMjBWOHptLTYgOGgtNHYtMmg0djJ6bTAtNGgtNHYtMmg0djJ6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-build: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE0LjkgMTcuNDVDMTYuMjUgMTcuNDUgMTcuMzUgMTYuMzUgMTcuMzUgMTVDMTcuMzUgMTMuNjUgMTYuMjUgMTIuNTUgMTQuOSAxMi41NUMxMy41NCAxMi41NSAxMi40NSAxMy42NSAxMi40NSAxNUMxMi40NSAxNi4zNSAxMy41NCAxNy40NSAxNC45IDE3LjQ1Wk0yMC4xIDE1LjY4TDIxLjU4IDE2Ljg0QzIxLjcxIDE2Ljk1IDIxLjc1IDE3LjEzIDIxLjY2IDE3LjI5TDIwLjI2IDE5LjcxQzIwLjE3IDE5Ljg2IDIwIDE5LjkyIDE5LjgzIDE5Ljg2TDE4LjA5IDE5LjE2QzE3LjczIDE5LjQ0IDE3LjMzIDE5LjY3IDE2LjkxIDE5Ljg1TDE2LjY0IDIxLjdDMTYuNjIgMjEuODcgMTYuNDcgMjIgMTYuMyAyMkgxMy41QzEzLjMyIDIyIDEzLjE4IDIxLjg3IDEzLjE1IDIxLjdMMTIuODkgMTkuODVDMTIuNDYgMTkuNjcgMTIuMDcgMTkuNDQgMTEuNzEgMTkuMTZMOS45NjAwMiAxOS44NkM5LjgxMDAyIDE5LjkyIDkuNjIwMDIgMTkuODYgOS41NDAwMiAxOS43MUw4LjE0MDAyIDE3LjI5QzguMDUwMDIgMTcuMTMgOC4wOTAwMiAxNi45NSA4LjIyMDAyIDE2Ljg0TDkuNzAwMDIgMTUuNjhMOS42NTAwMSAxNUw5LjcwMDAyIDE0LjMxTDguMjIwMDIgMTMuMTZDOC4wOTAwMiAxMy4wNSA4LjA1MDAyIDEyLjg2IDguMTQwMDIgMTIuNzFMOS41NDAwMiAxMC4yOUM5LjYyMDAyIDEwLjEzIDkuODEwMDIgMTAuMDcgOS45NjAwMiAxMC4xM0wxMS43MSAxMC44NEMxMi4wNyAxMC41NiAxMi40NiAxMC4zMiAxMi44OSAxMC4xNUwxMy4xNSA4LjI4OTk4QzEzLjE4IDguMTI5OTggMTMuMzIgNy45OTk5OCAxMy41IDcuOTk5OThIMTYuM0MxNi40NyA3Ljk5OTk4IDE2LjYyIDguMTI5OTggMTYuNjQgOC4yODk5OEwxNi45MSAxMC4xNUMxNy4zMyAxMC4zMiAxNy43MyAxMC41NiAxOC4wOSAxMC44NEwxOS44MyAxMC4xM0MyMCAxMC4wNyAyMC4xNyAxMC4xMyAyMC4yNiAxMC4yOUwyMS42NiAxMi43MUMyMS43NSAxMi44NiAyMS43MSAxMy4wNSAyMS41OCAxMy4xNkwyMC4xIDE0LjMxTDIwLjE1IDE1TDIwLjEgMTUuNjhaIi8+CiAgICA8cGF0aCBkPSJNNy4zMjk2NiA3LjQ0NDU0QzguMDgzMSA3LjAwOTU0IDguMzM5MzIgNi4wNTMzMiA3LjkwNDMyIDUuMjk5ODhDNy40NjkzMiA0LjU0NjQzIDYuNTA4MSA0LjI4MTU2IDUuNzU0NjYgNC43MTY1NkM1LjM5MTc2IDQuOTI2MDggNS4xMjY5NSA1LjI3MTE4IDUuMDE4NDkgNS42NzU5NEM0LjkxMDA0IDYuMDgwNzEgNC45NjY4MiA2LjUxMTk4IDUuMTc2MzQgNi44NzQ4OEM1LjYxMTM0IDcuNjI4MzIgNi41NzYyMiA3Ljg3OTU0IDcuMzI5NjYgNy40NDQ1NFpNOS42NTcxOCA0Ljc5NTkzTDEwLjg2NzIgNC45NTE3OUMxMC45NjI4IDQuOTc3NDEgMTEuMDQwMiA1LjA3MTMzIDExLjAzODIgNS4xODc5M0wxMS4wMzg4IDYuOTg4OTNDMTEuMDQ1NSA3LjEwMDU0IDEwLjk2MTYgNy4xOTUxOCAxMC44NTUgNy4yMTA1NEw5LjY2MDAxIDcuMzgwODNMOS4yMzkxNSA4LjEzMTg4TDkuNjY5NjEgOS4yNTc0NUM5LjcwNzI5IDkuMzYyNzEgOS42NjkzNCA5LjQ3Njk5IDkuNTc0MDggOS41MzE5OUw4LjAxNTIzIDEwLjQzMkM3LjkxMTMxIDEwLjQ5MiA3Ljc5MzM3IDEwLjQ2NzcgNy43MjEwNSAxMC4zODI0TDYuOTg3NDggOS40MzE4OEw2LjEwOTMxIDkuNDMwODNMNS4zNDcwNCAxMC4zOTA1QzUuMjg5MDkgMTAuNDcwMiA1LjE3MzgzIDEwLjQ5MDUgNS4wNzE4NyAxMC40MzM5TDMuNTEyNDUgOS41MzI5M0MzLjQxMDQ5IDkuNDc2MzMgMy4zNzY0NyA5LjM1NzQxIDMuNDEwNzUgOS4yNTY3OUwzLjg2MzQ3IDguMTQwOTNMMy42MTc0OSA3Ljc3NDg4TDMuNDIzNDcgNy4zNzg4M0wyLjIzMDc1IDcuMjEyOTdDMi4xMjY0NyA3LjE5MjM1IDIuMDQwNDkgNy4xMDM0MiAyLjA0MjQ1IDYuOTg2ODJMMi4wNDE4NyA1LjE4NTgyQzIuMDQzODMgNS4wNjkyMiAyLjExOTA5IDQuOTc5NTggMi4yMTcwNCA0Ljk2OTIyTDMuNDIwNjUgNC43OTM5M0wzLjg2NzQ5IDQuMDI3ODhMMy40MTEwNSAyLjkxNzMxQzMuMzczMzcgMi44MTIwNCAzLjQxMTMxIDIuNjk3NzYgMy41MTUyMyAyLjYzNzc2TDUuMDc0MDggMS43Mzc3NkM1LjE2OTM0IDEuNjgyNzYgNS4yODcyOSAxLjcwNzA0IDUuMzU5NjEgMS43OTIzMUw2LjExOTE1IDIuNzI3ODhMNi45ODAwMSAyLjczODkzTDcuNzI0OTYgMS43ODkyMkM3Ljc5MTU2IDEuNzA0NTggNy45MTU0OCAxLjY3OTIyIDguMDA4NzkgMS43NDA4Mkw5LjU2ODIxIDIuNjQxODJDOS42NzAxNyAyLjY5ODQyIDkuNzEyODUgMi44MTIzNCA5LjY4NzIzIDIuOTA3OTdMOS4yMTcxOCA0LjAzMzgzTDkuNDYzMTYgNC4zOTk4OEw5LjY1NzE4IDQuNzk1OTNaIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iOS45LDEzLjYgMy42LDcuNCA0LjQsNi42IDkuOSwxMi4yIDE1LjQsNi43IDE2LjEsNy40ICIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNS45TDksOS43bDMuOC0zLjhsMS4yLDEuMmwtNC45LDVsLTQuOS01TDUuMiw1Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-caret-down: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik01LjIsNy41TDksMTEuMmwzLjgtMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-left: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik0xMC44LDEyLjhMNy4xLDlsMy44LTMuOGwwLDcuNkgxMC44eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-right: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiIHNoYXBlLXJlbmRlcmluZz0iZ2VvbWV0cmljUHJlY2lzaW9uIj4KICAgIDxwYXRoIGQ9Ik03LjIsNS4yTDEwLjksOWwtMy44LDMuOFY1LjJINy4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-caret-up-empty-thin: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwb2x5Z29uIGNsYXNzPSJzdDEiIHBvaW50cz0iMTUuNCwxMy4zIDkuOSw3LjcgNC40LDEzLjIgMy42LDEyLjUgOS45LDYuMyAxNi4xLDEyLjYgIi8+Cgk8L2c+Cjwvc3ZnPgo=);
  --jp-icon-caret-up: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSIgc2hhcGUtcmVuZGVyaW5nPSJnZW9tZXRyaWNQcmVjaXNpb24iPgoJCTxwYXRoIGQ9Ik01LjIsMTAuNUw5LDYuOGwzLjgsMy44SDUuMnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-case-sensitive: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWFjY2VudDIiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTcuNiw4aDAuOWwzLjUsOGgtMS4xTDEwLDE0SDZsLTAuOSwySDRMNy42LDh6IE04LDkuMUw2LjQsMTNoMy4yTDgsOS4xeiIvPgogICAgPHBhdGggZD0iTTE2LjYsOS44Yy0wLjIsMC4xLTAuNCwwLjEtMC43LDAuMWMtMC4yLDAtMC40LTAuMS0wLjYtMC4yYy0wLjEtMC4xLTAuMi0wLjQtMC4yLTAuNyBjLTAuMywwLjMtMC42LDAuNS0wLjksMC43Yy0wLjMsMC4xLTAuNywwLjItMS4xLDAuMmMtMC4zLDAtMC41LDAtMC43LTAuMWMtMC4yLTAuMS0wLjQtMC4yLTAuNi0wLjNjLTAuMi0wLjEtMC4zLTAuMy0wLjQtMC41IGMtMC4xLTAuMi0wLjEtMC40LTAuMS0wLjdjMC0wLjMsMC4xLTAuNiwwLjItMC44YzAuMS0wLjIsMC4zLTAuNCwwLjQtMC41QzEyLDcsMTIuMiw2LjksMTIuNSw2LjhjMC4yLTAuMSwwLjUtMC4xLDAuNy0wLjIgYzAuMy0wLjEsMC41LTAuMSwwLjctMC4xYzAuMiwwLDAuNC0wLjEsMC42LTAuMWMwLjIsMCwwLjMtMC4xLDAuNC0wLjJjMC4xLTAuMSwwLjItMC4yLDAuMi0wLjRjMC0xLTEuMS0xLTEuMy0xIGMtMC40LDAtMS40LDAtMS40LDEuMmgtMC45YzAtMC40LDAuMS0wLjcsMC4yLTFjMC4xLTAuMiwwLjMtMC40LDAuNS0wLjZjMC4yLTAuMiwwLjUtMC4zLDAuOC0wLjNDMTMuMyw0LDEzLjYsNCwxMy45LDQgYzAuMywwLDAuNSwwLDAuOCwwLjFjMC4zLDAsMC41LDAuMSwwLjcsMC4yYzAuMiwwLjEsMC40LDAuMywwLjUsMC41QzE2LDUsMTYsNS4yLDE2LDUuNnYyLjljMCwwLjIsMCwwLjQsMCwwLjUgYzAsMC4xLDAuMSwwLjIsMC4zLDAuMmMwLjEsMCwwLjIsMCwwLjMsMFY5Ljh6IE0xNS4yLDYuOWMtMS4yLDAuNi0zLjEsMC4yLTMuMSwxLjRjMCwxLjQsMy4xLDEsMy4xLTAuNVY2Ljl6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik05IDE2LjE3TDQuODMgMTJsLTEuNDIgMS40MUw5IDE5IDIxIDdsLTEuNDEtMS40MXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-circle-empty: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDJDNi40NyAyIDIgNi40NyAyIDEyczQuNDcgMTAgMTAgMTAgMTAtNC40NyAxMC0xMFMxNy41MyAyIDEyIDJ6bTAgMThjLTQuNDEgMC04LTMuNTktOC04czMuNTktOCA4LTggOCAzLjU5IDggOC0zLjU5IDgtOCA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-circle: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iOSIgY3k9IjkiIHI9IjgiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-clear: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8bWFzayBpZD0iZG9udXRIb2xlIj4KICAgIDxyZWN0IHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgZmlsbD0id2hpdGUiIC8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSI4IiBmaWxsPSJibGFjayIvPgogIDwvbWFzaz4KCiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxyZWN0IGhlaWdodD0iMTgiIHdpZHRoPSIyIiB4PSIxMSIgeT0iMyIgdHJhbnNmb3JtPSJyb3RhdGUoMzE1LCAxMiwgMTIpIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIxMCIgbWFzaz0idXJsKCNkb251dEhvbGUpIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-close: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1ub25lIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIGpwLWljb24zLWhvdmVyIiBmaWxsPSJub25lIj4KICAgIDxjaXJjbGUgY3g9IjEyIiBjeT0iMTIiIHI9IjExIi8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIGpwLWljb24tYWNjZW50Mi1ob3ZlciIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMTkgNi40MUwxNy41OSA1IDEyIDEwLjU5IDYuNDEgNSA1IDYuNDEgMTAuNTkgMTIgNSAxNy41OSA2LjQxIDE5IDEyIDEzLjQxIDE3LjU5IDE5IDE5IDE3LjU5IDEzLjQxIDEyeiIvPgogIDwvZz4KCiAgPGcgY2xhc3M9ImpwLWljb24tbm9uZSBqcC1pY29uLWJ1c3kiIGZpbGw9Im5vbmUiPgogICAgPGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code-check: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBzaGFwZS1yZW5kZXJpbmc9Imdlb21ldHJpY1ByZWNpc2lvbiI+CiAgICA8cGF0aCBkPSJNNi41OSwzLjQxTDIsOEw2LjU5LDEyLjZMOCwxMS4xOEw0LjgyLDhMOCw0LjgyTDYuNTksMy40MU0xMi40MSwzLjQxTDExLDQuODJMMTQuMTgsOEwxMSwxMS4xOEwxMi40MSwxMi42TDE3LDhMMTIuNDEsMy40MU0yMS41OSwxMS41OUwxMy41LDE5LjY4TDkuODMsMTZMOC40MiwxNy40MUwxMy41LDIyLjVMMjMsMTNMMjEuNTksMTEuNTlaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-code: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTExLjQgMTguNkw2LjggMTRMMTEuNCA5LjRMMTAgOEw0IDE0TDEwIDIwTDExLjQgMTguNlpNMTYuNiAxOC42TDIxLjIgMTRMMTYuNiA5LjRMMTggOEwyNCAxNEwxOCAyMEwxNi42IDE4LjZWMTguNloiLz4KCTwvZz4KPC9zdmc+Cg==);
  --jp-icon-collapse-all: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTggMmMxIDAgMTEgMCAxMiAwczIgMSAyIDJjMCAxIDAgMTEgMCAxMnMwIDItMiAyQzIwIDE0IDIwIDQgMjAgNFMxMCA0IDYgNGMwLTIgMS0yIDItMnoiIC8+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTE4IDhjMC0xLTEtMi0yLTJTNSA2IDQgNnMtMiAxLTIgMmMwIDEgMCAxMSAwIDEyczEgMiAyIDJjMSAwIDExIDAgMTIgMHMyLTEgMi0yYzAtMSAwLTExIDAtMTJ6bS0yIDB2MTJINFY4eiIgLz4KICAgICAgICA8cGF0aCBkPSJNNiAxM3YyaDh2LTJ6IiAvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-console: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwMCAyMDAiPgogIDxnIGNsYXNzPSJqcC1jb25zb2xlLWljb24tYmFja2dyb3VuZC1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMjg4RDEiPgogICAgPHBhdGggZD0iTTIwIDE5LjhoMTYwdjE1OS45SDIweiIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtY29uc29sZS1pY29uLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIj4KICAgIDxwYXRoIGQ9Ik0xMDUgMTI3LjNoNDB2MTIuOGgtNDB6TTUxLjEgNzdMNzQgOTkuOWwtMjMuMyAyMy4zIDEwLjUgMTAuNSAyMy4zLTIzLjNMOTUgOTkuOSA4NC41IDg5LjQgNjEuNiA2Ni41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copy: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTExLjksMUgzLjJDMi40LDEsMS43LDEuNywxLjcsMi41djEwLjJoMS41VjIuNWg4LjdWMXogTTE0LjEsMy45aC04Yy0wLjgsMC0xLjUsMC43LTEuNSwxLjV2MTAuMmMwLDAuOCwwLjcsMS41LDEuNSwxLjVoOCBjMC44LDAsMS41LTAuNywxLjUtMS41VjUuNEMxNS41LDQuNiwxNC45LDMuOSwxNC4xLDMuOXogTTE0LjEsMTUuNWgtOFY1LjRoOFYxNS41eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-copyright: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGVuYWJsZS1iYWNrZ3JvdW5kPSJuZXcgMCAwIDI0IDI0IiBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCI+CiAgPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0xMS44OCw5LjE0YzEuMjgsMC4wNiwxLjYxLDEuMTUsMS42MywxLjY2aDEuNzljLTAuMDgtMS45OC0xLjQ5LTMuMTktMy40NS0zLjE5QzkuNjQsNy42MSw4LDksOCwxMi4xNCBjMCwxLjk0LDAuOTMsNC4yNCwzLjg0LDQuMjRjMi4yMiwwLDMuNDEtMS42NSwzLjQ0LTIuOTVoLTEuNzljLTAuMDMsMC41OS0wLjQ1LDEuMzgtMS42MywxLjQ0QzEwLjU1LDE0LjgzLDEwLDEzLjgxLDEwLDEyLjE0IEMxMCw5LjI1LDExLjI4LDkuMTYsMTEuODgsOS4xNHogTTEyLDJDNi40OCwyLDIsNi40OCwyLDEyczQuNDgsMTAsMTAsMTBzMTAtNC40OCwxMC0xMFMxNy41MiwyLDEyLDJ6IE0xMiwyMGMtNC40MSwwLTgtMy41OS04LTggczMuNTktOCw4LThzOCwzLjU5LDgsOFMxNi40MSwyMCwxMiwyMHoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-cut: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkuNjQgNy42NGMuMjMtLjUuMzYtMS4wNS4zNi0xLjY0IDAtMi4yMS0xLjc5LTQtNC00UzIgMy43OSAyIDZzMS43OSA0IDQgNGMuNTkgMCAxLjE0LS4xMyAxLjY0LS4zNkwxMCAxMmwtMi4zNiAyLjM2QzcuMTQgMTQuMTMgNi41OSAxNCA2IDE0Yy0yLjIxIDAtNCAxLjc5LTQgNHMxLjc5IDQgNCA0IDQtMS43OSA0LTRjMC0uNTktLjEzLTEuMTQtLjM2LTEuNjRMMTIgMTRsNyA3aDN2LTFMOS42NCA3LjY0ek02IDhjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTAgMTJjLTEuMSAwLTItLjg5LTItMnMuOS0yIDItMiAyIC44OSAyIDItLjkgMi0yIDJ6bTYtNy41Yy0uMjggMC0uNS0uMjItLjUtLjVzLjIyLS41LjUtLjUuNS4yMi41LjUtLjIyLjUtLjUuNXpNMTkgM2wtNiA2IDIgMiA3LTdWM3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-delete: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2cHgiIGhlaWdodD0iMTZweCI+CiAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIiAvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjI2MjYyIiBkPSJNNiAxOWMwIDEuMS45IDIgMiAyaDhjMS4xIDAgMi0uOSAyLTJWN0g2djEyek0xOSA0aC0zLjVsLTEtMWgtNWwtMSAxSDV2MmgxNFY0eiIgLz4KPC9zdmc+Cg==);
  --jp-icon-download: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE5IDloLTRWM0g5djZINWw3IDcgNy03ek01IDE4djJoMTR2LTJINXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-duplicate: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBmaWxsLXJ1bGU9ImV2ZW5vZGQiIGNsaXAtcnVsZT0iZXZlbm9kZCIgZD0iTTIuNzk5OTggMC44NzVIOC44OTU4MkM5LjIwMDYxIDAuODc1IDkuNDQ5OTggMS4xMzkxNCA5LjQ0OTk4IDEuNDYxOThDOS40NDk5OCAxLjc4NDgyIDkuMjAwNjEgMi4wNDg5NiA4Ljg5NTgyIDIuMDQ4OTZIMy4zNTQxNUMzLjA0OTM2IDIuMDQ4OTYgMi43OTk5OCAyLjMxMzEgMi43OTk5OCAyLjYzNTk0VjkuNjc5NjlDMi43OTk5OCAxMC4wMDI1IDIuNTUwNjEgMTAuMjY2NyAyLjI0NTgyIDEwLjI2NjdDMS45NDEwMyAxMC4yNjY3IDEuNjkxNjUgMTAuMDAyNSAxLjY5MTY1IDkuNjc5NjlWMi4wNDg5NkMxLjY5MTY1IDEuNDAzMjggMi4xOTA0IDAuODc1IDIuNzk5OTggMC44NzVaTTUuMzY2NjUgMTEuOVY0LjU1SDExLjA4MzNWMTEuOUg1LjM2NjY1Wk00LjE0MTY1IDQuMTQxNjdDNC4xNDE2NSAzLjY5MDYzIDQuNTA3MjggMy4zMjUgNC45NTgzMiAzLjMyNUgxMS40OTE3QzExLjk0MjcgMy4zMjUgMTIuMzA4MyAzLjY5MDYzIDEyLjMwODMgNC4xNDE2N1YxMi4zMDgzQzEyLjMwODMgMTIuNzU5NCAxMS45NDI3IDEzLjEyNSAxMS40OTE3IDEzLjEyNUg0Ljk1ODMyQzQuNTA3MjggMTMuMTI1IDQuMTQxNjUgMTIuNzU5NCA0LjE0MTY1IDEyLjMwODNWNC4xNDE2N1oiIGZpbGw9IiM2MTYxNjEiLz4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNOS40MzU3NCA4LjI2NTA3SDguMzY0MzFWOS4zMzY1QzguMzY0MzEgOS40NTQzNSA4LjI2Nzg4IDkuNTUwNzggOC4xNTAwMiA5LjU1MDc4QzguMDMyMTcgOS41NTA3OCA3LjkzNTc0IDkuNDU0MzUgNy45MzU3NCA5LjMzNjVWOC4yNjUwN0g2Ljg2NDMxQzYuNzQ2NDUgOC4yNjUwNyA2LjY1MDAyIDguMTY4NjQgNi42NTAwMiA4LjA1MDc4QzYuNjUwMDIgNy45MzI5MiA2Ljc0NjQ1IDcuODM2NSA2Ljg2NDMxIDcuODM2NUg3LjkzNTc0VjYuNzY1MDdDNy45MzU3NCA2LjY0NzIxIDguMDMyMTcgNi41NTA3OCA4LjE1MDAyIDYuNTUwNzhDOC4yNjc4OCA2LjU1MDc4IDguMzY0MzEgNi42NDcyMSA4LjM2NDMxIDYuNzY1MDdWNy44MzY1SDkuNDM1NzRDOS41NTM2IDcuODM2NSA5LjY1MDAyIDcuOTMyOTIgOS42NTAwMiA4LjA1MDc4QzkuNjUwMDIgOC4xNjg2NCA5LjU1MzYgOC4yNjUwNyA5LjQzNTc0IDguMjY1MDdaIiBmaWxsPSIjNjE2MTYxIiBzdHJva2U9IiM2MTYxNjEiIHN0cm9rZS13aWR0aD0iMC41Ii8+Cjwvc3ZnPgo=);
  --jp-icon-edit: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMgMTcuMjVWMjFoMy43NUwxNy44MSA5Ljk0bC0zLjc1LTMuNzVMMyAxNy4yNXpNMjAuNzEgNy4wNGMuMzktLjM5LjM5LTEuMDIgMC0xLjQxbC0yLjM0LTIuMzRjLS4zOS0uMzktMS4wMi0uMzktMS40MSAwbC0xLjgzIDEuODMgMy43NSAzLjc1IDEuODMtMS44M3oiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-ellipses: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPGNpcmNsZSBjeD0iNSIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxMiIgY3k9IjEyIiByPSIyIi8+CiAgICA8Y2lyY2xlIGN4PSIxOSIgY3k9IjEyIiByPSIyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-error: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj48Y2lyY2xlIGN4PSIxMiIgY3k9IjE5IiByPSIyIi8+PHBhdGggZD0iTTEwIDNoNHYxMmgtNHoiLz48L2c+CjxwYXRoIGZpbGw9Im5vbmUiIGQ9Ik0wIDBoMjR2MjRIMHoiLz4KPC9zdmc+Cg==);
  --jp-icon-expand-all: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTggMmMxIDAgMTEgMCAxMiAwczIgMSAyIDJjMCAxIDAgMTEgMCAxMnMwIDItMiAyQzIwIDE0IDIwIDQgMjAgNFMxMCA0IDYgNGMwLTIgMS0yIDItMnoiIC8+CiAgICAgICAgPHBhdGgKICAgICAgICAgICAgZD0iTTE4IDhjMC0xLTEtMi0yLTJTNSA2IDQgNnMtMiAxLTIgMmMwIDEgMCAxMSAwIDEyczEgMiAyIDJjMSAwIDExIDAgMTIgMHMyLTEgMi0yYzAtMSAwLTExIDAtMTJ6bS0yIDB2MTJINFY4eiIgLz4KICAgICAgICA8cGF0aCBkPSJNMTEgMTBIOXYzSDZ2MmgzdjNoMnYtM2gzdi0yaC0zeiIgLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-extension: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwLjUgMTFIMTlWN2MwLTEuMS0uOS0yLTItMmgtNFYzLjVDMTMgMi4xMiAxMS44OCAxIDEwLjUgMVM4IDIuMTIgOCAzLjVWNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAydjMuOEgzLjVjMS40OSAwIDIuNyAxLjIxIDIuNyAyLjdzLTEuMjEgMi43LTIuNyAyLjdIMlYyMGMwIDEuMS45IDIgMiAyaDMuOHYtMS41YzAtMS40OSAxLjIxLTIuNyAyLjctMi43IDEuNDkgMCAyLjcgMS4yMSAyLjcgMi43VjIySDE3YzEuMSAwIDItLjkgMi0ydi00aDEuNWMxLjM4IDAgMi41LTEuMTIgMi41LTIuNVMyMS44OCAxMSAyMC41IDExeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-fast-forward: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTQgMThsOC41LTZMNCA2djEyem05LTEydjEybDguNS02TDEzIDZ6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-file-upload: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTkgMTZoNnYtNmg0bC03LTctNyA3aDR6bS00IDJoMTR2Mkg1eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-file: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuMyA4LjJsLTUuNS01LjVjLS4zLS4zLS43LS41LTEuMi0uNUgzLjljLS44LjEtMS42LjktMS42IDEuOHYxNC4xYzAgLjkuNyAxLjYgMS42IDEuNmgxNC4yYy45IDAgMS42LS43IDEuNi0xLjZWOS40Yy4xLS41LS4xLS45LS40LTEuMnptLTUuOC0zLjNsMy40IDMuNmgtMy40VjQuOXptMy45IDEyLjdINC43Yy0uMSAwLS4yIDAtLjItLjJWNC43YzAtLjIuMS0uMy4yLS4zaDcuMnY0LjRzMCAuOC4zIDEuMWMuMy4zIDEuMS4zIDEuMS4zaDQuM3Y3LjJzLS4xLjItLjIuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-filter-dot: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTE0LDEyVjE5Ljg4QzE0LjA0LDIwLjE4IDEzLjk0LDIwLjUgMTMuNzEsMjAuNzFDMTMuMzIsMjEuMSAxMi42OSwyMS4xIDEyLjMsMjAuNzFMMTAuMjksMTguN0MxMC4wNiwxOC40NyA5Ljk2LDE4LjE2IDEwLDE3Ljg3VjEySDkuOTdMNC4yMSw0LjYyQzMuODcsNC4xOSAzLjk1LDMuNTYgNC4zOCwzLjIyQzQuNTcsMy4wOCA0Ljc4LDMgNSwzVjNIMTlWM0MxOS4yMiwzIDE5LjQzLDMuMDggMTkuNjIsMy4yMkMyMC4wNSwzLjU2IDIwLjEzLDQuMTkgMTkuNzksNC42MkwxNC4wMywxMkgxNFoiIC8+CiAgPC9nPgogIDxnIGNsYXNzPSJqcC1pY29uLWRvdCIgZmlsbD0iI0ZGRiI+CiAgICA8Y2lyY2xlIGN4PSIxOCIgY3k9IjE3IiByPSIzIj48L2NpcmNsZT4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-filter-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEwIDE4aDR2LTJoLTR2MnpNMyA2djJoMThWNkgzem0zIDdoMTJ2LTJINnYyeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-filter: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiNGRkYiPgogICAgPHBhdGggZD0iTTE0LDEyVjE5Ljg4QzE0LjA0LDIwLjE4IDEzLjk0LDIwLjUgMTMuNzEsMjAuNzFDMTMuMzIsMjEuMSAxMi42OSwyMS4xIDEyLjMsMjAuNzFMMTAuMjksMTguN0MxMC4wNiwxOC40NyA5Ljk2LDE4LjE2IDEwLDE3Ljg3VjEySDkuOTdMNC4yMSw0LjYyQzMuODcsNC4xOSAzLjk1LDMuNTYgNC4zOCwzLjIyQzQuNTcsMy4wOCA0Ljc4LDMgNSwzVjNIMTlWM0MxOS4yMiwzIDE5LjQzLDMuMDggMTkuNjIsMy4yMkMyMC4wNSwzLjU2IDIwLjEzLDQuMTkgMTkuNzksNC42MkwxNC4wMywxMkgxNFoiIC8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-folder-favorite: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iIzAwMDAwMCI+CiAgPHBhdGggZD0iTTAgMGgyNHYyNEgwVjB6IiBmaWxsPSJub25lIi8+PHBhdGggY2xhc3M9ImpwLWljb24zIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxNjE2MSIgZD0iTTIwIDZoLThsLTItMkg0Yy0xLjEgMC0yIC45LTIgMnYxMmMwIDEuMS45IDIgMiAyaDE2YzEuMSAwIDItLjkgMi0yVjhjMC0xLjEtLjktMi0yLTJ6bS0yLjA2IDExTDE1IDE1LjI4IDEyLjA2IDE3bC43OC0zLjMzLTIuNTktMi4yNCAzLjQxLS4yOUwxNSA4bDEuMzQgMy4xNCAzLjQxLjI5LTIuNTkgMi4yNC43OCAzLjMzeiIvPgo8L3N2Zz4K);
  --jp-icon-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY4YzAtMS4xLS45LTItMi0yaC04bC0yLTJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-home: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjRweCIgdmlld0JveD0iMCAwIDI0IDI0IiB3aWR0aD0iMjRweCIgZmlsbD0iIzAwMDAwMCI+CiAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPjxwYXRoIGNsYXNzPSJqcC1pY29uMyBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xMCAyMHYtNmg0djZoNXYtOGgzTDEyIDMgMiAxMmgzdjh6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-html5: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uMCBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiMwMDAiIGQ9Ik0xMDguNCAwaDIzdjIyLjhoMjEuMlYwaDIzdjY5aC0yM1Y0NmgtMjF2MjNoLTIzLjJNMjA2IDIzaC0yMC4zVjBoNjMuN3YyM0gyMjl2NDZoLTIzbTUzLjUtNjloMjQuMWwxNC44IDI0LjNMMzEzLjIgMGgyNC4xdjY5aC0yM1YzNC44bC0xNi4xIDI0LjgtMTYuMS0yNC44VjY5aC0yMi42bTg5LjItNjloMjN2NDYuMmgzMi42VjY5aC01NS42Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2U0NGQyNiIgZD0iTTEwNy42IDQ3MWwtMzMtMzcwLjRoMzYyLjhsLTMzIDM3MC4yTDI1NS43IDUxMiIvPgogIDxwYXRoIGNsYXNzPSJqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNmMTY1MjkiIGQ9Ik0yNTYgNDgwLjVWMTMxaDE0OC4zTDM3NiA0NDciLz4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNlYmViZWIiIGQ9Ik0xNDIgMTc2LjNoMTE0djQ1LjRoLTY0LjJsNC4yIDQ2LjVoNjB2NDUuM0gxNTQuNG0yIDIyLjhIMjAybDMuMiAzNi4zIDUwLjggMTMuNnY0Ny40bC05My4yLTI2Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZS1pbnZlcnNlIiBmaWxsPSIjZmZmIiBkPSJNMzY5LjYgMTc2LjNIMjU1Ljh2NDUuNGgxMDkuNm0tNC4xIDQ2LjVIMjU1Ljh2NDUuNGg1NmwtNS4zIDU5LTUwLjcgMTMuNnY0Ny4ybDkzLTI1LjgiLz4KPC9zdmc+Cg==);
  --jp-icon-image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1icmFuZDQganAtaWNvbi1zZWxlY3RhYmxlLWludmVyc2UiIGZpbGw9IiNGRkYiIGQ9Ik0yLjIgMi4yaDE3LjV2MTcuNUgyLjJ6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzNGNTFCNSIgZD0iTTIuMiAyLjJ2MTcuNWgxNy41bC4xLTE3LjVIMi4yem0xMi4xIDIuMmMxLjIgMCAyLjIgMSAyLjIgMi4ycy0xIDIuMi0yLjIgMi4yLTIuMi0xLTIuMi0yLjIgMS0yLjIgMi4yLTIuMnpNNC40IDE3LjZsMy4zLTguOCAzLjMgNi42IDIuMi0zLjIgNC40IDUuNEg0LjR6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-info: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUwLjk3OCA1MC45NzgiPgoJPGcgY2xhc3M9ImpwLWljb24zIiBmaWxsPSIjNjE2MTYxIj4KCQk8cGF0aCBkPSJNNDMuNTIsNy40NThDMzguNzExLDIuNjQ4LDMyLjMwNywwLDI1LjQ4OSwwQzE4LjY3LDAsMTIuMjY2LDIuNjQ4LDcuNDU4LDcuNDU4CgkJCWMtOS45NDMsOS45NDEtOS45NDMsMjYuMTE5LDAsMzYuMDYyYzQuODA5LDQuODA5LDExLjIxMiw3LjQ1NiwxOC4wMzEsNy40NThjMCwwLDAuMDAxLDAsMC4wMDIsMAoJCQljNi44MTYsMCwxMy4yMjEtMi42NDgsMTguMDI5LTcuNDU4YzQuODA5LTQuODA5LDcuNDU3LTExLjIxMiw3LjQ1Ny0xOC4wM0M1MC45NzcsMTguNjcsNDguMzI4LDEyLjI2Niw0My41Miw3LjQ1OHoKCQkJIE00Mi4xMDYsNDIuMTA1Yy00LjQzMiw0LjQzMS0xMC4zMzIsNi44NzItMTYuNjE1LDYuODcyaC0wLjAwMmMtNi4yODUtMC4wMDEtMTIuMTg3LTIuNDQxLTE2LjYxNy02Ljg3MgoJCQljLTkuMTYyLTkuMTYzLTkuMTYyLTI0LjA3MSwwLTMzLjIzM0MxMy4zMDMsNC40NCwxOS4yMDQsMiwyNS40ODksMmM2LjI4NCwwLDEyLjE4NiwyLjQ0LDE2LjYxNyw2Ljg3MgoJCQljNC40MzEsNC40MzEsNi44NzEsMTAuMzMyLDYuODcxLDE2LjYxN0M0OC45NzcsMzEuNzcyLDQ2LjUzNiwzNy42NzUsNDIuMTA2LDQyLjEwNXoiLz4KCQk8cGF0aCBkPSJNMjMuNTc4LDMyLjIxOGMtMC4wMjMtMS43MzQsMC4xNDMtMy4wNTksMC40OTYtMy45NzJjMC4zNTMtMC45MTMsMS4xMS0xLjk5NywyLjI3Mi0zLjI1MwoJCQljMC40NjgtMC41MzYsMC45MjMtMS4wNjIsMS4zNjctMS41NzVjMC42MjYtMC43NTMsMS4xMDQtMS40NzgsMS40MzYtMi4xNzVjMC4zMzEtMC43MDcsMC40OTUtMS41NDEsMC40OTUtMi41CgkJCWMwLTEuMDk2LTAuMjYtMi4wODgtMC43NzktMi45NzljLTAuNTY1LTAuODc5LTEuNTAxLTEuMzM2LTIuODA2LTEuMzY5Yy0xLjgwMiwwLjA1Ny0yLjk4NSwwLjY2Ny0zLjU1LDEuODMyCgkJCWMtMC4zMDEsMC41MzUtMC41MDMsMS4xNDEtMC42MDcsMS44MTRjLTAuMTM5LDAuNzA3LTAuMjA3LDEuNDMyLTAuMjA3LDIuMTc0aC0yLjkzN2MtMC4wOTEtMi4yMDgsMC40MDctNC4xMTQsMS40OTMtNS43MTkKCQkJYzEuMDYyLTEuNjQsMi44NTUtMi40ODEsNS4zNzgtMi41MjdjMi4xNiwwLjAyMywzLjg3NCwwLjYwOCw1LjE0MSwxLjc1OGMxLjI3OCwxLjE2LDEuOTI5LDIuNzY0LDEuOTUsNC44MTEKCQkJYzAsMS4xNDItMC4xMzcsMi4xMTEtMC40MSwyLjkxMWMtMC4zMDksMC44NDUtMC43MzEsMS41OTMtMS4yNjgsMi4yNDNjLTAuNDkyLDAuNjUtMS4wNjgsMS4zMTgtMS43MywyLjAwMgoJCQljLTAuNjUsMC42OTctMS4zMTMsMS40NzktMS45ODcsMi4zNDZjLTAuMjM5LDAuMzc3LTAuNDI5LDAuNzc3LTAuNTY1LDEuMTk5Yy0wLjE2LDAuOTU5LTAuMjE3LDEuOTUxLTAuMTcxLDIuOTc5CgkJCUMyNi41ODksMzIuMjE4LDIzLjU3OCwzMi4yMTgsMjMuNTc4LDMyLjIxOHogTTIzLjU3OCwzOC4yMnYtMy40ODRoMy4wNzZ2My40ODRIMjMuNTc4eiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-inspector: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaW5zcGVjdG9yLWljb24tY29sb3IganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNEg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMThjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY2YzAtMS4xLS45LTItMi0yem0tNSAxNEg0di00aDExdjR6bTAtNUg0VjloMTF2NHptNSA1aC00VjloNHY5eiIvPgo8L3N2Zz4K);
  --jp-icon-json: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtanNvbi1pY29uLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0Y5QTgyNSI+CiAgICA8cGF0aCBkPSJNMjAuMiAxMS44Yy0xLjYgMC0xLjcuNS0xLjcgMSAwIC40LjEuOS4xIDEuMy4xLjUuMS45LjEgMS4zIDAgMS43LTEuNCAyLjMtMy41IDIuM2gtLjl2LTEuOWguNWMxLjEgMCAxLjQgMCAxLjQtLjggMC0uMyAwLS42LS4xLTEgMC0uNC0uMS0uOC0uMS0xLjIgMC0xLjMgMC0xLjggMS4zLTItMS4zLS4yLTEuMy0uNy0xLjMtMiAwLS40LjEtLjguMS0xLjIuMS0uNC4xLS43LjEtMSAwLS44LS40LS43LTEuNC0uOGgtLjVWNC4xaC45YzIuMiAwIDMuNS43IDMuNSAyLjMgMCAuNC0uMS45LS4xIDEuMy0uMS41LS4xLjktLjEgMS4zIDAgLjUuMiAxIDEuNyAxdjEuOHpNMS44IDEwLjFjMS42IDAgMS43LS41IDEuNy0xIDAtLjQtLjEtLjktLjEtMS4zLS4xLS41LS4xLS45LS4xLTEuMyAwLTEuNiAxLjQtMi4zIDMuNS0yLjNoLjl2MS45aC0uNWMtMSAwLTEuNCAwLTEuNC44IDAgLjMgMCAuNi4xIDEgMCAuMi4xLjYuMSAxIDAgMS4zIDAgMS44LTEuMyAyQzYgMTEuMiA2IDExLjcgNiAxM2MwIC40LS4xLjgtLjEgMS4yLS4xLjMtLjEuNy0uMSAxIDAgLjguMy44IDEuNC44aC41djEuOWgtLjljLTIuMSAwLTMuNS0uNi0zLjUtMi4zIDAtLjQuMS0uOS4xLTEuMy4xLS41LjEtLjkuMS0xLjMgMC0uNS0uMi0xLTEuNy0xdi0xLjl6Ii8+CiAgICA8Y2lyY2xlIGN4PSIxMSIgY3k9IjEzLjgiIHI9IjIuMSIvPgogICAgPGNpcmNsZSBjeD0iMTEiIGN5PSI4LjIiIHI9IjIuMSIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-julia: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDMyNSAzMDAiPgogIDxnIGNsYXNzPSJqcC1icmFuZDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjY2IzYzMzIj4KICAgIDxwYXRoIGQ9Ik0gMTUwLjg5ODQzOCAyMjUgQyAxNTAuODk4NDM4IDI2Ni40MjE4NzUgMTE3LjMyMDMxMiAzMDAgNzUuODk4NDM4IDMwMCBDIDM0LjQ3NjU2MiAzMDAgMC44OTg0MzggMjY2LjQyMTg3NSAwLjg5ODQzOCAyMjUgQyAwLjg5ODQzOCAxODMuNTc4MTI1IDM0LjQ3NjU2MiAxNTAgNzUuODk4NDM4IDE1MCBDIDExNy4zMjAzMTIgMTUwIDE1MC44OTg0MzggMTgzLjU3ODEyNSAxNTAuODk4NDM4IDIyNSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzM4OTgyNiI+CiAgICA8cGF0aCBkPSJNIDIzNy41IDc1IEMgMjM3LjUgMTE2LjQyMTg3NSAyMDMuOTIxODc1IDE1MCAxNjIuNSAxNTAgQyAxMjEuMDc4MTI1IDE1MCA4Ny41IDExNi40MjE4NzUgODcuNSA3NSBDIDg3LjUgMzMuNTc4MTI1IDEyMS4wNzgxMjUgMCAxNjIuNSAwIEMgMjAzLjkyMTg3NSAwIDIzNy41IDMzLjU3ODEyNSAyMzcuNSA3NSIvPgogIDwvZz4KICA8ZyBjbGFzcz0ianAtYnJhbmQwIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzk1NThiMiI+CiAgICA8cGF0aCBkPSJNIDMyNC4xMDE1NjIgMjI1IEMgMzI0LjEwMTU2MiAyNjYuNDIxODc1IDI5MC41MjM0MzggMzAwIDI0OS4xMDE1NjIgMzAwIEMgMjA3LjY3OTY4OCAzMDAgMTc0LjEwMTU2MiAyNjYuNDIxODc1IDE3NC4xMDE1NjIgMjI1IEMgMTc0LjEwMTU2MiAxODMuNTc4MTI1IDIwNy42Nzk2ODggMTUwIDI0OS4xMDE1NjIgMTUwIEMgMjkwLjUyMzQzOCAxNTAgMzI0LjEwMTU2MiAxODMuNTc4MTI1IDMyNC4xMDE1NjIgMjI1Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-jupyter-favicon: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTUyIiBoZWlnaHQ9IjE2NSIgdmlld0JveD0iMCAwIDE1MiAxNjUiIHZlcnNpb249IjEuMSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgPGcgY2xhc3M9ImpwLWp1cHl0ZXItaWNvbi1jb2xvciIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA3ODk0NywgMTEwLjU4MjkyNykiIGQ9Ik03NS45NDIyODQyLDI5LjU4MDQ1NjEgQzQzLjMwMjM5NDcsMjkuNTgwNDU2MSAxNC43OTY3ODMyLDE3LjY1MzQ2MzQgMCwwIEM1LjUxMDgzMjExLDE1Ljg0MDY4MjkgMTUuNzgxNTM4OSwyOS41NjY3NzMyIDI5LjM5MDQ5NDcsMzkuMjc4NDE3MSBDNDIuOTk5Nyw0OC45ODk4NTM3IDU5LjI3MzcsNTQuMjA2NzgwNSA3NS45NjA1Nzg5LDU0LjIwNjc4MDUgQzkyLjY0NzQ1NzksNTQuMjA2NzgwNSAxMDguOTIxNDU4LDQ4Ljk4OTg1MzcgMTIyLjUzMDY2MywzOS4yNzg0MTcxIEMxMzYuMTM5NDUzLDI5LjU2Njc3MzIgMTQ2LjQxMDI4NCwxNS44NDA2ODI5IDE1MS45MjExNTgsMCBDMTM3LjA4Nzg2OCwxNy42NTM0NjM0IDEwOC41ODI1ODksMjkuNTgwNDU2MSA3NS45NDIyODQyLDI5LjU4MDQ1NjEgTDc1Ljk0MjI4NDIsMjkuNTgwNDU2MSBaIiAvPgogICAgPHBhdGggdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMzczNjgsIDAuNzA0ODc4KSIgZD0iTTc1Ljk3ODQ1NzksMjQuNjI2NDA3MyBDMTA4LjYxODc2MywyNC42MjY0MDczIDEzNy4xMjQ0NTgsMzYuNTUzNDQxNSAxNTEuOTIxMTU4LDU0LjIwNjc4MDUgQzE0Ni40MTAyODQsMzguMzY2MjIyIDEzNi4xMzk0NTMsMjQuNjQwMTMxNyAxMjIuNTMwNjYzLDE0LjkyODQ4NzggQzEwOC45MjE0NTgsNS4yMTY4NDM5IDkyLjY0NzQ1NzksMCA3NS45NjA1Nzg5LDAgQzU5LjI3MzcsMCA0Mi45OTk3LDUuMjE2ODQzOSAyOS4zOTA0OTQ3LDE0LjkyODQ4NzggQzE1Ljc4MTUzODksMjQuNjQwMTMxNyA1LjUxMDgzMjExLDM4LjM2NjIyMiAwLDU0LjIwNjc4MDUgQzE0LjgzMzA4MTYsMzYuNTg5OTI5MyA0My4zMzg1Njg0LDI0LjYyNjQwNzMgNzUuOTc4NDU3OSwyNC42MjY0MDczIEw3NS45Nzg0NTc5LDI0LjYyNjQwNzMgWiIgLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-jupyter: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMzkiIGhlaWdodD0iNTEiIHZpZXdCb3g9IjAgMCAzOSA1MSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgtMTYzOCAtMjI4MSkiPgogICAgIDxnIGNsYXNzPSJqcC1qdXB5dGVyLWljb24tY29sb3IiIGZpbGw9IiNGMzc3MjYiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5Ljc0IDIzMTEuOTgpIiBkPSJNIDE4LjI2NDYgNy4xMzQxMUMgMTAuNDE0NSA3LjEzNDExIDMuNTU4NzIgNC4yNTc2IDAgMEMgMS4zMjUzOSAzLjgyMDQgMy43OTU1NiA3LjEzMDgxIDcuMDY4NiA5LjQ3MzAzQyAxMC4zNDE3IDExLjgxNTIgMTQuMjU1NyAxMy4wNzM0IDE4LjI2OSAxMy4wNzM0QyAyMi4yODIzIDEzLjA3MzQgMjYuMTk2MyAxMS44MTUyIDI5LjQ2OTQgOS40NzMwM0MgMzIuNzQyNCA3LjEzMDgxIDM1LjIxMjYgMy44MjA0IDM2LjUzOCAwQyAzMi45NzA1IDQuMjU3NiAyNi4xMTQ4IDcuMTM0MTEgMTguMjY0NiA3LjEzNDExWiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM5LjczIDIyODUuNDgpIiBkPSJNIDE4LjI3MzMgNS45MzkzMUMgMjYuMTIzNSA1LjkzOTMxIDMyLjk3OTMgOC44MTU4MyAzNi41MzggMTMuMDczNEMgMzUuMjEyNiA5LjI1MzAzIDMyLjc0MjQgNS45NDI2MiAyOS40Njk0IDMuNjAwNEMgMjYuMTk2MyAxLjI1ODE4IDIyLjI4MjMgMCAxOC4yNjkgMEMgMTQuMjU1NyAwIDEwLjM0MTcgMS4yNTgxOCA3LjA2ODYgMy42MDA0QyAzLjc5NTU2IDUuOTQyNjIgMS4zMjUzOSA5LjI1MzAzIDAgMTMuMDczNEMgMy41Njc0NSA4LjgyNDYzIDEwLjQyMzIgNS45MzkzMSAxOC4yNzMzIDUuOTM5MzFaIi8+CiAgICA8L2c+CiAgICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjY5LjMgMjI4MS4zMSkiIGQ9Ik0gNS44OTM1MyAyLjg0NEMgNS45MTg4OSAzLjQzMTY1IDUuNzcwODUgNC4wMTM2NyA1LjQ2ODE1IDQuNTE2NDVDIDUuMTY1NDUgNS4wMTkyMiA0LjcyMTY4IDUuNDIwMTUgNC4xOTI5OSA1LjY2ODUxQyAzLjY2NDMgNS45MTY4OCAzLjA3NDQ0IDYuMDAxNTEgMi40OTgwNSA1LjkxMTcxQyAxLjkyMTY2IDUuODIxOSAxLjM4NDYzIDUuNTYxNyAwLjk1NDg5OCA1LjE2NDAxQyAwLjUyNTE3IDQuNzY2MzMgMC4yMjIwNTYgNC4yNDkwMyAwLjA4MzkwMzcgMy42Nzc1N0MgLTAuMDU0MjQ4MyAzLjEwNjExIC0wLjAyMTIzIDIuNTA2MTcgMC4xNzg3ODEgMS45NTM2NEMgMC4zNzg3OTMgMS40MDExIDAuNzM2ODA5IDAuOTIwODE3IDEuMjA3NTQgMC41NzM1MzhDIDEuNjc4MjYgMC4yMjYyNTkgMi4yNDA1NSAwLjAyNzU5MTkgMi44MjMyNiAwLjAwMjY3MjI5QyAzLjYwMzg5IC0wLjAzMDcxMTUgNC4zNjU3MyAwLjI0OTc4OSA0Ljk0MTQyIDAuNzgyNTUxQyA1LjUxNzExIDEuMzE1MzEgNS44NTk1NiAyLjA1Njc2IDUuODkzNTMgMi44NDRaIi8+CiAgICAgIDxwYXRoIHRyYW5zZm9ybT0idHJhbnNsYXRlKDE2MzkuOCAyMzIzLjgxKSIgZD0iTSA3LjQyNzg5IDMuNTgzMzhDIDcuNDYwMDggNC4zMjQzIDcuMjczNTUgNS4wNTgxOSA2Ljg5MTkzIDUuNjkyMTNDIDYuNTEwMzEgNi4zMjYwNyA1Ljk1MDc1IDYuODMxNTYgNS4yODQxMSA3LjE0NDZDIDQuNjE3NDcgNy40NTc2MyAzLjg3MzcxIDcuNTY0MTQgMy4xNDcwMiA3LjQ1MDYzQyAyLjQyMDMyIDcuMzM3MTIgMS43NDMzNiA3LjAwODcgMS4yMDE4NCA2LjUwNjk1QyAwLjY2MDMyOCA2LjAwNTIgMC4yNzg2MSA1LjM1MjY4IDAuMTA1MDE3IDQuNjMyMDJDIC0wLjA2ODU3NTcgMy45MTEzNSAtMC4wMjYyMzYxIDMuMTU0OTQgMC4yMjY2NzUgMi40NTg1NkMgMC40Nzk1ODcgMS43NjIxNyAwLjkzMTY5NyAxLjE1NzEzIDEuNTI1NzYgMC43MjAwMzNDIDIuMTE5ODMgMC4yODI5MzUgMi44MjkxNCAwLjAzMzQzOTUgMy41NjM4OSAwLjAwMzEzMzQ0QyA0LjU0NjY3IC0wLjAzNzQwMzMgNS41MDUyOSAwLjMxNjcwNiA2LjIyOTYxIDAuOTg3ODM1QyA2Ljk1MzkzIDEuNjU4OTYgNy4zODQ4NCAyLjU5MjM1IDcuNDI3ODkgMy41ODMzOEwgNy40Mjc4OSAzLjU4MzM4WiIvPgogICAgICA8cGF0aCB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxNjM4LjM2IDIyODYuMDYpIiBkPSJNIDIuMjc0NzEgNC4zOTYyOUMgMS44NDM2MyA0LjQxNTA4IDEuNDE2NzEgNC4zMDQ0NSAxLjA0Nzk5IDQuMDc4NDNDIDAuNjc5MjY4IDMuODUyNCAwLjM4NTMyOCAzLjUyMTE0IDAuMjAzMzcxIDMuMTI2NTZDIDAuMDIxNDEzNiAyLjczMTk4IC0wLjA0MDM3OTggMi4yOTE4MyAwLjAyNTgxMTYgMS44NjE4MUMgMC4wOTIwMDMxIDEuNDMxOCAwLjI4MzIwNCAxLjAzMTI2IDAuNTc1MjEzIDAuNzEwODgzQyAwLjg2NzIyMiAwLjM5MDUxIDEuMjQ2OTEgMC4xNjQ3MDggMS42NjYyMiAwLjA2MjA1OTJDIDIuMDg1NTMgLTAuMDQwNTg5NyAyLjUyNTYxIC0wLjAxNTQ3MTQgMi45MzA3NiAwLjEzNDIzNUMgMy4zMzU5MSAwLjI4Mzk0MSAzLjY4NzkyIDAuNTUxNTA1IDMuOTQyMjIgMC45MDMwNkMgNC4xOTY1MiAxLjI1NDYyIDQuMzQxNjkgMS42NzQzNiA0LjM1OTM1IDIuMTA5MTZDIDQuMzgyOTkgMi42OTEwNyA0LjE3Njc4IDMuMjU4NjkgMy43ODU5NyAzLjY4NzQ2QyAzLjM5NTE2IDQuMTE2MjQgMi44NTE2NiA0LjM3MTE2IDIuMjc0NzEgNC4zOTYyOUwgMi4yNzQ3MSA0LjM5NjI5WiIvPgogICAgPC9nPgogIDwvZz4+Cjwvc3ZnPgo=);
  --jp-icon-jupyterlab-wordmark: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyMDAiIHZpZXdCb3g9IjAgMCAxODYwLjggNDc1Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0RTRFNEUiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDQ4MC4xMzY0MDEsIDY0LjI3MTQ5MykiPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC4wMDAwMDAsIDU4Ljg3NTU2NikiPgogICAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgwLjA4NzYwMywgMC4xNDAyOTQpIj4KICAgICAgICA8cGF0aCBkPSJNLTQyNi45LDE2OS44YzAsNDguNy0zLjcsNjQuNy0xMy42LDc2LjRjLTEwLjgsMTAtMjUsMTUuNS0zOS43LDE1LjVsMy43LDI5IGMyMi44LDAuMyw0NC44LTcuOSw2MS45LTIzLjFjMTcuOC0xOC41LDI0LTQ0LjEsMjQtODMuM1YwSC00Mjd2MTcwLjFMLTQyNi45LDE2OS44TC00MjYuOSwxNjkuOHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMTU1LjA0NTI5NiwgNTYuODM3MTA0KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuNTYyNDUzLCAxLjc5OTg0MikiPgogICAgICAgIDxwYXRoIGQ9Ik0tMzEyLDE0OGMwLDIxLDAsMzkuNSwxLjcsNTUuNGgtMzEuOGwtMi4xLTMzLjNoLTAuOGMtNi43LDExLjYtMTYuNCwyMS4zLTI4LDI3LjkgYy0xMS42LDYuNi0yNC44LDEwLTM4LjIsOS44Yy0zMS40LDAtNjktMTcuNy02OS04OVYwaDM2LjR2MTEyLjdjMCwzOC43LDExLjYsNjQuNyw0NC42LDY0LjdjMTAuMy0wLjIsMjAuNC0zLjUsMjguOS05LjQgYzguNS01LjksMTUuMS0xNC4zLDE4LjktMjMuOWMyLjItNi4xLDMuMy0xMi41LDMuMy0xOC45VjAuMmgzNi40VjE0OEgtMzEyTC0zMTIsMTQ4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgzOTAuMDEzMzIyLCA1My40Nzk2MzgpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS43MDY0NTgsIDAuMjMxNDI1KSI+CiAgICAgICAgPHBhdGggZD0iTS00NzguNiw3MS40YzAtMjYtMC44LTQ3LTEuNy02Ni43aDMyLjdsMS43LDM0LjhoMC44YzcuMS0xMi41LDE3LjUtMjIuOCwzMC4xLTI5LjcgYzEyLjUtNywyNi43LTEwLjMsNDEtOS44YzQ4LjMsMCw4NC43LDQxLjcsODQuNywxMDMuM2MwLDczLjEtNDMuNywxMDkuMi05MSwxMDkuMmMtMTIuMSwwLjUtMjQuMi0yLjItMzUtNy44IGMtMTAuOC01LjYtMTkuOS0xMy45LTI2LjYtMjQuMmgtMC44VjI5MWgtMzZ2LTIyMEwtNDc4LjYsNzEuNEwtNDc4LjYsNzEuNHogTS00NDIuNiwxMjUuNmMwLjEsNS4xLDAuNiwxMC4xLDEuNywxNS4xIGMzLDEyLjMsOS45LDIzLjMsMTkuOCwzMS4xYzkuOSw3LjgsMjIuMSwxMi4xLDM0LjcsMTIuMWMzOC41LDAsNjAuNy0zMS45LDYwLjctNzguNWMwLTQwLjctMjEuMS03NS42LTU5LjUtNzUuNiBjLTEyLjksMC40LTI1LjMsNS4xLTM1LjMsMTMuNGMtOS45LDguMy0xNi45LDE5LjctMTkuNiwzMi40Yy0xLjUsNC45LTIuMywxMC0yLjUsMTUuMVYxMjUuNkwtNDQyLjYsMTI1LjZMLTQ0Mi42LDEyNS42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSg2MDYuNzQwNzI2LCA1Ni44MzcxMDQpIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMC43NTEyMjYsIDEuOTg5Mjk5KSI+CiAgICAgICAgPHBhdGggZD0iTS00NDAuOCwwbDQzLjcsMTIwLjFjNC41LDEzLjQsOS41LDI5LjQsMTIuOCw0MS43aDAuOGMzLjctMTIuMiw3LjktMjcuNywxMi44LTQyLjQgbDM5LjctMTE5LjJoMzguNUwtMzQ2LjksMTQ1Yy0yNiw2OS43LTQzLjcsMTA1LjQtNjguNiwxMjcuMmMtMTIuNSwxMS43LTI3LjksMjAtNDQuNiwyMy45bC05LjEtMzEuMSBjMTEuNy0zLjksMjIuNS0xMC4xLDMxLjgtMTguMWMxMy4yLTExLjEsMjMuNy0yNS4yLDMwLjYtNDEuMmMxLjUtMi44LDIuNS01LjcsMi45LTguOGMtMC4zLTMuMy0xLjItNi42LTIuNS05LjdMLTQ4MC4yLDAuMSBoMzkuN0wtNDQwLjgsMEwtNDQwLjgsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoODIyLjc0ODEwNCwgMC4wMDAwMDApIj4KICAgICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoMS40NjQwNTAsIDAuMzc4OTE0KSI+CiAgICAgICAgPHBhdGggZD0iTS00MTMuNywwdjU4LjNoNTJ2MjguMmgtNTJWMTk2YzAsMjUsNywzOS41LDI3LjMsMzkuNWM3LjEsMC4xLDE0LjItMC43LDIxLjEtMi41IGwxLjcsMjcuN2MtMTAuMywzLjctMjEuMyw1LjQtMzIuMiw1Yy03LjMsMC40LTE0LjYtMC43LTIxLjMtMy40Yy02LjgtMi43LTEyLjktNi44LTE3LjktMTIuMWMtMTAuMy0xMC45LTE0LjEtMjktMTQuMS01Mi45IFY4Ni41aC0zMVY1OC4zaDMxVjkuNkwtNDEzLjcsMEwtNDEzLjcsMHoiLz4KICAgICAgPC9nPgogICAgPC9nPgogICAgPGcgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOTc0LjQzMzI4NiwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDAuOTkwMDM0LCAwLjYxMDMzOSkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDQ1LjgsMTEzYzAuOCw1MCwzMi4yLDcwLjYsNjguNiw3MC42YzE5LDAuNiwzNy45LTMsNTUuMy0xMC41bDYuMiwyNi40IGMtMjAuOSw4LjktNDMuNSwxMy4xLTY2LjIsMTIuNmMtNjEuNSwwLTk4LjMtNDEuMi05OC4zLTEwMi41Qy00ODAuMiw0OC4yLTQ0NC43LDAtMzg2LjUsMGM2NS4yLDAsODIuNyw1OC4zLDgyLjcsOTUuNyBjLTAuMSw1LjgtMC41LDExLjUtMS4yLDE3LjJoLTE0MC42SC00NDUuOEwtNDQ1LjgsMTEzeiBNLTMzOS4yLDg2LjZjMC40LTIzLjUtOS41LTYwLjEtNTAuNC02MC4xIGMtMzYuOCwwLTUyLjgsMzQuNC01NS43LDYwLjFILTMzOS4yTC0zMzkuMiw4Ni42TC0zMzkuMiw4Ni42eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgICA8ZyB0cmFuc2Zvcm09InRyYW5zbGF0ZSgxMjAxLjk2MTA1OCwgNTMuNDc5NjM4KSI+CiAgICAgIDxnIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEuMTc5NjQwLCAwLjcwNTA2OCkiPgogICAgICAgIDxwYXRoIGQ9Ik0tNDc4LjYsNjhjMC0yMy45LTAuNC00NC41LTEuNy02My40aDMxLjhsMS4yLDM5LjloMS43YzkuMS0yNy4zLDMxLTQ0LjUsNTUuMy00NC41IGMzLjUtMC4xLDcsMC40LDEwLjMsMS4ydjM0LjhjLTQuMS0wLjktOC4yLTEuMy0xMi40LTEuMmMtMjUuNiwwLTQzLjcsMTkuNy00OC43LDQ3LjRjLTEsNS43LTEuNiwxMS41LTEuNywxNy4ydjEwOC4zaC0zNlY2OCBMLTQ3OC42LDY4eiIvPgogICAgICA8L2c+CiAgICA8L2c+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi13YXJuMCIgZmlsbD0iI0YzNzcyNiI+CiAgICA8cGF0aCBkPSJNMTM1Mi4zLDMyNi4yaDM3VjI4aC0zN1YzMjYuMnogTTE2MDQuOCwzMjYuMmMtMi41LTEzLjktMy40LTMxLjEtMy40LTQ4Ljd2LTc2IGMwLTQwLjctMTUuMS04My4xLTc3LjMtODMuMWMtMjUuNiwwLTUwLDcuMS02Ni44LDE4LjFsOC40LDI0LjRjMTQuMy05LjIsMzQtMTUuMSw1My0xNS4xYzQxLjYsMCw0Ni4yLDMwLjIsNDYuMiw0N3Y0LjIgYy03OC42LTAuNC0xMjIuMywyNi41LTEyMi4zLDc1LjZjMCwyOS40LDIxLDU4LjQsNjIuMiw1OC40YzI5LDAsNTAuOS0xNC4zLDYyLjItMzAuMmgxLjNsMi45LDI1LjZIMTYwNC44eiBNMTU2NS43LDI1Ny43IGMwLDMuOC0wLjgsOC0yLjEsMTEuOGMtNS45LDE3LjItMjIuNywzNC00OS4yLDM0Yy0xOC45LDAtMzQuOS0xMS4zLTM0LjktMzUuM2MwLTM5LjUsNDUuOC00Ni42LDg2LjItNDUuOFYyNTcuN3ogTTE2OTguNSwzMjYuMiBsMS43LTMzLjZoMS4zYzE1LjEsMjYuOSwzOC43LDM4LjIsNjguMSwzOC4yYzQ1LjQsMCw5MS4yLTM2LjEsOTEuMi0xMDguOGMwLjQtNjEuNy0zNS4zLTEwMy43LTg1LjctMTAzLjcgYy0zMi44LDAtNTYuMywxNC43LTY5LjMsMzcuNGgtMC44VjI4aC0zNi42djI0NS43YzAsMTguMS0wLjgsMzguNi0xLjcsNTIuNUgxNjk4LjV6IE0xNzA0LjgsMjA4LjJjMC01LjksMS4zLTEwLjksMi4xLTE1LjEgYzcuNi0yOC4xLDMxLjEtNDUuNCw1Ni4zLTQ1LjRjMzkuNSwwLDYwLjUsMzQuOSw2MC41LDc1LjZjMCw0Ni42LTIzLjEsNzguMS02MS44LDc4LjFjLTI2LjksMC00OC4zLTE3LjYtNTUuNS00My4zIGMtMC44LTQuMi0xLjctOC44LTEuNy0xMy40VjIwOC4yeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzYxNjE2MSIgZD0iTTE1IDlIOXY2aDZWOXptLTIgNGgtMnYtMmgydjJ6bTgtMlY5aC0yVjdjMC0xLjEtLjktMi0yLTJoLTJWM2gtMnYyaC0yVjNIOXYySDdjLTEuMSAwLTIgLjktMiAydjJIM3YyaDJ2MkgzdjJoMnYyYzAgMS4xLjkgMiAyIDJoMnYyaDJ2LTJoMnYyaDJ2LTJoMmMxLjEgMCAyLS45IDItMnYtMmgydi0yaC0ydi0yaDJ6bS00IDZIN1Y3aDEwdjEweiIvPgo8L3N2Zz4K);
  --jp-icon-keyboard: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMjAgNUg0Yy0xLjEgMC0xLjk5LjktMS45OSAyTDIgMTdjMCAxLjEuOSAyIDIgMmgxNmMxLjEgMCAyLS45IDItMlY3YzAtMS4xLS45LTItMi0yem0tOSAzaDJ2MmgtMlY4em0wIDNoMnYyaC0ydi0yek04IDhoMnYySDhWOHptMCAzaDJ2Mkg4di0yem0tMSAySDV2LTJoMnYyem0wLTNINVY4aDJ2MnptOSA3SDh2LTJoOHYyem0wLTRoLTJ2LTJoMnYyem0wLTNoLTJWOGgydjJ6bTMgM2gtMnYtMmgydjJ6bTAtM2gtMlY4aDJ2MnoiLz4KPC9zdmc+Cg==);
  --jp-icon-launch: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMzIgMzIiIHdpZHRoPSIzMiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik0yNiwyOEg2YTIuMDAyNywyLjAwMjcsMCwwLDEtMi0yVjZBMi4wMDI3LDIuMDAyNywwLDAsMSw2LDRIMTZWNkg2VjI2SDI2VjE2aDJWMjZBMi4wMDI3LDIuMDAyNywwLDAsMSwyNiwyOFoiLz4KICAgIDxwb2x5Z29uIHBvaW50cz0iMjAgMiAyMCA0IDI2LjU4NiA0IDE4IDEyLjU4NiAxOS40MTQgMTQgMjggNS40MTQgMjggMTIgMzAgMTIgMzAgMiAyMCAyIi8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-launcher: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkgMTlINVY1aDdWM0g1YTIgMiAwIDAwLTIgMnYxNGEyIDIgMCAwMDIgMmgxNGMxLjEgMCAyLS45IDItMnYtN2gtMnY3ek0xNCAzdjJoMy41OWwtOS44MyA5LjgzIDEuNDEgMS40MUwxOSA2LjQxVjEwaDJWM2gtN3oiLz4KPC9zdmc+Cg==);
  --jp-icon-line-form: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGZpbGw9IndoaXRlIiBkPSJNNS44OCA0LjEyTDEzLjc2IDEybC03Ljg4IDcuODhMOCAyMmwxMC0xMEw4IDJ6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-link: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTMuOSAxMmMwLTEuNzEgMS4zOS0zLjEgMy4xLTMuMWg0VjdIN2MtMi43NiAwLTUgMi4yNC01IDVzMi4yNCA1IDUgNWg0di0xLjlIN2MtMS43MSAwLTMuMS0xLjM5LTMuMS0zLjF6TTggMTNoOHYtMkg4djJ6bTktNmgtNHYxLjloNGMxLjcxIDAgMy4xIDEuMzkgMy4xIDMuMXMtMS4zOSAzLjEtMy4xIDMuMWgtNFYxN2g0YzIuNzYgMCA1LTIuMjQgNS01cy0yLjI0LTUtNS01eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-list: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xOSA1djE0SDVWNWgxNG0xLjEtMkgzLjljLS41IDAtLjkuNC0uOS45djE2LjJjMCAuNC40LjkuOS45aDE2LjJjLjQgMCAuOS0uNS45LS45VjMuOWMwLS41LS41LS45LS45LS45ek0xMSA3aDZ2MmgtNlY3em0wIDRoNnYyaC02di0yem0wIDRoNnYyaC02ek03IDdoMnYySDd6bTAgNGgydjJIN3ptMCA0aDJ2Mkg3eiIvPgo8L3N2Zz4K);
  --jp-icon-markdown: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDAganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjN0IxRkEyIiBkPSJNNSAxNC45aDEybC02LjEgNnptOS40LTYuOGMwLTEuMy0uMS0yLjktLjEtNC41LS40IDEuNC0uOSAyLjktMS4zIDQuM2wtMS4zIDQuM2gtMkw4LjUgNy45Yy0uNC0xLjMtLjctMi45LTEtNC4zLS4xIDEuNi0uMSAzLjItLjIgNC42TDcgMTIuNEg0LjhsLjctMTFoMy4zTDEwIDVjLjQgMS4yLjcgMi43IDEgMy45LjMtMS4yLjctMi42IDEtMy45bDEuMi0zLjdoMy4zbC42IDExaC0yLjRsLS4zLTQuMnoiLz4KPC9zdmc+Cg==);
  --jp-icon-move-down: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNMTIuNDcxIDcuNTI4OTlDMTIuNzYzMiA3LjIzNjg0IDEyLjc2MzIgNi43NjMxNiAxMi40NzEgNi40NzEwMVY2LjQ3MTAxQzEyLjE3OSA2LjE3OTA1IDExLjcwNTcgNi4xNzg4NCAxMS40MTM1IDYuNDcwNTRMNy43NSAxMC4xMjc1VjEuNzVDNy43NSAxLjMzNTc5IDcuNDE0MjEgMSA3IDFWMUM2LjU4NTc5IDEgNi4yNSAxLjMzNTc5IDYuMjUgMS43NVYxMC4xMjc1TDIuNTk3MjYgNi40NjgyMkMyLjMwMzM4IDYuMTczODEgMS44MjY0MSA2LjE3MzU5IDEuNTMyMjYgNi40Njc3NFY2LjQ2Nzc0QzEuMjM4MyA2Ljc2MTcgMS4yMzgzIDcuMjM4MyAxLjUzMjI2IDcuNTMyMjZMNi4yOTI4OSAxMi4yOTI5QzYuNjgzNDIgMTIuNjgzNCA3LjMxNjU4IDEyLjY4MzQgNy43MDcxMSAxMi4yOTI5TDEyLjQ3MSA3LjUyODk5WiIgZmlsbD0iIzYxNjE2MSIvPgo8L3N2Zz4K);
  --jp-icon-move-up: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTQiIGhlaWdodD0iMTQiIHZpZXdCb3g9IjAgMCAxNCAxNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggY2xhc3M9ImpwLWljb24zIiBkPSJNMS41Mjg5OSA2LjQ3MTAxQzEuMjM2ODQgNi43NjMxNiAxLjIzNjg0IDcuMjM2ODQgMS41Mjg5OSA3LjUyODk5VjcuNTI4OTlDMS44MjA5NSA3LjgyMDk1IDIuMjk0MjYgNy44MjExNiAyLjU4NjQ5IDcuNTI5NDZMNi4yNSAzLjg3MjVWMTIuMjVDNi4yNSAxMi42NjQyIDYuNTg1NzkgMTMgNyAxM1YxM0M3LjQxNDIxIDEzIDcuNzUgMTIuNjY0MiA3Ljc1IDEyLjI1VjMuODcyNUwxMS40MDI3IDcuNTMxNzhDMTEuNjk2NiA3LjgyNjE5IDEyLjE3MzYgNy44MjY0MSAxMi40Njc3IDcuNTMyMjZWNy41MzIyNkMxMi43NjE3IDcuMjM4MyAxMi43NjE3IDYuNzYxNyAxMi40Njc3IDYuNDY3NzRMNy43MDcxMSAxLjcwNzExQzcuMzE2NTggMS4zMTY1OCA2LjY4MzQyIDEuMzE2NTggNi4yOTI4OSAxLjcwNzExTDEuNTI4OTkgNi40NzEwMVoiIGZpbGw9IiM2MTYxNjEiLz4KPC9zdmc+Cg==);
  --jp-icon-new-folder: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIwIDZoLThsLTItMkg0Yy0xLjExIDAtMS45OS44OS0xLjk5IDJMMiAxOGMwIDEuMTEuODkgMiAyIDJoMTZjMS4xMSAwIDItLjg5IDItMlY4YzAtMS4xMS0uODktMi0yLTJ6bS0xIDhoLTN2M2gtMnYtM2gtM3YtMmgzVjloMnYzaDN2MnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-not-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI1IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMTkgMTcuMTg0NCAyLjk2OTY4IDE0LjMwMzIgMS44NjA5NCAxMS40NDA5WiIvPgogICAgPHBhdGggY2xhc3M9ImpwLWljb24yIiBzdHJva2U9IiMzMzMzMzMiIHN0cm9rZS13aWR0aD0iMiIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOS4zMTU5MiA5LjMyMDMxKSIgZD0iTTcuMzY4NDIgMEwwIDcuMzY0NzkiLz4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDkuMzE1OTIgMTYuNjgzNikgc2NhbGUoMSAtMSkiIGQ9Ik03LjM2ODQyIDBMMCA3LjM2NDc5Ii8+Cjwvc3ZnPgo=);
  --jp-icon-notebook: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtbm90ZWJvb2staWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiNFRjZDMDAiPgogICAgPHBhdGggZD0iTTE4LjcgMy4zdjE1LjRIMy4zVjMuM2gxNS40bTEuNS0xLjVIMS44djE4LjNoMTguM2wuMS0xOC4zeiIvPgogICAgPHBhdGggZD0iTTE2LjUgMTYuNWwtNS40LTQuMy01LjYgNC4zdi0xMWgxMXoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-numbering: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjIiIGhlaWdodD0iMjIiIHZpZXdCb3g9IjAgMCAyOCAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTQgMTlINlYxOS41SDVWMjAuNUg2VjIxSDRWMjJIN1YxOEg0VjE5Wk01IDEwSDZWNkg0VjdINVYxMFpNNCAxM0g1LjhMNCAxNS4xVjE2SDdWMTVINS4yTDcgMTIuOVYxMkg0VjEzWk05IDdWOUgyM1Y3SDlaTTkgMjFIMjNWMTlIOVYyMVpNOSAxNUgyM1YxM0g5VjE1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-offline-bolt: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyIDIuMDJjLTUuNTEgMC05Ljk4IDQuNDctOS45OCA5Ljk4czQuNDcgOS45OCA5Ljk4IDkuOTggOS45OC00LjQ3IDkuOTgtOS45OFMxNy41MSAyLjAyIDEyIDIuMDJ6TTExLjQ4IDIwdi02LjI2SDhMMTMgNHY2LjI2aDMuMzVMMTEuNDggMjB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-palette: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE4IDEzVjIwSDRWNkg5LjAyQzkuMDcgNS4yOSA5LjI0IDQuNjIgOS41IDRINEMyLjkgNCAyIDQuOSAyIDZWMjBDMiAyMS4xIDIuOSAyMiA0IDIySDE4QzE5LjEgMjIgMjAgMjEuMSAyMCAyMFYxNUwxOCAxM1pNMTkuMyA4Ljg5QzE5Ljc0IDguMTkgMjAgNy4zOCAyMCA2LjVDMjAgNC4wMSAxNy45OSAyIDE1LjUgMkMxMy4wMSAyIDExIDQuMDEgMTEgNi41QzExIDguOTkgMTMuMDEgMTEgMTUuNDkgMTFDMTYuMzcgMTEgMTcuMTkgMTAuNzQgMTcuODggMTAuM0wyMSAxMy40MkwyMi40MiAxMkwxOS4zIDguODlaTTE1LjUgOUMxNC4xMiA5IDEzIDcuODggMTMgNi41QzEzIDUuMTIgMTQuMTIgNCAxNS41IDRDMTYuODggNCAxOCA1LjEyIDE4IDYuNUMxOCA3Ljg4IDE2Ljg4IDkgMTUuNSA5WiIvPgogICAgPHBhdGggZmlsbC1ydWxlPSJldmVub2RkIiBjbGlwLXJ1bGU9ImV2ZW5vZGQiIGQ9Ik00IDZIOS4wMTg5NEM5LjAwNjM5IDYuMTY1MDIgOSA2LjMzMTc2IDkgNi41QzkgOC44MTU3NyAxMC4yMTEgMTAuODQ4NyAxMi4wMzQzIDEySDlWMTRIMTZWMTIuOTgxMUMxNi41NzAzIDEyLjkzNzcgMTcuMTIgMTIuODIwNyAxNy42Mzk2IDEyLjYzOTZMMTggMTNWMjBINFY2Wk04IDhINlYxMEg4VjhaTTYgMTJIOFYxNEg2VjEyWk04IDE2SDZWMThIOFYxNlpNOSAxNkgxNlYxOEg5VjE2WiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-paste: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE5IDJoLTQuMThDMTQuNC44NCAxMy4zIDAgMTIgMGMtMS4zIDAtMi40Ljg0LTIuODIgMkg1Yy0xLjEgMC0yIC45LTIgMnYxNmMwIDEuMS45IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjRjMC0xLjEtLjktMi0yLTJ6bS03IDBjLjU1IDAgMSAuNDUgMSAxcy0uNDUgMS0xIDEtMS0uNDUtMS0xIC40NS0xIDEtMXptNyAxOEg1VjRoMnYzaDEwVjRoMnYxNnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-pdf: url(data:image/svg+xml;base64,PHN2ZwogICB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyMiAyMiIgd2lkdGg9IjE2Ij4KICAgIDxwYXRoIHRyYW5zZm9ybT0icm90YXRlKDQ1KSIgY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI0ZGMkEyQSIKICAgICAgIGQ9Im0gMjIuMzQ0MzY5LC0zLjAxNjM2NDIgaCA1LjYzODYwNCB2IDEuNTc5MjQzMyBoIC0zLjU0OTIyNyB2IDEuNTA4NjkyOTkgaCAzLjMzNzU3NiBWIDEuNjUwODE1NCBoIC0zLjMzNzU3NiB2IDMuNDM1MjYxMyBoIC0yLjA4OTM3NyB6IG0gLTcuMTM2NDQ0LDEuNTc5MjQzMyB2IDQuOTQzOTU0MyBoIDAuNzQ4OTIgcSAxLjI4MDc2MSwwIDEuOTUzNzAzLC0wLjYzNDk1MzUgMC42NzgzNjksLTAuNjM0OTUzNSAwLjY3ODM2OSwtMS44NDUxNjQxIDAsLTEuMjA0NzgzNTUgLTAuNjcyOTQyLC0xLjgzNDMxMDExIC0wLjY3Mjk0MiwtMC42Mjk1MjY1OSAtMS45NTkxMywtMC42Mjk1MjY1OSB6IG0gLTIuMDg5Mzc3LC0xLjU3OTI0MzMgaCAyLjIwMzM0MyBxIDEuODQ1MTY0LDAgMi43NDYwMzksMC4yNjU5MjA3IDAuOTA2MzAxLDAuMjYwNDkzNyAxLjU1MjEwOCwwLjg5MDAyMDMgMC41Njk4MywwLjU0ODEyMjMgMC44NDY2MDUsMS4yNjQ0ODAwNiAwLjI3Njc3NCwwLjcxNjM1NzgxIDAuMjc2Nzc0LDEuNjIyNjU4OTQgMCwwLjkxNzE1NTEgLTAuMjc2Nzc0LDEuNjM4OTM5OSAtMC4yNzY3NzUsMC43MTYzNTc4IC0wLjg0NjYwNSwxLjI2NDQ4IC0wLjY1MTIzNCwwLjYyOTUyNjYgLTEuNTYyOTYyLDAuODk1NDQ3MyAtMC45MTE3MjgsMC4yNjA0OTM3IC0yLjczNTE4NSwwLjI2MDQ5MzcgaCAtMi4yMDMzNDMgeiBtIC04LjE0NTg1NjUsMCBoIDMuNDY3ODIzIHEgMS41NDY2ODE2LDAgMi4zNzE1Nzg1LDAuNjg5MjIzIDAuODMwMzI0LDAuNjgzNzk2MSAwLjgzMDMyNCwxLjk1MzcwMzE0IDAsMS4yNzUzMzM5NyAtMC44MzAzMjQsMS45NjQ1NTcwNiBRIDkuOTg3MTk2MSwyLjI3NDkxNSA4LjQ0MDUxNDUsMi4yNzQ5MTUgSCA3LjA2MjA2ODQgViA1LjA4NjA3NjcgSCA0Ljk3MjY5MTUgWiBtIDIuMDg5Mzc2OSwxLjUxNDExOTkgdiAyLjI2MzAzOTQzIGggMS4xNTU5NDEgcSAwLjYwNzgxODgsMCAwLjkzODg2MjksLTAuMjkzMDU1NDcgMC4zMzEwNDQxLC0wLjI5ODQ4MjQxIDAuMzMxMDQ0MSwtMC44NDExNzc3MiAwLC0wLjU0MjY5NTMxIC0wLjMzMTA0NDEsLTAuODM1NzUwNzQgLTAuMzMxMDQ0MSwtMC4yOTMwNTU1IC0wLjkzODg2MjksLTAuMjkzMDU1NSB6IgovPgo8L3N2Zz4K);
  --jp-icon-python: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iLTEwIC0xMCAxMzEuMTYxMzYxNjk0MzM1OTQgMTMyLjM4ODk5OTkzODk2NDg0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMzA2OTk4IiBkPSJNIDU0LjkxODc4NSw5LjE5Mjc0MjFlLTQgQyA1MC4zMzUxMzIsMC4wMjIyMTcyNyA0NS45NTc4NDYsMC40MTMxMzY5NyA0Mi4xMDYyODUsMS4wOTQ2NjkzIDMwLjc2MDA2OSwzLjA5OTE3MzEgMjguNzAwMDM2LDcuMjk0NzcxNCAyOC43MDAwMzUsMTUuMDMyMTY5IHYgMTAuMjE4NzUgaCAyNi44MTI1IHYgMy40MDYyNSBoIC0yNi44MTI1IC0xMC4wNjI1IGMgLTcuNzkyNDU5LDAgLTE0LjYxNTc1ODgsNC42ODM3MTcgLTE2Ljc0OTk5OTgsMTMuNTkzNzUgLTIuNDYxODE5OTgsMTAuMjEyOTY2IC0yLjU3MTAxNTA4LDE2LjU4NjAyMyAwLDI3LjI1IDEuOTA1OTI4Myw3LjkzNzg1MiA2LjQ1NzU0MzIsMTMuNTkzNzQ4IDE0LjI0OTk5OTgsMTMuNTkzNzUgaCA5LjIxODc1IHYgLTEyLjI1IGMgMCwtOC44NDk5MDIgNy42NTcxNDQsLTE2LjY1NjI0OCAxNi43NSwtMTYuNjU2MjUgaCAyNi43ODEyNSBjIDcuNDU0OTUxLDAgMTMuNDA2MjUzLC02LjEzODE2NCAxMy40MDYyNSwtMTMuNjI1IHYgLTI1LjUzMTI1IGMgMCwtNy4yNjYzMzg2IC02LjEyOTk4LC0xMi43MjQ3NzcxIC0xMy40MDYyNSwtMTMuOTM3NDk5NyBDIDY0LjI4MTU0OCwwLjMyNzk0Mzk3IDU5LjUwMjQzOCwtMC4wMjAzNzkwMyA1NC45MTg3ODUsOS4xOTI3NDIxZS00IFogbSAtMTQuNSw4LjIxODc1MDEyNTc5IGMgMi43Njk1NDcsMCA1LjAzMTI1LDIuMjk4NjQ1NiA1LjAzMTI1LDUuMTI0OTk5NiAtMmUtNiwyLjgxNjMzNiAtMi4yNjE3MDMsNS4wOTM3NSAtNS4wMzEyNSw1LjA5Mzc1IC0yLjc3OTQ3NiwtMWUtNiAtNS4wMzEyNSwtMi4yNzc0MTUgLTUuMDMxMjUsLTUuMDkzNzUgLTEwZS03LC0yLjgyNjM1MyAyLjI1MTc3NCwtNS4xMjQ5OTk2IDUuMDMxMjUsLTUuMTI0OTk5NiB6Ii8+CiAgPHBhdGggY2xhc3M9ImpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iI2ZmZDQzYiIgZD0ibSA4NS42Mzc1MzUsMjguNjU3MTY5IHYgMTEuOTA2MjUgYyAwLDkuMjMwNzU1IC03LjgyNTg5NSwxNi45OTk5OTkgLTE2Ljc1LDE3IGggLTI2Ljc4MTI1IGMgLTcuMzM1ODMzLDAgLTEzLjQwNjI0OSw2LjI3ODQ4MyAtMTMuNDA2MjUsMTMuNjI1IHYgMjUuNTMxMjQ3IGMgMCw3LjI2NjM0NCA2LjMxODU4OCwxMS41NDAzMjQgMTMuNDA2MjUsMTMuNjI1MDA0IDguNDg3MzMxLDIuNDk1NjEgMTYuNjI2MjM3LDIuOTQ2NjMgMjYuNzgxMjUsMCA2Ljc1MDE1NSwtMS45NTQzOSAxMy40MDYyNTMsLTUuODg3NjEgMTMuNDA2MjUsLTEzLjYyNTAwNCBWIDg2LjUwMDkxOSBoIC0yNi43ODEyNSB2IC0zLjQwNjI1IGggMjYuNzgxMjUgMTMuNDA2MjU0IGMgNy43OTI0NjEsMCAxMC42OTYyNTEsLTUuNDM1NDA4IDEzLjQwNjI0MSwtMTMuNTkzNzUgMi43OTkzMywtOC4zOTg4ODYgMi42ODAyMiwtMTYuNDc1Nzc2IDAsLTI3LjI1IC0xLjkyNTc4LC03Ljc1NzQ0MSAtNS42MDM4NywtMTMuNTkzNzUgLTEzLjQwNjI0MSwtMTMuNTkzNzUgeiBtIC0xNS4wNjI1LDY0LjY1NjI1IGMgMi43Nzk0NzgsM2UtNiA1LjAzMTI1LDIuMjc3NDE3IDUuMDMxMjUsNS4wOTM3NDcgLTJlLTYsMi44MjYzNTQgLTIuMjUxNzc1LDUuMTI1MDA0IC01LjAzMTI1LDUuMTI1MDA0IC0yLjc2OTU1LDAgLTUuMDMxMjUsLTIuMjk4NjUgLTUuMDMxMjUsLTUuMTI1MDA0IDJlLTYsLTIuODE2MzMgMi4yNjE2OTcsLTUuMDkzNzQ3IDUuMDMxMjUsLTUuMDkzNzQ3IHoiLz4KPC9zdmc+Cg==);
  --jp-icon-r-kernel: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjE5NkYzIiBkPSJNNC40IDIuNWMxLjItLjEgMi45LS4zIDQuOS0uMyAyLjUgMCA0LjEuNCA1LjIgMS4zIDEgLjcgMS41IDEuOSAxLjUgMy41IDAgMi0xLjQgMy41LTIuOSA0LjEgMS4yLjQgMS43IDEuNiAyLjIgMyAuNiAxLjkgMSAzLjkgMS4zIDQuNmgtMy44Yy0uMy0uNC0uOC0xLjctMS4yLTMuN3MtMS4yLTIuNi0yLjYtMi42aC0uOXY2LjRINC40VjIuNXptMy43IDYuOWgxLjRjMS45IDAgMi45LS45IDIuOS0yLjNzLTEtMi4zLTIuOC0yLjNjLS43IDAtMS4zIDAtMS42LjJ2NC41aC4xdi0uMXoiLz4KPC9zdmc+Cg==);
  --jp-icon-react: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMTUwIDE1MCA1NDEuOSAyOTUuMyI+CiAgPGcgY2xhc3M9ImpwLWljb24tYnJhbmQyIGpwLWljb24tc2VsZWN0YWJsZSIgZmlsbD0iIzYxREFGQiI+CiAgICA8cGF0aCBkPSJNNjY2LjMgMjk2LjVjMC0zMi41LTQwLjctNjMuMy0xMDMuMS04Mi40IDE0LjQtNjMuNiA4LTExNC4yLTIwLjItMTMwLjQtNi41LTMuOC0xNC4xLTUuNi0yMi40LTUuNnYyMi4zYzQuNiAwIDguMy45IDExLjQgMi42IDEzLjYgNy44IDE5LjUgMzcuNSAxNC45IDc1LjctMS4xIDkuNC0yLjkgMTkuMy01LjEgMjkuNC0xOS42LTQuOC00MS04LjUtNjMuNS0xMC45LTEzLjUtMTguNS0yNy41LTM1LjMtNDEuNi01MCAzMi42LTMwLjMgNjMuMi00Ni45IDg0LTQ2LjlWNzhjLTI3LjUgMC02My41IDE5LjYtOTkuOSA1My42LTM2LjQtMzMuOC03Mi40LTUzLjItOTkuOS01My4ydjIyLjNjMjAuNyAwIDUxLjQgMTYuNSA4NCA0Ni42LTE0IDE0LjctMjggMzEuNC00MS4zIDQ5LjktMjIuNiAyLjQtNDQgNi4xLTYzLjYgMTEtMi4zLTEwLTQtMTkuNy01LjItMjktNC43LTM4LjIgMS4xLTY3LjkgMTQuNi03NS44IDMtMS44IDYuOS0yLjYgMTEuNS0yLjZWNzguNWMtOC40IDAtMTYgMS44LTIyLjYgNS42LTI4LjEgMTYuMi0zNC40IDY2LjctMTkuOSAxMzAuMS02Mi4yIDE5LjItMTAyLjcgNDkuOS0xMDIuNyA4Mi4zIDAgMzIuNSA0MC43IDYzLjMgMTAzLjEgODIuNC0xNC40IDYzLjYtOCAxMTQuMiAyMC4yIDEzMC40IDYuNSAzLjggMTQuMSA1LjYgMjIuNSA1LjYgMjcuNSAwIDYzLjUtMTkuNiA5OS45LTUzLjYgMzYuNCAzMy44IDcyLjQgNTMuMiA5OS45IDUzLjIgOC40IDAgMTYtMS44IDIyLjYtNS42IDI4LjEtMTYuMiAzNC40LTY2LjcgMTkuOS0xMzAuMSA2Mi0xOS4xIDEwMi41LTQ5LjkgMTAyLjUtODIuM3ptLTEzMC4yLTY2LjdjLTMuNyAxMi45LTguMyAyNi4yLTEzLjUgMzkuNS00LjEtOC04LjQtMTYtMTMuMS0yNC00LjYtOC05LjUtMTUuOC0xNC40LTIzLjQgMTQuMiAyLjEgMjcuOSA0LjcgNDEgNy45em0tNDUuOCAxMDYuNWMtNy44IDEzLjUtMTUuOCAyNi4zLTI0LjEgMzguMi0xNC45IDEuMy0zMCAyLTQ1LjIgMi0xNS4xIDAtMzAuMi0uNy00NS0xLjktOC4zLTExLjktMTYuNC0yNC42LTI0LjItMzgtNy42LTEzLjEtMTQuNS0yNi40LTIwLjgtMzkuOCA2LjItMTMuNCAxMy4yLTI2LjggMjAuNy0zOS45IDcuOC0xMy41IDE1LjgtMjYuMyAyNC4xLTM4LjIgMTQuOS0xLjMgMzAtMiA0NS4yLTIgMTUuMSAwIDMwLjIuNyA0NSAxLjkgOC4zIDExLjkgMTYuNCAyNC42IDI0LjIgMzggNy42IDEzLjEgMTQuNSAyNi40IDIwLjggMzkuOC02LjMgMTMuNC0xMy4yIDI2LjgtMjAuNyAzOS45em0zMi4zLTEzYzUuNCAxMy40IDEwIDI2LjggMTMuOCAzOS44LTEzLjEgMy4yLTI2LjkgNS45LTQxLjIgOCA0LjktNy43IDkuOC0xNS42IDE0LjQtMjMuNyA0LjYtOCA4LjktMTYuMSAxMy0yNC4xek00MjEuMiA0MzBjLTkuMy05LjYtMTguNi0yMC4zLTI3LjgtMzIgOSAuNCAxOC4yLjcgMjcuNS43IDkuNCAwIDE4LjctLjIgMjcuOC0uNy05IDExLjctMTguMyAyMi40LTI3LjUgMzJ6bS03NC40LTU4LjljLTE0LjItMi4xLTI3LjktNC43LTQxLTcuOSAzLjctMTIuOSA4LjMtMjYuMiAxMy41LTM5LjUgNC4xIDggOC40IDE2IDEzLjEgMjQgNC43IDggOS41IDE1LjggMTQuNCAyMy40ek00MjAuNyAxNjNjOS4zIDkuNiAxOC42IDIwLjMgMjcuOCAzMi05LS40LTE4LjItLjctMjcuNS0uNy05LjQgMC0xOC43LjItMjcuOC43IDktMTEuNyAxOC4zLTIyLjQgMjcuNS0zMnptLTc0IDU4LjljLTQuOSA3LjctOS44IDE1LjYtMTQuNCAyMy43LTQuNiA4LTguOSAxNi0xMyAyNC01LjQtMTMuNC0xMC0yNi44LTEzLjgtMzkuOCAxMy4xLTMuMSAyNi45LTUuOCA0MS4yLTcuOXptLTkwLjUgMTI1LjJjLTM1LjQtMTUuMS01OC4zLTM0LjktNTguMy01MC42IDAtMTUuNyAyMi45LTM1LjYgNTguMy01MC42IDguNi0zLjcgMTgtNyAyNy43LTEwLjEgNS43IDE5LjYgMTMuMiA0MCAyMi41IDYwLjktOS4yIDIwLjgtMTYuNiA0MS4xLTIyLjIgNjAuNi05LjktMy4xLTE5LjMtNi41LTI4LTEwLjJ6TTMxMCA0OTBjLTEzLjYtNy44LTE5LjUtMzcuNS0xNC45LTc1LjcgMS4xLTkuNCAyLjktMTkuMyA1LjEtMjkuNCAxOS42IDQuOCA0MSA4LjUgNjMuNSAxMC45IDEzLjUgMTguNSAyNy41IDM1LjMgNDEuNiA1MC0zMi42IDMwLjMtNjMuMiA0Ni45LTg0IDQ2LjktNC41LS4xLTguMy0xLTExLjMtMi43em0yMzcuMi03Ni4yYzQuNyAzOC4yLTEuMSA2Ny45LTE0LjYgNzUuOC0zIDEuOC02LjkgMi42LTExLjUgMi42LTIwLjcgMC01MS40LTE2LjUtODQtNDYuNiAxNC0xNC43IDI4LTMxLjQgNDEuMy00OS45IDIyLjYtMi40IDQ0LTYuMSA2My42LTExIDIuMyAxMC4xIDQuMSAxOS44IDUuMiAyOS4xem0zOC41LTY2LjdjLTguNiAzLjctMTggNy0yNy43IDEwLjEtNS43LTE5LjYtMTMuMi00MC0yMi41LTYwLjkgOS4yLTIwLjggMTYuNi00MS4xIDIyLjItNjAuNiA5LjkgMy4xIDE5LjMgNi41IDI4LjEgMTAuMiAzNS40IDE1LjEgNTguMyAzNC45IDU4LjMgNTAuNi0uMSAxNS43LTIzIDM1LjYtNTguNCA1MC42ek0zMjAuOCA3OC40eiIvPgogICAgPGNpcmNsZSBjeD0iNDIwLjkiIGN5PSIyOTYuNSIgcj0iNDUuNyIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-redo: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgd2lkdGg9IjE2Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgICA8cGF0aCBkPSJNMCAwaDI0djI0SDB6IiBmaWxsPSJub25lIi8+PHBhdGggZD0iTTE4LjQgMTAuNkMxNi41NSA4Ljk5IDE0LjE1IDggMTEuNSA4Yy00LjY1IDAtOC41OCAzLjAzLTkuOTYgNy4yMkwzLjkgMTZjMS4wNS0zLjE5IDQuMDUtNS41IDcuNi01LjUgMS45NSAwIDMuNzMuNzIgNS4xMiAxLjg4TDEzIDE2aDlWN2wtMy42IDMuNnoiLz4KICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-refresh: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDE4IDE4Ij4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTkgMTMuNWMtMi40OSAwLTQuNS0yLjAxLTQuNS00LjVTNi41MSA0LjUgOSA0LjVjMS4yNCAwIDIuMzYuNTIgMy4xNyAxLjMzTDEwIDhoNVYzbC0xLjc2IDEuNzZDMTIuMTUgMy42OCAxMC42NiAzIDkgMyA1LjY5IDMgMy4wMSA1LjY5IDMuMDEgOVM1LjY5IDE1IDkgMTVjMi45NyAwIDUuNDMtMi4xNiA1LjktNWgtMS41MmMtLjQ2IDItMi4yNCAzLjUtNC4zOCAzLjV6Ii8+CiAgICA8L2c+Cjwvc3ZnPgo=);
  --jp-icon-regex: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KICA8ZyBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiM0MTQxNDEiPgogICAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiAgPC9nPgoKICA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiBmaWxsPSIjRkZGIj4KICAgIDxjaXJjbGUgY2xhc3M9InN0MiIgY3g9IjUuNSIgY3k9IjE0LjUiIHI9IjEuNSIvPgogICAgPHJlY3QgeD0iMTIiIHk9IjQiIGNsYXNzPSJzdDIiIHdpZHRoPSIxIiBoZWlnaHQ9IjgiLz4KICAgIDxyZWN0IHg9IjguNSIgeT0iNy41IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjg2NiAtMC41IDAuNSAwLjg2NiAtMi4zMjU1IDcuMzIxOSkiIGNsYXNzPSJzdDIiIHdpZHRoPSI4IiBoZWlnaHQ9IjEiLz4KICAgIDxyZWN0IHg9IjEyIiB5PSI0IiB0cmFuc2Zvcm09Im1hdHJpeCgwLjUgLTAuODY2IDAuODY2IDAuNSAtMC42Nzc5IDE0LjgyNTIpIiBjbGFzcz0ic3QyIiB3aWR0aD0iMSIgaGVpZ2h0PSI4Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-run: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTggNXYxNGwxMS03eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-running: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDUxMiA1MTIiPgogIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICA8cGF0aCBkPSJNMjU2IDhDMTE5IDggOCAxMTkgOCAyNTZzMTExIDI0OCAyNDggMjQ4IDI0OC0xMTEgMjQ4LTI0OFMzOTMgOCAyNTYgOHptOTYgMzI4YzAgOC44LTcuMiAxNi0xNiAxNkgxNzZjLTguOCAwLTE2LTcuMi0xNi0xNlYxNzZjMC04LjggNy4yLTE2IDE2LTE2aDE2MGM4LjggMCAxNiA3LjIgMTYgMTZ2MTYweiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-save: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTE3IDNINWMtMS4xMSAwLTIgLjktMiAydjE0YzAgMS4xLjg5IDIgMiAyaDE0YzEuMSAwIDItLjkgMi0yVjdsLTQtNHptLTUgMTZjLTEuNjYgMC0zLTEuMzQtMy0zczEuMzQtMyAzLTMgMyAxLjM0IDMgMy0xLjM0IDMtMyAzem0zLTEwSDVWNWgxMHY0eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-search: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMTggMTgiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjEsMTAuOWgtMC43bC0wLjItMC4yYzAuOC0wLjksMS4zLTIuMiwxLjMtMy41YzAtMy0yLjQtNS40LTUuNC01LjRTMS44LDQuMiwxLjgsNy4xczIuNCw1LjQsNS40LDUuNCBjMS4zLDAsMi41LTAuNSwzLjUtMS4zbDAuMiwwLjJ2MC43bDQuMSw0LjFsMS4yLTEuMkwxMi4xLDEwLjl6IE03LjEsMTAuOWMtMi4xLDAtMy43LTEuNy0zLjctMy43czEuNy0zLjcsMy43LTMuN3MzLjcsMS43LDMuNywzLjcgUzkuMiwxMC45LDcuMSwxMC45eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-settings: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIiBkPSJNMTkuNDMgMTIuOThjLjA0LS4zMi4wNy0uNjQuMDctLjk4cy0uMDMtLjY2LS4wNy0uOThsMi4xMS0xLjY1Yy4xOS0uMTUuMjQtLjQyLjEyLS42NGwtMi0zLjQ2Yy0uMTItLjIyLS4zOS0uMy0uNjEtLjIybC0yLjQ5IDFjLS41Mi0uNC0xLjA4LS43My0xLjY5LS45OGwtLjM4LTIuNjVBLjQ4OC40ODggMCAwMDE0IDJoLTRjLS4yNSAwLS40Ni4xOC0uNDkuNDJsLS4zOCAyLjY1Yy0uNjEuMjUtMS4xNy41OS0xLjY5Ljk4bC0yLjQ5LTFjLS4yMy0uMDktLjQ5IDAtLjYxLjIybC0yIDMuNDZjLS4xMy4yMi0uMDcuNDkuMTIuNjRsMi4xMSAxLjY1Yy0uMDQuMzItLjA3LjY1LS4wNy45OHMuMDMuNjYuMDcuOThsLTIuMTEgMS42NWMtLjE5LjE1LS4yNC40Mi0uMTIuNjRsMiAzLjQ2Yy4xMi4yMi4zOS4zLjYxLjIybDIuNDktMWMuNTIuNCAxLjA4LjczIDEuNjkuOThsLjM4IDIuNjVjLjAzLjI0LjI0LjQyLjQ5LjQyaDRjLjI1IDAgLjQ2LS4xOC40OS0uNDJsLjM4LTIuNjVjLjYxLS4yNSAxLjE3LS41OSAxLjY5LS45OGwyLjQ5IDFjLjIzLjA5LjQ5IDAgLjYxLS4yMmwyLTMuNDZjLjEyLS4yMi4wNy0uNDktLjEyLS42NGwtMi4xMS0xLjY1ek0xMiAxNS41Yy0xLjkzIDAtMy41LTEuNTctMy41LTMuNXMxLjU3LTMuNSAzLjUtMy41IDMuNSAxLjU3IDMuNSAzLjUtMS41NyAzLjUtMy41IDMuNXoiLz4KPC9zdmc+Cg==);
  --jp-icon-share: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTSAxOCAyIEMgMTYuMzU0OTkgMiAxNSAzLjM1NDk5MDQgMTUgNSBDIDE1IDUuMTkwOTUyOSAxNS4wMjE3OTEgNS4zNzcxMjI0IDE1LjA1NjY0MSA1LjU1ODU5MzggTCA3LjkyMTg3NSA5LjcyMDcwMzEgQyA3LjM5ODUzOTkgOS4yNzc4NTM5IDYuNzMyMDc3MSA5IDYgOSBDIDQuMzU0OTkwNCA5IDMgMTAuMzU0OTkgMyAxMiBDIDMgMTMuNjQ1MDEgNC4zNTQ5OTA0IDE1IDYgMTUgQyA2LjczMjA3NzEgMTUgNy4zOTg1Mzk5IDE0LjcyMjE0NiA3LjkyMTg3NSAxNC4yNzkyOTcgTCAxNS4wNTY2NDEgMTguNDM5NDUzIEMgMTUuMDIxNTU1IDE4LjYyMTUxNCAxNSAxOC44MDgzODYgMTUgMTkgQyAxNSAyMC42NDUwMSAxNi4zNTQ5OSAyMiAxOCAyMiBDIDE5LjY0NTAxIDIyIDIxIDIwLjY0NTAxIDIxIDE5IEMgMjEgMTcuMzU0OTkgMTkuNjQ1MDEgMTYgMTggMTYgQyAxNy4yNjc0OCAxNiAxNi42MDE1OTMgMTYuMjc5MzI4IDE2LjA3ODEyNSAxNi43MjI2NTYgTCA4Ljk0MzM1OTQgMTIuNTU4NTk0IEMgOC45NzgyMDk1IDEyLjM3NzEyMiA5IDEyLjE5MDk1MyA5IDEyIEMgOSAxMS44MDkwNDcgOC45NzgyMDk1IDExLjYyMjg3OCA4Ljk0MzM1OTQgMTEuNDQxNDA2IEwgMTYuMDc4MTI1IDcuMjc5Mjk2OSBDIDE2LjYwMTQ2IDcuNzIyMTQ2MSAxNy4yNjc5MjMgOCAxOCA4IEMgMTkuNjQ1MDEgOCAyMSA2LjY0NTAwOTYgMjEgNSBDIDIxIDMuMzU0OTkwNCAxOS42NDUwMSAyIDE4IDIgeiBNIDE4IDQgQyAxOC41NjQxMjkgNCAxOSA0LjQzNTg3MDYgMTkgNSBDIDE5IDUuNTY0MTI5NCAxOC41NjQxMjkgNiAxOCA2IEMgMTcuNDM1ODcxIDYgMTcgNS41NjQxMjk0IDE3IDUgQyAxNyA0LjQzNTg3MDYgMTcuNDM1ODcxIDQgMTggNCB6IE0gNiAxMSBDIDYuNTY0MTI5NCAxMSA3IDExLjQzNTg3MSA3IDEyIEMgNyAxMi41NjQxMjkgNi41NjQxMjk0IDEzIDYgMTMgQyA1LjQzNTg3MDYgMTMgNSAxMi41NjQxMjkgNSAxMiBDIDUgMTEuNDM1ODcxIDUuNDM1ODcwNiAxMSA2IDExIHogTSAxOCAxOCBDIDE4LjU2NDEyOSAxOCAxOSAxOC40MzU4NzEgMTkgMTkgQyAxOSAxOS41NjQxMjkgMTguNTY0MTI5IDIwIDE4IDIwIEMgMTcuNDM1ODcxIDIwIDE3IDE5LjU2NDEyOSAxNyAxOSBDIDE3IDE4LjQzNTg3MSAxNy40MzU4NzEgMTggMTggMTggeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-spreadsheet: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8cGF0aCBjbGFzcz0ianAtaWNvbi1jb250cmFzdDEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNENBRjUwIiBkPSJNMi4yIDIuMnYxNy42aDE3LjZWMi4ySDIuMnptMTUuNCA3LjdoLTUuNVY0LjRoNS41djUuNXpNOS45IDQuNHY1LjVINC40VjQuNGg1LjV6bS01LjUgNy43aDUuNXY1LjVINC40di01LjV6bTcuNyA1LjV2LTUuNWg1LjV2NS41aC01LjV6Ii8+Cjwvc3ZnPgo=);
  --jp-icon-stop: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik02IDZoMTJ2MTJINnoiLz4KICAgIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tab: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTIxIDNIM2MtMS4xIDAtMiAuOS0yIDJ2MTRjMCAxLjEuOSAyIDIgMmgxOGMxLjEgMCAyLS45IDItMlY1YzAtMS4xLS45LTItMi0yem0wIDE2SDNWNWgxMHY0aDh2MTB6Ii8+CiAgPC9nPgo8L3N2Zz4K);
  --jp-icon-table-rows: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMSw4SDNWNGgxOFY4eiBNMjEsMTBIM3Y0aDE4VjEweiBNMjEsMTZIM3Y0aDE4VjE2eiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-tag: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjgiIGhlaWdodD0iMjgiIHZpZXdCb3g9IjAgMCA0MyAyOCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KCTxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CgkJPHBhdGggZD0iTTI4LjgzMzIgMTIuMzM0TDMyLjk5OTggMTYuNTAwN0wzNy4xNjY1IDEyLjMzNEgyOC44MzMyWiIvPgoJCTxwYXRoIGQ9Ik0xNi4yMDk1IDIxLjYxMDRDMTUuNjg3MyAyMi4xMjk5IDE0Ljg0NDMgMjIuMTI5OSAxNC4zMjQ4IDIxLjYxMDRMNi45ODI5IDE0LjcyNDVDNi41NzI0IDE0LjMzOTQgNi4wODMxMyAxMy42MDk4IDYuMDQ3ODYgMTMuMDQ4MkM1Ljk1MzQ3IDExLjUyODggNi4wMjAwMiA4LjYxOTQ0IDYuMDY2MjEgNy4wNzY5NUM2LjA4MjgxIDYuNTE0NzcgNi41NTU0OCA2LjA0MzQ3IDcuMTE4MDQgNi4wMzA1NUM5LjA4ODYzIDUuOTg0NzMgMTMuMjYzOCA1LjkzNTc5IDEzLjY1MTggNi4zMjQyNUwyMS43MzY5IDEzLjYzOUMyMi4yNTYgMTQuMTU4NSAyMS43ODUxIDE1LjQ3MjQgMjEuMjYyIDE1Ljk5NDZMMTYuMjA5NSAyMS42MTA0Wk05Ljc3NTg1IDguMjY1QzkuMzM1NTEgNy44MjU2NiA4LjYyMzUxIDcuODI1NjYgOC4xODI4IDguMjY1QzcuNzQzNDYgOC43MDU3MSA3Ljc0MzQ2IDkuNDE3MzMgOC4xODI4IDkuODU2NjdDOC42MjM4MiAxMC4yOTY0IDkuMzM1ODIgMTAuMjk2NCA5Ljc3NTg1IDkuODU2NjdDMTAuMjE1NiA5LjQxNzMzIDEwLjIxNTYgOC43MDUzMyA5Ljc3NTg1IDguMjY1WiIvPgoJPC9nPgo8L3N2Zz4K);
  --jp-icon-terminal: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiA+CiAgICA8cmVjdCBjbGFzcz0ianAtdGVybWluYWwtaWNvbi1iYWNrZ3JvdW5kLWNvbG9yIGpwLWljb24tc2VsZWN0YWJsZSIgd2lkdGg9IjIwIiBoZWlnaHQ9IjIwIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSgyIDIpIiBmaWxsPSIjMzMzMzMzIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtdGVybWluYWwtaWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUtaW52ZXJzZSIgZD0iTTUuMDU2NjQgOC43NjE3MkM1LjA1NjY0IDguNTk3NjYgNS4wMzEyNSA4LjQ1MzEyIDQuOTgwNDcgOC4zMjgxMkM0LjkzMzU5IDguMTk5MjIgNC44NTU0NyA4LjA4MjAzIDQuNzQ2MDkgNy45NzY1NkM0LjY0MDYyIDcuODcxMDkgNC41IDcuNzc1MzkgNC4zMjQyMiA3LjY4OTQ1QzQuMTUyMzQgNy41OTk2MSAzLjk0MzM2IDcuNTExNzIgMy42OTcyNyA3LjQyNTc4QzMuMzAyNzMgNy4yODUxNiAyLjk0MzM2IDcuMTM2NzIgMi42MTkxNCA2Ljk4MDQ3QzIuMjk0OTIgNi44MjQyMiAyLjAxNzU4IDYuNjQyNTggMS43ODcxMSA2LjQzNTU1QzEuNTYwNTUgNi4yMjg1MiAxLjM4NDc3IDUuOTg4MjggMS4yNTk3NyA1LjcxNDg0QzEuMTM0NzcgNS40Mzc1IDEuMDcyMjcgNS4xMDkzOCAxLjA3MjI3IDQuNzMwNDdDMS4wNzIyNyA0LjM5ODQ0IDEuMTI4OTEgNC4wOTU3IDEuMjQyMTkgMy44MjIyN0MxLjM1NTQ3IDMuNTQ0OTIgMS41MTU2MiAzLjMwNDY5IDEuNzIyNjYgMy4xMDE1NkMxLjkyOTY5IDIuODk4NDQgMi4xNzk2OSAyLjczNDM3IDIuNDcyNjYgMi42MDkzOEMyLjc2NTYyIDIuNDg0MzggMy4wOTE4IDIuNDA0MyAzLjQ1MTE3IDIuMzY5MTRWMS4xMDkzOEg0LjM4ODY3VjIuMzgwODZDNC43NDAyMyAyLjQyNzczIDUuMDU2NjQgMi41MjM0NCA1LjMzNzg5IDIuNjY3OTdDNS42MTkxNCAyLjgxMjUgNS44NTc0MiAzLjAwMTk1IDYuMDUyNzMgMy4yMzYzM0M2LjI1MTk1IDMuNDY2OCA2LjQwNDMgMy43NDAyMyA2LjUwOTc3IDQuMDU2NjRDNi42MTkxNCA0LjM2OTE0IDYuNjczODMgNC43MjA3IDYuNjczODMgNS4xMTEzM0g1LjA0NDkyQzUuMDQ0OTIgNC42Mzg2NyA0LjkzNzUgNC4yODEyNSA0LjcyMjY2IDQuMDM5MDZDNC41MDc4MSAzLjc5Mjk3IDQuMjE2OCAzLjY2OTkyIDMuODQ5NjEgMy42Njk5MkMzLjY1MDM5IDMuNjY5OTIgMy40NzY1NiAzLjY5NzI3IDMuMzI4MTIgMy43NTE5NUMzLjE4MzU5IDMuODAyNzMgMy4wNjQ0NSAzLjg3Njk1IDIuOTcwNyAzLjk3NDYxQzIuODc2OTUgNC4wNjgzNiAyLjgwNjY0IDQuMTc5NjkgMi43NTk3NyA0LjMwODU5QzIuNzE2OCA0LjQzNzUgMi42OTUzMSA0LjU3ODEyIDIuNjk1MzEgNC43MzA0N0MyLjY5NTMxIDQuODgyODEgMi43MTY4IDUuMDE5NTMgMi43NTk3NyA1LjE0MDYyQzIuODA2NjQgNS4yNTc4MSAyLjg4MjgxIDUuMzY3MTkgMi45ODgyOCA1LjQ2ODc1QzMuMDk3NjYgNS41NzAzMSAzLjI0MDIzIDUuNjY3OTcgMy40MTYwMiA1Ljc2MTcyQzMuNTkxOCA1Ljg1MTU2IDMuODEwNTUgNS45NDMzNiA0LjA3MjI3IDYuMDM3MTFDNC40NjY4IDYuMTg1NTUgNC44MjQyMiA2LjMzOTg0IDUuMTQ0NTMgNi41QzUuNDY0ODQgNi42NTYyNSA1LjczODI4IDYuODM5ODQgNS45NjQ4NCA3LjA1MDc4QzYuMTk1MzEgNy4yNTc4MSA2LjM3MTA5IDcuNSA2LjQ5MjE5IDcuNzc3MzRDNi42MTcxOSA4LjA1MDc4IDYuNjc5NjkgOC4zNzUgNi42Nzk2OSA4Ljc1QzYuNjc5NjkgOS4wOTM3NSA2LjYyMzA1IDkuNDA0MyA2LjUwOTc3IDkuNjgxNjRDNi4zOTY0OCA5Ljk1NTA4IDYuMjM0MzggMTAuMTkxNCA2LjAyMzQ0IDEwLjM5MDZDNS44MTI1IDEwLjU4OTggNS41NTg1OSAxMC43NSA1LjI2MTcyIDEwLjg3MTFDNC45NjQ4NCAxMC45ODgzIDQuNjMyODEgMTEuMDY0NSA0LjI2NTYyIDExLjA5OTZWMTIuMjQ4SDMuMzMzOThWMTEuMDk5NkMzLjAwMTk1IDExLjA2ODQgMi42Nzk2OSAxMC45OTYxIDIuMzY3MTkgMTAuODgyOEMyLjA1NDY5IDEwLjc2NTYgMS43NzczNCAxMC41OTc3IDEuNTM1MTYgMTAuMzc4OUMxLjI5Njg4IDEwLjE2MDIgMS4xMDU0NyA5Ljg4NDc3IDAuOTYwOTM4IDkuNTUyNzNDMC44MTY0MDYgOS4yMTY4IDAuNzQ0MTQxIDguODE0NDUgMC43NDQxNDEgOC4zNDU3SDIuMzc4OTFDMi4zNzg5MSA4LjYyNjk1IDIuNDE5OTIgOC44NjMyOCAyLjUwMTk1IDkuMDU0NjlDMi41ODM5OCA5LjI0MjE5IDIuNjg5NDUgOS4zOTI1OCAyLjgxODM2IDkuNTA1ODZDMi45NTExNyA5LjYxNTIzIDMuMTAxNTYgOS42OTMzNiAzLjI2OTUzIDkuNzQwMjNDMy40Mzc1IDkuNzg3MTEgMy42MDkzOCA5LjgxMDU1IDMuNzg1MTYgOS44MTA1NUM0LjIwMzEyIDkuODEwNTUgNC41MTk1MyA5LjcxMjg5IDQuNzM0MzggOS41MTc1OEM0Ljk0OTIyIDkuMzIyMjcgNS4wNTY2NCA5LjA3MDMxIDUuMDU2NjQgOC43NjE3MlpNMTMuNDE4IDEyLjI3MTVIOC4wNzQyMlYxMUgxMy40MThWMTIuMjcxNVoiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDMuOTUyNjQgNikiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo=);
  --jp-icon-text-editor: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8cGF0aCBjbGFzcz0ianAtdGV4dC1lZGl0b3ItaWNvbi1jb2xvciBqcC1pY29uLXNlbGVjdGFibGUiIGZpbGw9IiM2MTYxNjEiIGQ9Ik0xNSAxNUgzdjJoMTJ2LTJ6bTAtOEgzdjJoMTJWN3pNMyAxM2gxOHYtMkgzdjJ6bTAgOGgxOHYtMkgzdjJ6TTMgM3YyaDE4VjNIM3oiLz4KPC9zdmc+Cg==);
  --jp-icon-toc: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0Ij4KICA8ZyBjbGFzcz0ianAtaWNvbjMganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjNjE2MTYxIj4KICAgIDxwYXRoIGQ9Ik03LDVIMjFWN0g3VjVNNywxM1YxMUgyMVYxM0g3TTQsNC41QTEuNSwxLjUgMCAwLDEgNS41LDZBMS41LDEuNSAwIDAsMSA0LDcuNUExLjUsMS41IDAgMCwxIDIuNSw2QTEuNSwxLjUgMCAwLDEgNCw0LjVNNCwxMC41QTEuNSwxLjUgMCAwLDEgNS41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMy41QTEuNSwxLjUgMCAwLDEgMi41LDEyQTEuNSwxLjUgMCAwLDEgNCwxMC41TTcsMTlWMTdIMjFWMTlIN000LDE2LjVBMS41LDEuNSAwIDAsMSA1LjUsMThBMS41LDEuNSAwIDAsMSA0LDE5LjVBMS41LDEuNSAwIDAsMSAyLjUsMThBMS41LDEuNSAwIDAsMSA0LDE2LjVaIiAvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-tree-view: url(data:image/svg+xml;base64,PHN2ZyBoZWlnaHQ9IjI0IiB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICAgIDxnIGNsYXNzPSJqcC1pY29uMyIgZmlsbD0iIzYxNjE2MSI+CiAgICAgICAgPHBhdGggZD0iTTAgMGgyNHYyNEgweiIgZmlsbD0ibm9uZSIvPgogICAgICAgIDxwYXRoIGQ9Ik0yMiAxMVYzaC03djNIOVYzSDJ2OGg3VjhoMnYxMGg0djNoN3YtOGgtN3YzaC0yVjhoMnYzeiIvPgogICAgPC9nPgo8L3N2Zz4K);
  --jp-icon-trusted: url(data:image/svg+xml;base64,PHN2ZyBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDI0IDI1Ij4KICAgIDxwYXRoIGNsYXNzPSJqcC1pY29uMiIgc3Ryb2tlPSIjMzMzMzMzIiBzdHJva2Utd2lkdGg9IjIiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDIgMykiIGQ9Ik0xLjg2MDk0IDExLjQ0MDlDMC44MjY0NDggOC43NzAyNyAwLjg2Mzc3OSA2LjA1NzY0IDEuMjQ5MDcgNC4xOTkzMkMyLjQ4MjA2IDMuOTMzNDcgNC4wODA2OCAzLjQwMzQ3IDUuNjAxMDIgMi44NDQ5QzcuMjM1NDkgMi4yNDQ0IDguODU2NjYgMS41ODE1IDkuOTg3NiAxLjA5NTM5QzExLjA1OTcgMS41ODM0MSAxMi42MDk0IDIuMjQ0NCAxNC4yMTggMi44NDMzOUMxNS43NTAzIDMuNDEzOTQgMTcuMzk5NSAzLjk1MjU4IDE4Ljc1MzkgNC4yMTM4NUMxOS4xMzY0IDYuMDcxNzcgMTkuMTcwOSA4Ljc3NzIyIDE4LjEzOSAxMS40NDA5QzE3LjAzMDMgMTQuMzAzMiAxNC42NjY4IDE3LjE4NDQgOS45OTk5OSAxOC45MzU0QzUuMzMzMiAxNy4xODQ0IDIuOTY5NjggMTQuMzAzMiAxLjg2MDk0IDExLjQ0MDlaIi8+CiAgICA8cGF0aCBjbGFzcz0ianAtaWNvbjIiIGZpbGw9IiMzMzMzMzMiIHN0cm9rZT0iIzMzMzMzMyIgdHJhbnNmb3JtPSJ0cmFuc2xhdGUoOCA5Ljg2NzE5KSIgZD0iTTIuODYwMTUgNC44NjUzNUwwLjcyNjU0OSAyLjk5OTU5TDAgMy42MzA0NUwyLjg2MDE1IDYuMTMxNTdMOCAwLjYzMDg3Mkw3LjI3ODU3IDBMMi44NjAxNSA0Ljg2NTM1WiIvPgo8L3N2Zz4K);
  --jp-icon-undo: url(data:image/svg+xml;base64,PHN2ZyB2aWV3Qm94PSIwIDAgMjQgMjQiIHdpZHRoPSIxNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTEyLjUgOGMtMi42NSAwLTUuMDUuOTktNi45IDIuNkwyIDd2OWg5bC0zLjYyLTMuNjJjMS4zOS0xLjE2IDMuMTYtMS44OCA1LjEyLTEuODggMy41NCAwIDYuNTUgMi4zMSA3LjYgNS41bDIuMzctLjc4QzIxLjA4IDExLjAzIDE3LjE1IDggMTIuNSA4eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-user: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMTYiIHZpZXdCb3g9IjAgMCAyNCAyNCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KICA8ZyBjbGFzcz0ianAtaWNvbjMiIGZpbGw9IiM2MTYxNjEiPgogICAgPHBhdGggZD0iTTE2IDdhNCA0IDAgMTEtOCAwIDQgNCAwIDAxOCAwek0xMiAxNGE3IDcgMCAwMC03IDdoMTRhNyA3IDAgMDAtNy03eiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-users: url(data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZlcnNpb249IjEuMSIgdmlld0JveD0iMCAwIDM2IDI0IiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogPGcgY2xhc3M9ImpwLWljb24zIiB0cmFuc2Zvcm09Im1hdHJpeCgxLjczMjcgMCAwIDEuNzMyNyAtMy42MjgyIC4wOTk1NzcpIiBmaWxsPSIjNjE2MTYxIj4KICA8cGF0aCB0cmFuc2Zvcm09Im1hdHJpeCgxLjUsMCwwLDEuNSwwLC02KSIgZD0ibTEyLjE4NiA3LjUwOThjLTEuMDUzNSAwLTEuOTc1NyAwLjU2NjUtMi40Nzg1IDEuNDEwMiAwLjc1MDYxIDAuMzEyNzcgMS4zOTc0IDAuODI2NDggMS44NzMgMS40NzI3aDMuNDg2M2MwLTEuNTkyLTEuMjg4OS0yLjg4MjgtMi44ODA5LTIuODgyOHoiLz4KICA8cGF0aCBkPSJtMjAuNDY1IDIuMzg5NWEyLjE4ODUgMi4xODg1IDAgMCAxLTIuMTg4NCAyLjE4ODUgMi4xODg1IDIuMTg4NSAwIDAgMS0yLjE4ODUtMi4xODg1IDIuMTg4NSAyLjE4ODUgMCAwIDEgMi4xODg1LTIuMTg4NSAyLjE4ODUgMi4xODg1IDAgMCAxIDIuMTg4NCAyLjE4ODV6Ii8+CiAgPHBhdGggdHJhbnNmb3JtPSJtYXRyaXgoMS41LDAsMCwxLjUsMCwtNikiIGQ9Im0zLjU4OTggOC40MjE5Yy0xLjExMjYgMC0yLjAxMzcgMC45MDExMS0yLjAxMzcgMi4wMTM3aDIuODE0NWMwLjI2Nzk3LTAuMzczMDkgMC41OTA3LTAuNzA0MzUgMC45NTg5OC0wLjk3ODUyLTAuMzQ0MzMtMC42MTY4OC0xLjAwMzEtMS4wMzUyLTEuNzU5OC0xLjAzNTJ6Ii8+CiAgPHBhdGggZD0ibTYuOTE1NCA0LjYyM2ExLjUyOTQgMS41Mjk0IDAgMCAxLTEuNTI5NCAxLjUyOTQgMS41Mjk0IDEuNTI5NCAwIDAgMS0xLjUyOTQtMS41Mjk0IDEuNTI5NCAxLjUyOTQgMCAwIDEgMS41Mjk0LTEuNTI5NCAxLjUyOTQgMS41Mjk0IDAgMCAxIDEuNTI5NCAxLjUyOTR6Ii8+CiAgPHBhdGggZD0ibTYuMTM1IDEzLjUzNWMwLTMuMjM5MiAyLjYyNTktNS44NjUgNS44NjUtNS44NjUgMy4yMzkyIDAgNS44NjUgMi42MjU5IDUuODY1IDUuODY1eiIvPgogIDxjaXJjbGUgY3g9IjEyIiBjeT0iMy43Njg1IiByPSIyLjk2ODUiLz4KIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-vega: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbjEganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjMjEyMTIxIj4KICAgIDxwYXRoIGQ9Ik0xMC42IDUuNGwyLjItMy4ySDIuMnY3LjNsNC02LjZ6Ii8+CiAgICA8cGF0aCBkPSJNMTUuOCAyLjJsLTQuNCA2LjZMNyA2LjNsLTQuOCA4djUuNWgxNy42VjIuMmgtNHptLTcgMTUuNEg1LjV2LTQuNGgzLjN2NC40em00LjQgMEg5LjhWOS44aDMuNHY3Ljh6bTQuNCAwaC0zLjRWNi41aDMuNHYxMS4xeiIvPgogIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-word: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIwIDIwIj4KIDxnIGNsYXNzPSJqcC1pY29uMiIgZmlsbD0iIzQxNDE0MSI+CiAgPHJlY3QgeD0iMiIgeT0iMiIgd2lkdGg9IjE2IiBoZWlnaHQ9IjE2Ii8+CiA8L2c+CiA8ZyBjbGFzcz0ianAtaWNvbi1hY2NlbnQyIiB0cmFuc2Zvcm09InRyYW5zbGF0ZSguNDMgLjA0MDEpIiBmaWxsPSIjZmZmIj4KICA8cGF0aCBkPSJtNC4xNCA4Ljc2cTAuMDY4Mi0xLjg5IDIuNDItMS44OSAxLjE2IDAgMS42OCAwLjQyIDAuNTY3IDAuNDEgMC41NjcgMS4xNnYzLjQ3cTAgMC40NjIgMC41MTQgMC40NjIgMC4xMDMgMCAwLjItMC4wMjMxdjAuNzE0cS0wLjM5OSAwLjEwMy0wLjY1MSAwLjEwMy0wLjQ1MiAwLTAuNjkzLTAuMjItMC4yMzEtMC4yLTAuMjg0LTAuNjYyLTAuOTU2IDAuODcyLTIgMC44NzItMC45MDMgMC0xLjQ3LTAuNDcyLTAuNTI1LTAuNDcyLTAuNTI1LTEuMjYgMC0wLjI2MiAwLjA0NTItMC40NzIgMC4wNTY3LTAuMjIgMC4xMTYtMC4zNzggMC4wNjgyLTAuMTY4IDAuMjMxLTAuMzA0IDAuMTU4LTAuMTQ3IDAuMjYyLTAuMjQyIDAuMTE2LTAuMDkxNCAwLjM2OC0wLjE2OCAwLjI2Mi0wLjA5MTQgMC4zOTktMC4xMjYgMC4xMzYtMC4wNDUyIDAuNDcyLTAuMTAzIDAuMzM2LTAuMDU3OCAwLjUwNC0wLjA3OTggMC4xNTgtMC4wMjMxIDAuNTY3LTAuMDc5OCAwLjU1Ni0wLjA2ODIgMC43NzctMC4yMjEgMC4yMi0wLjE1MiAwLjIyLTAuNDQxdi0wLjI1MnEwLTAuNDMtMC4zNTctMC42NjItMC4zMzYtMC4yMzEtMC45NzYtMC4yMzEtMC42NjIgMC0wLjk5OCAwLjI2Mi0wLjMzNiAwLjI1Mi0wLjM5OSAwLjc5OHptMS44OSAzLjY4cTAuNzg4IDAgMS4yNi0wLjQxIDAuNTA0LTAuNDIgMC41MDQtMC45MDN2LTEuMDVxLTAuMjg0IDAuMTM2LTAuODYxIDAuMjMxLTAuNTY3IDAuMDkxNC0wLjk4NyAwLjE1OC0wLjQyIDAuMDY4Mi0wLjc2NiAwLjMyNi0wLjMzNiAwLjI1Mi0wLjMzNiAwLjcwNHQwLjMwNCAwLjcwNCAwLjg2MSAwLjI1MnoiIHN0cm9rZS13aWR0aD0iMS4wNSIvPgogIDxwYXRoIGQ9Im0xMCA0LjU2aDAuOTQ1djMuMTVxMC42NTEtMC45NzYgMS44OS0wLjk3NiAxLjE2IDAgMS44OSAwLjg0IDAuNjgyIDAuODQgMC42ODIgMi4zMSAwIDEuNDctMC43MDQgMi40Mi0wLjcwNCAwLjg4Mi0xLjg5IDAuODgyLTEuMjYgMC0xLjg5LTEuMDJ2MC43NjZoLTAuODV6bTIuNjIgMy4wNHEtMC43NDYgMC0xLjE2IDAuNjQtMC40NTIgMC42My0wLjQ1MiAxLjY4IDAgMS4wNSAwLjQ1MiAxLjY4dDEuMTYgMC42M3EwLjc3NyAwIDEuMjYtMC42MyAwLjQ5NC0wLjY0IDAuNDk0LTEuNjggMC0xLjA1LTAuNDcyLTEuNjgtMC40NjItMC42NC0xLjI2LTAuNjR6IiBzdHJva2Utd2lkdGg9IjEuMDUiLz4KICA8cGF0aCBkPSJtMi43MyAxNS44IDEzLjYgMC4wMDgxYzAuMDA2OSAwIDAtMi42IDAtMi42IDAtMC4wMDc4LTEuMTUgMC0xLjE1IDAtMC4wMDY5IDAtMC4wMDgzIDEuNS0wLjAwODMgMS41LTJlLTMgLTAuMDAxNC0xMS4zLTAuMDAxNC0xMS4zLTAuMDAxNGwtMC4wMDU5Mi0xLjVjMC0wLjAwNzgtMS4xNyAwLjAwMTMtMS4xNyAwLjAwMTN6IiBzdHJva2Utd2lkdGg9Ii45NzUiLz4KIDwvZz4KPC9zdmc+Cg==);
  --jp-icon-yaml: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgdmlld0JveD0iMCAwIDIyIDIyIj4KICA8ZyBjbGFzcz0ianAtaWNvbi1jb250cmFzdDIganAtaWNvbi1zZWxlY3RhYmxlIiBmaWxsPSIjRDgxQjYwIj4KICAgIDxwYXRoIGQ9Ik03LjIgMTguNnYtNS40TDMgNS42aDMuM2wxLjQgMy4xYy4zLjkuNiAxLjYgMSAyLjUuMy0uOC42LTEuNiAxLTIuNWwxLjQtMy4xaDMuNGwtNC40IDcuNnY1LjVsLTIuOS0uMXoiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxNi41IiByPSIyLjEiLz4KICAgIDxjaXJjbGUgY2xhc3M9InN0MCIgY3g9IjE3LjYiIGN5PSIxMSIgcj0iMi4xIi8+CiAgPC9nPgo8L3N2Zz4K);
}

/* Icon CSS class declarations */

.jp-AddAboveIcon {
  background-image: var(--jp-icon-add-above);
}

.jp-AddBelowIcon {
  background-image: var(--jp-icon-add-below);
}

.jp-AddIcon {
  background-image: var(--jp-icon-add);
}

.jp-BellIcon {
  background-image: var(--jp-icon-bell);
}

.jp-BugDotIcon {
  background-image: var(--jp-icon-bug-dot);
}

.jp-BugIcon {
  background-image: var(--jp-icon-bug);
}

.jp-BuildIcon {
  background-image: var(--jp-icon-build);
}

.jp-CaretDownEmptyIcon {
  background-image: var(--jp-icon-caret-down-empty);
}

.jp-CaretDownEmptyThinIcon {
  background-image: var(--jp-icon-caret-down-empty-thin);
}

.jp-CaretDownIcon {
  background-image: var(--jp-icon-caret-down);
}

.jp-CaretLeftIcon {
  background-image: var(--jp-icon-caret-left);
}

.jp-CaretRightIcon {
  background-image: var(--jp-icon-caret-right);
}

.jp-CaretUpEmptyThinIcon {
  background-image: var(--jp-icon-caret-up-empty-thin);
}

.jp-CaretUpIcon {
  background-image: var(--jp-icon-caret-up);
}

.jp-CaseSensitiveIcon {
  background-image: var(--jp-icon-case-sensitive);
}

.jp-CheckIcon {
  background-image: var(--jp-icon-check);
}

.jp-CircleEmptyIcon {
  background-image: var(--jp-icon-circle-empty);
}

.jp-CircleIcon {
  background-image: var(--jp-icon-circle);
}

.jp-ClearIcon {
  background-image: var(--jp-icon-clear);
}

.jp-CloseIcon {
  background-image: var(--jp-icon-close);
}

.jp-CodeCheckIcon {
  background-image: var(--jp-icon-code-check);
}

.jp-CodeIcon {
  background-image: var(--jp-icon-code);
}

.jp-CollapseAllIcon {
  background-image: var(--jp-icon-collapse-all);
}

.jp-ConsoleIcon {
  background-image: var(--jp-icon-console);
}

.jp-CopyIcon {
  background-image: var(--jp-icon-copy);
}

.jp-CopyrightIcon {
  background-image: var(--jp-icon-copyright);
}

.jp-CutIcon {
  background-image: var(--jp-icon-cut);
}

.jp-DeleteIcon {
  background-image: var(--jp-icon-delete);
}

.jp-DownloadIcon {
  background-image: var(--jp-icon-download);
}

.jp-DuplicateIcon {
  background-image: var(--jp-icon-duplicate);
}

.jp-EditIcon {
  background-image: var(--jp-icon-edit);
}

.jp-EllipsesIcon {
  background-image: var(--jp-icon-ellipses);
}

.jp-ErrorIcon {
  background-image: var(--jp-icon-error);
}

.jp-ExpandAllIcon {
  background-image: var(--jp-icon-expand-all);
}

.jp-ExtensionIcon {
  background-image: var(--jp-icon-extension);
}

.jp-FastForwardIcon {
  background-image: var(--jp-icon-fast-forward);
}

.jp-FileIcon {
  background-image: var(--jp-icon-file);
}

.jp-FileUploadIcon {
  background-image: var(--jp-icon-file-upload);
}

.jp-FilterDotIcon {
  background-image: var(--jp-icon-filter-dot);
}

.jp-FilterIcon {
  background-image: var(--jp-icon-filter);
}

.jp-FilterListIcon {
  background-image: var(--jp-icon-filter-list);
}

.jp-FolderFavoriteIcon {
  background-image: var(--jp-icon-folder-favorite);
}

.jp-FolderIcon {
  background-image: var(--jp-icon-folder);
}

.jp-HomeIcon {
  background-image: var(--jp-icon-home);
}

.jp-Html5Icon {
  background-image: var(--jp-icon-html5);
}

.jp-ImageIcon {
  background-image: var(--jp-icon-image);
}

.jp-InfoIcon {
  background-image: var(--jp-icon-info);
}

.jp-InspectorIcon {
  background-image: var(--jp-icon-inspector);
}

.jp-JsonIcon {
  background-image: var(--jp-icon-json);
}

.jp-JuliaIcon {
  background-image: var(--jp-icon-julia);
}

.jp-JupyterFaviconIcon {
  background-image: var(--jp-icon-jupyter-favicon);
}

.jp-JupyterIcon {
  background-image: var(--jp-icon-jupyter);
}

.jp-JupyterlabWordmarkIcon {
  background-image: var(--jp-icon-jupyterlab-wordmark);
}

.jp-KernelIcon {
  background-image: var(--jp-icon-kernel);
}

.jp-KeyboardIcon {
  background-image: var(--jp-icon-keyboard);
}

.jp-LaunchIcon {
  background-image: var(--jp-icon-launch);
}

.jp-LauncherIcon {
  background-image: var(--jp-icon-launcher);
}

.jp-LineFormIcon {
  background-image: var(--jp-icon-line-form);
}

.jp-LinkIcon {
  background-image: var(--jp-icon-link);
}

.jp-ListIcon {
  background-image: var(--jp-icon-list);
}

.jp-MarkdownIcon {
  background-image: var(--jp-icon-markdown);
}

.jp-MoveDownIcon {
  background-image: var(--jp-icon-move-down);
}

.jp-MoveUpIcon {
  background-image: var(--jp-icon-move-up);
}

.jp-NewFolderIcon {
  background-image: var(--jp-icon-new-folder);
}

.jp-NotTrustedIcon {
  background-image: var(--jp-icon-not-trusted);
}

.jp-NotebookIcon {
  background-image: var(--jp-icon-notebook);
}

.jp-NumberingIcon {
  background-image: var(--jp-icon-numbering);
}

.jp-OfflineBoltIcon {
  background-image: var(--jp-icon-offline-bolt);
}

.jp-PaletteIcon {
  background-image: var(--jp-icon-palette);
}

.jp-PasteIcon {
  background-image: var(--jp-icon-paste);
}

.jp-PdfIcon {
  background-image: var(--jp-icon-pdf);
}

.jp-PythonIcon {
  background-image: var(--jp-icon-python);
}

.jp-RKernelIcon {
  background-image: var(--jp-icon-r-kernel);
}

.jp-ReactIcon {
  background-image: var(--jp-icon-react);
}

.jp-RedoIcon {
  background-image: var(--jp-icon-redo);
}

.jp-RefreshIcon {
  background-image: var(--jp-icon-refresh);
}

.jp-RegexIcon {
  background-image: var(--jp-icon-regex);
}

.jp-RunIcon {
  background-image: var(--jp-icon-run);
}

.jp-RunningIcon {
  background-image: var(--jp-icon-running);
}

.jp-SaveIcon {
  background-image: var(--jp-icon-save);
}

.jp-SearchIcon {
  background-image: var(--jp-icon-search);
}

.jp-SettingsIcon {
  background-image: var(--jp-icon-settings);
}

.jp-ShareIcon {
  background-image: var(--jp-icon-share);
}

.jp-SpreadsheetIcon {
  background-image: var(--jp-icon-spreadsheet);
}

.jp-StopIcon {
  background-image: var(--jp-icon-stop);
}

.jp-TabIcon {
  background-image: var(--jp-icon-tab);
}

.jp-TableRowsIcon {
  background-image: var(--jp-icon-table-rows);
}

.jp-TagIcon {
  background-image: var(--jp-icon-tag);
}

.jp-TerminalIcon {
  background-image: var(--jp-icon-terminal);
}

.jp-TextEditorIcon {
  background-image: var(--jp-icon-text-editor);
}

.jp-TocIcon {
  background-image: var(--jp-icon-toc);
}

.jp-TreeViewIcon {
  background-image: var(--jp-icon-tree-view);
}

.jp-TrustedIcon {
  background-image: var(--jp-icon-trusted);
}

.jp-UndoIcon {
  background-image: var(--jp-icon-undo);
}

.jp-UserIcon {
  background-image: var(--jp-icon-user);
}

.jp-UsersIcon {
  background-image: var(--jp-icon-users);
}

.jp-VegaIcon {
  background-image: var(--jp-icon-vega);
}

.jp-WordIcon {
  background-image: var(--jp-icon-word);
}

.jp-YamlIcon {
  background-image: var(--jp-icon-yaml);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * (DEPRECATED) Support for consuming icons as CSS background images
 */

.jp-Icon,
.jp-MaterialIcon {
  background-position: center;
  background-repeat: no-repeat;
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-cover {
  background-position: center;
  background-repeat: no-repeat;
  background-size: cover;
}

/**
 * (DEPRECATED) Support for specific CSS icon sizes
 */

.jp-Icon-16 {
  background-size: 16px;
  min-width: 16px;
  min-height: 16px;
}

.jp-Icon-18 {
  background-size: 18px;
  min-width: 18px;
  min-height: 18px;
}

.jp-Icon-20 {
  background-size: 20px;
  min-width: 20px;
  min-height: 20px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.lm-TabBar .lm-TabBar-addButton {
  align-items: center;
  display: flex;
  padding: 4px;
  padding-bottom: 5px;
  margin-right: 1px;
  background-color: var(--jp-layout-color2);
}

.lm-TabBar .lm-TabBar-addButton:hover {
  background-color: var(--jp-layout-color1);
}

.lm-DockPanel-tabBar .lm-TabBar-tab {
  width: var(--jp-private-horizontal-tab-width);
}

.lm-DockPanel-tabBar .lm-TabBar-content {
  flex: unset;
}

.lm-DockPanel-tabBar[data-orientation='horizontal'] {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for icons as inline SVG HTMLElements
 */

/* recolor the primary elements of an icon */
.jp-icon0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-accent0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-accent1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-accent2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-accent3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-accent4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-accent0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-accent1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-accent2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-accent3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-accent4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-none[fill] {
  fill: none;
}

.jp-icon-none[stroke] {
  stroke: none;
}

/* brand icon colors. Same for light and dark */
.jp-icon-brand0[fill] {
  fill: var(--jp-brand-color0);
}

.jp-icon-brand1[fill] {
  fill: var(--jp-brand-color1);
}

.jp-icon-brand2[fill] {
  fill: var(--jp-brand-color2);
}

.jp-icon-brand3[fill] {
  fill: var(--jp-brand-color3);
}

.jp-icon-brand4[fill] {
  fill: var(--jp-brand-color4);
}

.jp-icon-brand0[stroke] {
  stroke: var(--jp-brand-color0);
}

.jp-icon-brand1[stroke] {
  stroke: var(--jp-brand-color1);
}

.jp-icon-brand2[stroke] {
  stroke: var(--jp-brand-color2);
}

.jp-icon-brand3[stroke] {
  stroke: var(--jp-brand-color3);
}

.jp-icon-brand4[stroke] {
  stroke: var(--jp-brand-color4);
}

/* warn icon colors. Same for light and dark */
.jp-icon-warn0[fill] {
  fill: var(--jp-warn-color0);
}

.jp-icon-warn1[fill] {
  fill: var(--jp-warn-color1);
}

.jp-icon-warn2[fill] {
  fill: var(--jp-warn-color2);
}

.jp-icon-warn3[fill] {
  fill: var(--jp-warn-color3);
}

.jp-icon-warn0[stroke] {
  stroke: var(--jp-warn-color0);
}

.jp-icon-warn1[stroke] {
  stroke: var(--jp-warn-color1);
}

.jp-icon-warn2[stroke] {
  stroke: var(--jp-warn-color2);
}

.jp-icon-warn3[stroke] {
  stroke: var(--jp-warn-color3);
}

/* icon colors that contrast well with each other and most backgrounds */
.jp-icon-contrast0[fill] {
  fill: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[fill] {
  fill: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[fill] {
  fill: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[fill] {
  fill: var(--jp-icon-contrast-color3);
}

.jp-icon-contrast0[stroke] {
  stroke: var(--jp-icon-contrast-color0);
}

.jp-icon-contrast1[stroke] {
  stroke: var(--jp-icon-contrast-color1);
}

.jp-icon-contrast2[stroke] {
  stroke: var(--jp-icon-contrast-color2);
}

.jp-icon-contrast3[stroke] {
  stroke: var(--jp-icon-contrast-color3);
}

.jp-icon-dot[fill] {
  fill: var(--jp-warn-color0);
}

.jp-jupyter-icon-color[fill] {
  fill: var(--jp-jupyter-icon-color, var(--jp-warn-color0));
}

.jp-notebook-icon-color[fill] {
  fill: var(--jp-notebook-icon-color, var(--jp-warn-color0));
}

.jp-json-icon-color[fill] {
  fill: var(--jp-json-icon-color, var(--jp-warn-color1));
}

.jp-console-icon-color[fill] {
  fill: var(--jp-console-icon-color, white);
}

.jp-console-icon-background-color[fill] {
  fill: var(--jp-console-icon-background-color, var(--jp-brand-color1));
}

.jp-terminal-icon-color[fill] {
  fill: var(--jp-terminal-icon-color, var(--jp-layout-color2));
}

.jp-terminal-icon-background-color[fill] {
  fill: var(
    --jp-terminal-icon-background-color,
    var(--jp-inverse-layout-color2)
  );
}

.jp-text-editor-icon-color[fill] {
  fill: var(--jp-text-editor-icon-color, var(--jp-inverse-layout-color3));
}

.jp-inspector-icon-color[fill] {
  fill: var(--jp-inspector-icon-color, var(--jp-inverse-layout-color3));
}

/* CSS for icons in selected filebrowser listing items */
.jp-DirListing-item.jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

.jp-DirListing-item.jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* stylelint-disable selector-max-class, selector-max-compound-selectors */

/**
* TODO: come up with non css-hack solution for showing the busy icon on top
*  of the close icon
* CSS for complex behavior of close icon of tabs in the main area tabbar
*/
.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon3[fill] {
  fill: none;
}

.lm-DockPanel-tabBar
  .lm-TabBar-tab.lm-mod-closable.jp-mod-dirty
  > .lm-TabBar-tabCloseIcon
  > :not(:hover)
  > .jp-icon-busy[fill] {
  fill: var(--jp-inverse-layout-color3);
}

/* stylelint-enable selector-max-class, selector-max-compound-selectors */

/* CSS for icons in status bar */
#jp-main-statusbar .jp-mod-selected .jp-icon-selectable[fill] {
  fill: #fff;
}

#jp-main-statusbar .jp-mod-selected .jp-icon-selectable-inverse[fill] {
  fill: var(--jp-brand-color1);
}

/* special handling for splash icon CSS. While the theme CSS reloads during
   splash, the splash icon can loose theming. To prevent that, we set a
   default for its color variable */
:root {
  --jp-warn-color0: var(--md-orange-700);
}

/* not sure what to do with this one, used in filebrowser listing */
.jp-DragIcon {
  margin-right: 4px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/**
 * Support for alt colors for icons as inline SVG HTMLElements
 */

/* alt recolor the primary elements of an icon */
.jp-icon-alt .jp-icon0[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-alt .jp-icon0[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-alt .jp-icon1[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-alt .jp-icon2[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-alt .jp-icon3[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-alt .jp-icon4[stroke] {
  stroke: var(--jp-layout-color4);
}

/* alt recolor the accent elements of an icon */
.jp-icon-alt .jp-icon-accent0[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-alt .jp-icon-accent0[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-alt .jp-icon-accent1[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-alt .jp-icon-accent2[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-alt .jp-icon-accent3[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-alt .jp-icon-accent4[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-icon-hoverShow:not(:hover) .jp-icon-hoverShow-content {
  display: none !important;
}

/**
 * Support for hover colors for icons as inline SVG HTMLElements
 */

/**
 * regular colors
 */

/* recolor the primary elements of an icon */
.jp-icon-hover :hover .jp-icon0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/* recolor the accent elements of an icon */
.jp-icon-hover :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* set the color of an icon to transparent */
.jp-icon-hover :hover .jp-icon-none-hover[fill] {
  fill: none;
}

.jp-icon-hover :hover .jp-icon-none-hover[stroke] {
  stroke: none;
}

/**
 * inverse colors
 */

/* inverse recolor the primary elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[fill] {
  fill: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[fill] {
  fill: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[fill] {
  fill: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[fill] {
  fill: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[fill] {
  fill: var(--jp-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon0-hover[stroke] {
  stroke: var(--jp-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon1-hover[stroke] {
  stroke: var(--jp-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon2-hover[stroke] {
  stroke: var(--jp-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon3-hover[stroke] {
  stroke: var(--jp-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon4-hover[stroke] {
  stroke: var(--jp-layout-color4);
}

/* inverse recolor the accent elements of an icon */
.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[fill] {
  fill: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[fill] {
  fill: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[fill] {
  fill: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[fill] {
  fill: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[fill] {
  fill: var(--jp-inverse-layout-color4);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent0-hover[stroke] {
  stroke: var(--jp-inverse-layout-color0);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent1-hover[stroke] {
  stroke: var(--jp-inverse-layout-color1);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent2-hover[stroke] {
  stroke: var(--jp-inverse-layout-color2);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent3-hover[stroke] {
  stroke: var(--jp-inverse-layout-color3);
}

.jp-icon-hover.jp-icon-alt :hover .jp-icon-accent4-hover[stroke] {
  stroke: var(--jp-inverse-layout-color4);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-IFrame {
  width: 100%;
  height: 100%;
}

.jp-IFrame > iframe {
  border: none;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-IFrame {
  position: relative;
}

body.lm-mod-override-cursor .jp-IFrame::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-HoverBox {
  position: fixed;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FormGroup-content fieldset {
  border: none;
  padding: 0;
  min-width: 0;
  width: 100%;
}

/* stylelint-disable selector-max-type */

.jp-FormGroup-content fieldset .jp-inputFieldWrapper input,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper select,
.jp-FormGroup-content fieldset .jp-inputFieldWrapper textarea {
  font-size: var(--jp-content-font-size2);
  border-color: var(--jp-input-border-color);
  border-style: solid;
  border-radius: var(--jp-border-radius);
  border-width: 1px;
  padding: 6px 8px;
  background: none;
  color: var(--jp-ui-font-color0);
  height: inherit;
}

.jp-FormGroup-content fieldset input[type='checkbox'] {
  position: relative;
  top: 2px;
  margin-left: 0;
}

.jp-FormGroup-content button.jp-mod-styled {
  cursor: pointer;
}

.jp-FormGroup-content .checkbox label {
  cursor: pointer;
  font-size: var(--jp-content-font-size1);
}

.jp-FormGroup-content .jp-root > fieldset > legend {
  display: none;
}

.jp-FormGroup-content .jp-root > fieldset > p {
  display: none;
}

/** copy of `input.jp-mod-styled:focus` style */
.jp-FormGroup-content fieldset input:focus,
.jp-FormGroup-content fieldset select:focus {
  -moz-outline-radius: unset;
  outline: var(--jp-border-width) solid var(--md-blue-500);
  outline-offset: -1px;
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-FormGroup-content fieldset input:hover:not(:focus),
.jp-FormGroup-content fieldset select:hover:not(:focus) {
  background-color: var(--jp-border-color2);
}

/* stylelint-enable selector-max-type */

.jp-FormGroup-content .checkbox .field-description {
  /* Disable default description field for checkbox:
   because other widgets do not have description fields,
   we add descriptions to each widget on the field level.
  */
  display: none;
}

.jp-FormGroup-content #root__description {
  display: none;
}

.jp-FormGroup-content .jp-modifiedIndicator {
  width: 5px;
  background-color: var(--jp-brand-color2);
  margin-top: 0;
  margin-left: calc(var(--jp-private-settingeditor-modifier-indent) * -1);
  flex-shrink: 0;
}

.jp-FormGroup-content .jp-modifiedIndicator.jp-errorIndicator {
  background-color: var(--jp-error-color0);
  margin-right: 0.5em;
}

/* RJSF ARRAY style */

.jp-arrayFieldWrapper legend {
  font-size: var(--jp-content-font-size2);
  color: var(--jp-ui-font-color0);
  flex-basis: 100%;
  padding: 4px 0;
  font-weight: var(--jp-content-heading-font-weight);
  border-bottom: 1px solid var(--jp-border-color2);
}

.jp-arrayFieldWrapper .field-description {
  padding: 4px 0;
  white-space: pre-wrap;
}

.jp-arrayFieldWrapper .array-item {
  width: 100%;
  border: 1px solid var(--jp-border-color2);
  border-radius: 4px;
  margin: 4px;
}

.jp-ArrayOperations {
  display: flex;
  margin-left: 8px;
}

.jp-ArrayOperationsButton {
  margin: 2px;
}

.jp-ArrayOperationsButton .jp-icon3[fill] {
  fill: var(--jp-ui-font-color0);
}

button.jp-ArrayOperationsButton.jp-mod-styled:disabled {
  cursor: not-allowed;
  opacity: 0.5;
}

/* RJSF form validation error */

.jp-FormGroup-content .validationErrors {
  color: var(--jp-error-color0);
}

/* Hide panel level error as duplicated the field level error */
.jp-FormGroup-content .panel.errors {
  display: none;
}

/* RJSF normal content (settings-editor) */

.jp-FormGroup-contentNormal {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-FormGroup-contentItem {
  margin-left: 7px;
  color: var(--jp-ui-font-color0);
}

.jp-FormGroup-contentNormal .jp-FormGroup-description {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-default {
  flex-basis: 100%;
  padding: 4px 7px;
}

.jp-FormGroup-contentNormal .jp-FormGroup-fieldLabel {
  font-size: var(--jp-content-font-size1);
  font-weight: normal;
  min-width: 120px;
}

.jp-FormGroup-contentNormal fieldset:not(:first-child) {
  margin-left: 7px;
}

.jp-FormGroup-contentNormal .field-array-of-string .array-item {
  /* Display `jp-ArrayOperations` buttons side-by-side with content except
    for small screens where flex-wrap will place them one below the other.
  */
  display: flex;
  align-items: center;
  flex-wrap: wrap;
}

.jp-FormGroup-contentNormal .jp-objectFieldWrapper .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

/* RJSF compact content (metadata-form) */

.jp-FormGroup-content.jp-FormGroup-contentCompact {
  width: 100%;
}

.jp-FormGroup-contentCompact .form-group {
  display: flex;
  padding: 0.5em 0.2em 0.5em 0;
}

.jp-FormGroup-contentCompact
  .jp-FormGroup-compactTitle
  .jp-FormGroup-description {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color2);
}

.jp-FormGroup-contentCompact .jp-FormGroup-fieldLabel {
  padding-bottom: 0.3em;
}

.jp-FormGroup-contentCompact .jp-inputFieldWrapper .form-control {
  width: 100%;
  box-sizing: border-box;
}

.jp-FormGroup-contentCompact .jp-arrayFieldWrapper .jp-FormGroup-compactTitle {
  padding-bottom: 7px;
}

.jp-FormGroup-contentCompact
  .jp-objectFieldWrapper
  .jp-objectFieldWrapper
  .form-group {
  padding: 2px 8px 2px var(--jp-private-settingeditor-modifier-indent);
  margin-top: 2px;
}

.jp-FormGroup-contentCompact ul.error-detail {
  margin-block-start: 0.5em;
  margin-block-end: 0.5em;
  padding-inline-start: 1em;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-SidePanel {
  display: flex;
  flex-direction: column;
  min-width: var(--jp-sidebar-min-width);
  overflow-y: auto;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);
  font-size: var(--jp-ui-font-size1);
}

.jp-SidePanel-header {
  flex: 0 0 auto;
  display: flex;
  border-bottom: var(--jp-border-width) solid var(--jp-border-color2);
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin: 0;
  padding: 2px;
  text-transform: uppercase;
}

.jp-SidePanel-toolbar {
  flex: 0 0 auto;
}

.jp-SidePanel-content {
  flex: 1 1 auto;
}

.jp-SidePanel-toolbar,
.jp-AccordionPanel-toolbar {
  height: var(--jp-private-toolbar-height);
}

.jp-SidePanel-toolbar.jp-Toolbar-micro {
  display: none;
}

.lm-AccordionPanel .jp-AccordionPanel-title {
  box-sizing: border-box;
  line-height: 25px;
  margin: 0;
  display: flex;
  align-items: center;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  font-size: var(--jp-ui-font-size0);
}

.jp-AccordionPanel-title {
  cursor: pointer;
  user-select: none;
  -moz-user-select: none;
  -webkit-user-select: none;
  text-transform: uppercase;
}

.lm-AccordionPanel[data-orientation='horizontal'] > .jp-AccordionPanel-title {
  /* Title is rotated for horizontal accordion panel using CSS */
  display: block;
  transform-origin: top left;
  transform: rotate(-90deg) translate(-100%);
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleLabel {
  user-select: none;
  text-overflow: ellipsis;
  white-space: nowrap;
  overflow: hidden;
}

.jp-AccordionPanel-title .lm-AccordionPanel-titleCollapser {
  transform: rotate(-90deg);
  margin: auto 0;
  height: 16px;
}

.jp-AccordionPanel-title.lm-mod-expanded .lm-AccordionPanel-titleCollapser {
  transform: rotate(0deg);
}

.lm-AccordionPanel .jp-AccordionPanel-toolbar {
  background: none;
  box-shadow: none;
  border: none;
  margin-left: auto;
}

.lm-AccordionPanel .lm-SplitPanel-handle:hover {
  background: var(--jp-layout-color3);
}

.jp-text-truncated {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Spinner {
  position: absolute;
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 10;
  left: 0;
  top: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-layout-color0);
  outline: none;
}

.jp-SpinnerContent {
  font-size: 10px;
  margin: 50px auto;
  text-indent: -9999em;
  width: 3em;
  height: 3em;
  border-radius: 50%;
  background: var(--jp-brand-color3);
  background: linear-gradient(
    to right,
    #f37626 10%,
    rgba(255, 255, 255, 0) 42%
  );
  position: relative;
  animation: load3 1s infinite linear, fadeIn 1s;
}

.jp-SpinnerContent::before {
  width: 50%;
  height: 50%;
  background: #f37626;
  border-radius: 100% 0 0;
  position: absolute;
  top: 0;
  left: 0;
  content: '';
}

.jp-SpinnerContent::after {
  background: var(--jp-layout-color0);
  width: 75%;
  height: 75%;
  border-radius: 50%;
  content: '';
  margin: auto;
  position: absolute;
  top: 0;
  left: 0;
  bottom: 0;
  right: 0;
}

@keyframes fadeIn {
  0% {
    opacity: 0;
  }

  100% {
    opacity: 1;
  }
}

@keyframes load3 {
  0% {
    transform: rotate(0deg);
  }

  100% {
    transform: rotate(360deg);
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

button.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: none;
  box-sizing: border-box;
  text-align: center;
  line-height: 32px;
  height: 32px;
  padding: 0 12px;
  letter-spacing: 0.8px;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input.jp-mod-styled {
  background: var(--jp-input-background);
  height: 28px;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color1);
  padding-left: 7px;
  padding-right: 7px;
  font-size: var(--jp-ui-font-size2);
  color: var(--jp-ui-font-color0);
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

input[type='checkbox'].jp-mod-styled {
  appearance: checkbox;
  -webkit-appearance: checkbox;
  -moz-appearance: checkbox;
  height: auto;
}

input.jp-mod-styled:focus {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-select-wrapper {
  display: flex;
  position: relative;
  flex-direction: column;
  padding: 1px;
  background-color: var(--jp-layout-color1);
  box-sizing: border-box;
  margin-bottom: 12px;
}

.jp-select-wrapper:not(.multiple) {
  height: 28px;
}

.jp-select-wrapper.jp-mod-focused select.jp-mod-styled {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-input-active-background);
}

select.jp-mod-styled:hover {
  cursor: pointer;
  color: var(--jp-ui-font-color0);
  background-color: var(--jp-input-hover-background);
  box-shadow: inset 0 0 1px rgba(0, 0, 0, 0.5);
}

select.jp-mod-styled {
  flex: 1 1 auto;
  width: 100%;
  font-size: var(--jp-ui-font-size2);
  background: var(--jp-input-background);
  color: var(--jp-ui-font-color0);
  padding: 0 25px 0 8px;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
}

select.jp-mod-styled:not([multiple]) {
  height: 32px;
}

select.jp-mod-styled[multiple] {
  max-height: 200px;
  overflow-y: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-switch {
  display: flex;
  align-items: center;
  padding-left: 4px;
  padding-right: 4px;
  font-size: var(--jp-ui-font-size1);
  background-color: transparent;
  color: var(--jp-ui-font-color1);
  border: none;
  height: 20px;
}

.jp-switch:hover {
  background-color: var(--jp-layout-color2);
}

.jp-switch-label {
  margin-right: 5px;
  font-family: var(--jp-ui-font-family);
}

.jp-switch-track {
  cursor: pointer;
  background-color: var(--jp-switch-color, var(--jp-border-color1));
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 34px;
  height: 16px;
  width: 35px;
  position: relative;
}

.jp-switch-track::before {
  content: '';
  position: absolute;
  height: 10px;
  width: 10px;
  margin: 3px;
  left: 0;
  background-color: var(--jp-ui-inverse-font-color1);
  -webkit-transition: 0.4s;
  transition: 0.4s;
  border-radius: 50%;
}

.jp-switch[aria-checked='true'] .jp-switch-track {
  background-color: var(--jp-switch-true-position-color, var(--jp-warn-color0));
}

.jp-switch[aria-checked='true'] .jp-switch-track::before {
  /* track width (35) - margins (3 + 3) - thumb width (10) */
  left: 19px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toolbar-height: calc(
    28px + var(--jp-border-width)
  ); /* leave 28px for content */
}

.jp-Toolbar {
  color: var(--jp-ui-font-color1);
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: 2px;
  z-index: 8;
  overflow-x: hidden;
}

/* Toolbar items */

.jp-Toolbar > .jp-Toolbar-item.jp-Toolbar-spacer {
  flex-grow: 1;
  flex-shrink: 1;
}

.jp-Toolbar-item.jp-Toolbar-kernelStatus {
  display: inline-block;
  width: 32px;
  background-repeat: no-repeat;
  background-position: center;
  background-size: 16px;
}

.jp-Toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  display: flex;
  padding-left: 1px;
  padding-right: 1px;
  font-size: var(--jp-ui-font-size1);
  line-height: var(--jp-private-toolbar-height);
  height: 100%;
}

/* Toolbar buttons */

/* This is the div we use to wrap the react component into a Widget */
div.jp-ToolbarButton {
  color: transparent;
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0;
  margin: 0;
}

button.jp-ToolbarButtonComponent {
  background: var(--jp-layout-color1);
  border: none;
  box-sizing: border-box;
  outline: none;
  appearance: none;
  -webkit-appearance: none;
  -moz-appearance: none;
  padding: 0 6px;
  margin: 0;
  height: 24px;
  border-radius: var(--jp-border-radius);
  display: flex;
  align-items: center;
  text-align: center;
  font-size: 14px;
  min-width: unset;
  min-height: unset;
}

button.jp-ToolbarButtonComponent:disabled {
  opacity: 0.4;
}

button.jp-ToolbarButtonComponent > span {
  padding: 0;
  flex: 0 0 auto;
}

button.jp-ToolbarButtonComponent .jp-ToolbarButtonComponent-label {
  font-size: var(--jp-ui-font-size1);
  line-height: 100%;
  padding-left: 2px;
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar.jp-Toolbar-micro {
  padding: 0;
  min-height: 0;
}

#jp-main-dock-panel[data-mode='single-document']
  .jp-MainAreaWidget
  > .jp-Toolbar {
  border: none;
  box-shadow: none;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-WindowedPanel-outer {
  position: relative;
  overflow-y: auto;
}

.jp-WindowedPanel-inner {
  position: relative;
}

.jp-WindowedPanel-window {
  position: absolute;
  left: 0;
  right: 0;
  overflow: visible;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/* Sibling imports */

body {
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
}

/* Disable native link decoration styles everywhere outside of dialog boxes */
a {
  text-decoration: unset;
  color: unset;
}

a:hover {
  text-decoration: unset;
  color: unset;
}

/* Accessibility for links inside dialog box text */
.jp-Dialog-content a {
  text-decoration: revert;
  color: var(--jp-content-link-color);
}

.jp-Dialog-content a:hover {
  text-decoration: revert;
}

/* Styles for ui-components */
.jp-Button {
  color: var(--jp-ui-font-color2);
  border-radius: var(--jp-border-radius);
  padding: 0 12px;
  font-size: var(--jp-ui-font-size1);

  /* Copy from blueprint 3 */
  display: inline-flex;
  flex-direction: row;
  border: none;
  cursor: pointer;
  align-items: center;
  justify-content: center;
  text-align: left;
  vertical-align: middle;
  min-height: 30px;
  min-width: 30px;
}

.jp-Button:disabled {
  cursor: not-allowed;
}

.jp-Button:empty {
  padding: 0 !important;
}

.jp-Button.jp-mod-small {
  min-height: 24px;
  min-width: 24px;
  font-size: 12px;
  padding: 0 7px;
}

/* Use our own theme for hover styles */
.jp-Button.jp-mod-minimal:hover {
  background-color: var(--jp-layout-color2);
}

.jp-Button.jp-mod-minimal {
  background: none;
}

.jp-InputGroup {
  display: block;
  position: relative;
}

.jp-InputGroup input {
  box-sizing: border-box;
  border: none;
  border-radius: 0;
  background-color: transparent;
  color: var(--jp-ui-font-color0);
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
  padding-bottom: 0;
  padding-top: 0;
  padding-left: 10px;
  padding-right: 28px;
  position: relative;
  width: 100%;
  -webkit-appearance: none;
  -moz-appearance: none;
  appearance: none;
  font-size: 14px;
  font-weight: 400;
  height: 30px;
  line-height: 30px;
  outline: none;
  vertical-align: middle;
}

.jp-InputGroup input:focus {
  box-shadow: inset 0 0 0 var(--jp-border-width)
      var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-InputGroup input:disabled {
  cursor: not-allowed;
  resize: block;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input:disabled ~ span {
  cursor: not-allowed;
  color: var(--jp-ui-font-color2);
}

.jp-InputGroup input::placeholder,
input::placeholder {
  color: var(--jp-ui-font-color2);
}

.jp-InputGroupAction {
  position: absolute;
  bottom: 1px;
  right: 0;
  padding: 6px;
}

.jp-HTMLSelect.jp-DefaultStyle select {
  background-color: initial;
  border: none;
  border-radius: 0;
  box-shadow: none;
  color: var(--jp-ui-font-color0);
  display: block;
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  height: 24px;
  line-height: 14px;
  padding: 0 25px 0 10px;
  text-align: left;
  -moz-appearance: none;
  -webkit-appearance: none;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color2);
  cursor: not-allowed;
  resize: block;
}

.jp-HTMLSelect.jp-DefaultStyle select:disabled ~ span {
  cursor: not-allowed;
}

/* Use our own theme for hover and option styles */
/* stylelint-disable-next-line selector-max-type */
.jp-HTMLSelect.jp-DefaultStyle select:hover,
.jp-HTMLSelect.jp-DefaultStyle select > option {
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color0);
}

select {
  box-sizing: border-box;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-StatusBar-Widget {
  display: flex;
  align-items: center;
  background: var(--jp-layout-color2);
  min-height: var(--jp-statusbar-height);
  justify-content: space-between;
  padding: 0 10px;
}

.jp-StatusBar-Left {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-StatusBar-Middle {
  display: flex;
  align-items: center;
}

.jp-StatusBar-Right {
  display: flex;
  align-items: center;
  flex-direction: row-reverse;
}

.jp-StatusBar-Item {
  max-height: var(--jp-statusbar-height);
  margin: 0 2px;
  height: var(--jp-statusbar-height);
  white-space: nowrap;
  text-overflow: ellipsis;
  color: var(--jp-ui-font-color1);
  padding: 0 6px;
}

.jp-mod-highlighted:hover {
  background-color: var(--jp-layout-color3);
}

.jp-mod-clicked {
  background-color: var(--jp-brand-color1);
}

.jp-mod-clicked:hover {
  background-color: var(--jp-brand-color0);
}

.jp-mod-clicked .jp-StatusBar-TextItem {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-StatusBar-HoverItem {
  box-shadow: '0px 4px 4px rgba(0, 0, 0, 0.25)';
}

.jp-StatusBar-TextItem {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  line-height: 24px;
  color: var(--jp-ui-font-color1);
}

.jp-StatusBar-GroupItem {
  display: flex;
  align-items: center;
  flex-direction: row;
}

.jp-Statusbar-ProgressCircle svg {
  display: block;
  margin: 0 auto;
  width: 16px;
  height: 24px;
  align-self: normal;
}

.jp-Statusbar-ProgressCircle path {
  fill: var(--jp-inverse-layout-color3);
}

.jp-Statusbar-ProgressBar-progress-bar {
  height: 10px;
  width: 100px;
  border: solid 0.25px var(--jp-brand-color2);
  border-radius: 3px;
  overflow: hidden;
  align-self: center;
}

.jp-Statusbar-ProgressBar-progress-bar > div {
  background-color: var(--jp-brand-color2);
  background-image: linear-gradient(
    -45deg,
    rgba(255, 255, 255, 0.2) 25%,
    transparent 25%,
    transparent 50%,
    rgba(255, 255, 255, 0.2) 50%,
    rgba(255, 255, 255, 0.2) 75%,
    transparent 75%,
    transparent
  );
  background-size: 40px 40px;
  float: left;
  width: 0%;
  height: 100%;
  font-size: 12px;
  line-height: 14px;
  color: #fff;
  text-align: center;
  animation: jp-Statusbar-ExecutionTime-progress-bar 2s linear infinite;
}

.jp-Statusbar-ProgressBar-progress-bar p {
  color: var(--jp-ui-font-color1);
  font-family: var(--jp-ui-font-family);
  font-size: var(--jp-ui-font-size1);
  line-height: 10px;
  width: 100px;
}

@keyframes jp-Statusbar-ExecutionTime-progress-bar {
  0% {
    background-position: 0 0;
  }

  100% {
    background-position: 40px 40px;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-commandpalette-search-height: 28px;
}

/*-----------------------------------------------------------------------------
| Overall styles
|----------------------------------------------------------------------------*/

.lm-CommandPalette {
  padding-bottom: 0;
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Modal variant
|----------------------------------------------------------------------------*/

.jp-ModalCommandPalette {
  position: absolute;
  z-index: 10000;
  top: 38px;
  left: 30%;
  margin: 0;
  padding: 4px;
  width: 40%;
  box-shadow: var(--jp-elevation-z4);
  border-radius: 4px;
  background: var(--jp-layout-color0);
}

.jp-ModalCommandPalette .lm-CommandPalette {
  max-height: 40vh;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-close-icon::after {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-header {
  display: none;
}

.jp-ModalCommandPalette .lm-CommandPalette .lm-CommandPalette-item {
  margin-left: 4px;
  margin-right: 4px;
}

.jp-ModalCommandPalette
  .lm-CommandPalette
  .lm-CommandPalette-item.lm-mod-disabled {
  display: none;
}

/*-----------------------------------------------------------------------------
| Search
|----------------------------------------------------------------------------*/

.lm-CommandPalette-search {
  padding: 4px;
  background-color: var(--jp-layout-color1);
  z-index: 2;
}

.lm-CommandPalette-wrapper {
  overflow: overlay;
  padding: 0 9px;
  background-color: var(--jp-input-active-background);
  height: 30px;
  box-shadow: inset 0 0 0 var(--jp-border-width) var(--jp-input-border-color);
}

.lm-CommandPalette.lm-mod-focused .lm-CommandPalette-wrapper {
  box-shadow: inset 0 0 0 1px var(--jp-input-active-box-shadow-color),
    inset 0 0 0 3px var(--jp-input-active-box-shadow-color);
}

.jp-SearchIconGroup {
  color: white;
  background-color: var(--jp-brand-color1);
  position: absolute;
  top: 4px;
  right: 4px;
  padding: 5px 5px 1px;
}

.jp-SearchIconGroup svg {
  height: 20px;
  width: 20px;
}

.jp-SearchIconGroup .jp-icon3[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-input {
  background: transparent;
  width: calc(100% - 18px);
  float: left;
  border: none;
  outline: none;
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  line-height: var(--jp-private-commandpalette-search-height);
}

.lm-CommandPalette-input::-webkit-input-placeholder,
.lm-CommandPalette-input::-moz-placeholder,
.lm-CommandPalette-input:-ms-input-placeholder {
  color: var(--jp-ui-font-color2);
  font-size: var(--jp-ui-font-size1);
}

/*-----------------------------------------------------------------------------
| Results
|----------------------------------------------------------------------------*/

.lm-CommandPalette-header:first-child {
  margin-top: 0;
}

.lm-CommandPalette-header {
  border-bottom: solid var(--jp-border-width) var(--jp-border-color2);
  color: var(--jp-ui-font-color1);
  cursor: pointer;
  display: flex;
  font-size: var(--jp-ui-font-size0);
  font-weight: 600;
  letter-spacing: 1px;
  margin-top: 8px;
  padding: 8px 0 8px 12px;
  text-transform: uppercase;
}

.lm-CommandPalette-header.lm-mod-active {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-header > mark {
  background-color: transparent;
  font-weight: bold;
  color: var(--jp-ui-font-color1);
}

.lm-CommandPalette-item {
  padding: 4px 12px 4px 4px;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  font-weight: 400;
  display: flex;
}

.lm-CommandPalette-item.lm-mod-disabled {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item.lm-mod-active {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item.lm-mod-active .lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-inverse-font-color0);
}

.lm-CommandPalette-item.lm-mod-active .jp-icon-selectable[fill] {
  fill: var(--jp-layout-color0);
}

.lm-CommandPalette-item.lm-mod-active:hover:not(.lm-mod-disabled) {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.lm-CommandPalette-item:hover:not(.lm-mod-active):not(.lm-mod-disabled) {
  background: var(--jp-layout-color2);
}

.lm-CommandPalette-itemContent {
  overflow: hidden;
}

.lm-CommandPalette-itemLabel > mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.lm-CommandPalette-item.lm-mod-disabled mark {
  color: var(--jp-ui-font-color2);
}

.lm-CommandPalette-item .lm-CommandPalette-itemIcon {
  margin: 0 4px 0 0;
  position: relative;
  width: 16px;
  top: 2px;
  flex: 0 0 auto;
}

.lm-CommandPalette-item.lm-mod-disabled .lm-CommandPalette-itemIcon {
  opacity: 0.6;
}

.lm-CommandPalette-item .lm-CommandPalette-itemShortcut {
  flex: 0 0 auto;
}

.lm-CommandPalette-itemCaption {
  display: none;
}

.lm-CommandPalette-content {
  background-color: var(--jp-layout-color1);
}

.lm-CommandPalette-content:empty::after {
  content: 'No results';
  margin: auto;
  margin-top: 20px;
  width: 100px;
  display: block;
  font-size: var(--jp-ui-font-size2);
  font-family: var(--jp-ui-font-family);
  font-weight: lighter;
}

.lm-CommandPalette-emptyMessage {
  text-align: center;
  margin-top: 24px;
  line-height: 1.32;
  padding: 0 8px;
  color: var(--jp-content-font-color3);
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Dialog {
  position: absolute;
  z-index: 10000;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  top: 0;
  left: 0;
  margin: 0;
  padding: 0;
  width: 100%;
  height: 100%;
  background: var(--jp-dialog-background);
}

.jp-Dialog-content {
  display: flex;
  flex-direction: column;
  margin-left: auto;
  margin-right: auto;
  background: var(--jp-layout-color1);
  padding: 24px 24px 12px;
  min-width: 300px;
  min-height: 150px;
  max-width: 1000px;
  max-height: 500px;
  box-sizing: border-box;
  box-shadow: var(--jp-elevation-z20);
  word-wrap: break-word;
  border-radius: var(--jp-border-radius);

  /* This is needed so that all font sizing of children done in ems is
   * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color1);
  resize: both;
}

.jp-Dialog-content.jp-Dialog-content-small {
  max-width: 500px;
}

.jp-Dialog-button {
  overflow: visible;
}

button.jp-Dialog-button:focus {
  outline: 1px solid var(--jp-brand-color1);
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button:focus::-moz-focus-inner {
  border: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus,
button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline-offset: 4px;
  -moz-outline-radius: 0;
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-accept:focus {
  outline: 1px solid var(--jp-accept-color-normal, var(--jp-brand-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-warn:focus {
  outline: 1px solid var(--jp-warn-color-normal, var(--jp-error-color1));
}

button.jp-Dialog-button.jp-mod-styled.jp-mod-reject:focus {
  outline: 1px solid var(--jp-reject-color-normal, var(--md-grey-600));
}

button.jp-Dialog-close-button {
  padding: 0;
  height: 100%;
  min-width: unset;
  min-height: unset;
}

.jp-Dialog-header {
  display: flex;
  justify-content: space-between;
  flex: 0 0 auto;
  padding-bottom: 12px;
  font-size: var(--jp-ui-font-size3);
  font-weight: 400;
  color: var(--jp-ui-font-color1);
}

.jp-Dialog-body {
  display: flex;
  flex-direction: column;
  flex: 1 1 auto;
  font-size: var(--jp-ui-font-size1);
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  overflow: auto;
}

.jp-Dialog-footer {
  display: flex;
  flex-direction: row;
  justify-content: flex-end;
  align-items: center;
  flex: 0 0 auto;
  margin-left: -12px;
  margin-right: -12px;
  padding: 12px;
}

.jp-Dialog-checkbox {
  padding-right: 5px;
}

.jp-Dialog-checkbox > input:focus-visible {
  outline: 1px solid var(--jp-input-active-border-color);
  outline-offset: 1px;
}

.jp-Dialog-spacer {
  flex: 1 1 auto;
}

.jp-Dialog-title {
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
}

.jp-Dialog-body > .jp-select-wrapper {
  width: 100%;
}

.jp-Dialog-body > button {
  padding: 0 16px;
}

.jp-Dialog-body > label {
  line-height: 1.4;
  color: var(--jp-ui-font-color0);
}

.jp-Dialog-button.jp-mod-styled:not(:last-child) {
  margin-right: 12px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Input-Boolean-Dialog {
  flex-direction: row-reverse;
  align-items: end;
  width: 100%;
}

.jp-Input-Boolean-Dialog > label {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MainAreaWidget > :focus {
  outline: none;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error {
  padding: 6px;
}

.jp-MainAreaWidget .jp-MainAreaWidget-error > pre {
  width: auto;
  padding: 10px;
  background: var(--jp-error-color3);
  border: var(--jp-border-width) solid var(--jp-error-color1);
  border-radius: var(--jp-border-radius);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  white-space: pre-wrap;
  word-wrap: break-word;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/**
 * google-material-color v1.2.6
 * https://github.com/danlevan/google-material-color
 */
:root {
  --md-red-50: #ffebee;
  --md-red-100: #ffcdd2;
  --md-red-200: #ef9a9a;
  --md-red-300: #e57373;
  --md-red-400: #ef5350;
  --md-red-500: #f44336;
  --md-red-600: #e53935;
  --md-red-700: #d32f2f;
  --md-red-800: #c62828;
  --md-red-900: #b71c1c;
  --md-red-A100: #ff8a80;
  --md-red-A200: #ff5252;
  --md-red-A400: #ff1744;
  --md-red-A700: #d50000;
  --md-pink-50: #fce4ec;
  --md-pink-100: #f8bbd0;
  --md-pink-200: #f48fb1;
  --md-pink-300: #f06292;
  --md-pink-400: #ec407a;
  --md-pink-500: #e91e63;
  --md-pink-600: #d81b60;
  --md-pink-700: #c2185b;
  --md-pink-800: #ad1457;
  --md-pink-900: #880e4f;
  --md-pink-A100: #ff80ab;
  --md-pink-A200: #ff4081;
  --md-pink-A400: #f50057;
  --md-pink-A700: #c51162;
  --md-purple-50: #f3e5f5;
  --md-purple-100: #e1bee7;
  --md-purple-200: #ce93d8;
  --md-purple-300: #ba68c8;
  --md-purple-400: #ab47bc;
  --md-purple-500: #9c27b0;
  --md-purple-600: #8e24aa;
  --md-purple-700: #7b1fa2;
  --md-purple-800: #6a1b9a;
  --md-purple-900: #4a148c;
  --md-purple-A100: #ea80fc;
  --md-purple-A200: #e040fb;
  --md-purple-A400: #d500f9;
  --md-purple-A700: #a0f;
  --md-deep-purple-50: #ede7f6;
  --md-deep-purple-100: #d1c4e9;
  --md-deep-purple-200: #b39ddb;
  --md-deep-purple-300: #9575cd;
  --md-deep-purple-400: #7e57c2;
  --md-deep-purple-500: #673ab7;
  --md-deep-purple-600: #5e35b1;
  --md-deep-purple-700: #512da8;
  --md-deep-purple-800: #4527a0;
  --md-deep-purple-900: #311b92;
  --md-deep-purple-A100: #b388ff;
  --md-deep-purple-A200: #7c4dff;
  --md-deep-purple-A400: #651fff;
  --md-deep-purple-A700: #6200ea;
  --md-indigo-50: #e8eaf6;
  --md-indigo-100: #c5cae9;
  --md-indigo-200: #9fa8da;
  --md-indigo-300: #7986cb;
  --md-indigo-400: #5c6bc0;
  --md-indigo-500: #3f51b5;
  --md-indigo-600: #3949ab;
  --md-indigo-700: #303f9f;
  --md-indigo-800: #283593;
  --md-indigo-900: #1a237e;
  --md-indigo-A100: #8c9eff;
  --md-indigo-A200: #536dfe;
  --md-indigo-A400: #3d5afe;
  --md-indigo-A700: #304ffe;
  --md-blue-50: #e3f2fd;
  --md-blue-100: #bbdefb;
  --md-blue-200: #90caf9;
  --md-blue-300: #64b5f6;
  --md-blue-400: #42a5f5;
  --md-blue-500: #2196f3;
  --md-blue-600: #1e88e5;
  --md-blue-700: #1976d2;
  --md-blue-800: #1565c0;
  --md-blue-900: #0d47a1;
  --md-blue-A100: #82b1ff;
  --md-blue-A200: #448aff;
  --md-blue-A400: #2979ff;
  --md-blue-A700: #2962ff;
  --md-light-blue-50: #e1f5fe;
  --md-light-blue-100: #b3e5fc;
  --md-light-blue-200: #81d4fa;
  --md-light-blue-300: #4fc3f7;
  --md-light-blue-400: #29b6f6;
  --md-light-blue-500: #03a9f4;
  --md-light-blue-600: #039be5;
  --md-light-blue-700: #0288d1;
  --md-light-blue-800: #0277bd;
  --md-light-blue-900: #01579b;
  --md-light-blue-A100: #80d8ff;
  --md-light-blue-A200: #40c4ff;
  --md-light-blue-A400: #00b0ff;
  --md-light-blue-A700: #0091ea;
  --md-cyan-50: #e0f7fa;
  --md-cyan-100: #b2ebf2;
  --md-cyan-200: #80deea;
  --md-cyan-300: #4dd0e1;
  --md-cyan-400: #26c6da;
  --md-cyan-500: #00bcd4;
  --md-cyan-600: #00acc1;
  --md-cyan-700: #0097a7;
  --md-cyan-800: #00838f;
  --md-cyan-900: #006064;
  --md-cyan-A100: #84ffff;
  --md-cyan-A200: #18ffff;
  --md-cyan-A400: #00e5ff;
  --md-cyan-A700: #00b8d4;
  --md-teal-50: #e0f2f1;
  --md-teal-100: #b2dfdb;
  --md-teal-200: #80cbc4;
  --md-teal-300: #4db6ac;
  --md-teal-400: #26a69a;
  --md-teal-500: #009688;
  --md-teal-600: #00897b;
  --md-teal-700: #00796b;
  --md-teal-800: #00695c;
  --md-teal-900: #004d40;
  --md-teal-A100: #a7ffeb;
  --md-teal-A200: #64ffda;
  --md-teal-A400: #1de9b6;
  --md-teal-A700: #00bfa5;
  --md-green-50: #e8f5e9;
  --md-green-100: #c8e6c9;
  --md-green-200: #a5d6a7;
  --md-green-300: #81c784;
  --md-green-400: #66bb6a;
  --md-green-500: #4caf50;
  --md-green-600: #43a047;
  --md-green-700: #388e3c;
  --md-green-800: #2e7d32;
  --md-green-900: #1b5e20;
  --md-green-A100: #b9f6ca;
  --md-green-A200: #69f0ae;
  --md-green-A400: #00e676;
  --md-green-A700: #00c853;
  --md-light-green-50: #f1f8e9;
  --md-light-green-100: #dcedc8;
  --md-light-green-200: #c5e1a5;
  --md-light-green-300: #aed581;
  --md-light-green-400: #9ccc65;
  --md-light-green-500: #8bc34a;
  --md-light-green-600: #7cb342;
  --md-light-green-700: #689f38;
  --md-light-green-800: #558b2f;
  --md-light-green-900: #33691e;
  --md-light-green-A100: #ccff90;
  --md-light-green-A200: #b2ff59;
  --md-light-green-A400: #76ff03;
  --md-light-green-A700: #64dd17;
  --md-lime-50: #f9fbe7;
  --md-lime-100: #f0f4c3;
  --md-lime-200: #e6ee9c;
  --md-lime-300: #dce775;
  --md-lime-400: #d4e157;
  --md-lime-500: #cddc39;
  --md-lime-600: #c0ca33;
  --md-lime-700: #afb42b;
  --md-lime-800: #9e9d24;
  --md-lime-900: #827717;
  --md-lime-A100: #f4ff81;
  --md-lime-A200: #eeff41;
  --md-lime-A400: #c6ff00;
  --md-lime-A700: #aeea00;
  --md-yellow-50: #fffde7;
  --md-yellow-100: #fff9c4;
  --md-yellow-200: #fff59d;
  --md-yellow-300: #fff176;
  --md-yellow-400: #ffee58;
  --md-yellow-500: #ffeb3b;
  --md-yellow-600: #fdd835;
  --md-yellow-700: #fbc02d;
  --md-yellow-800: #f9a825;
  --md-yellow-900: #f57f17;
  --md-yellow-A100: #ffff8d;
  --md-yellow-A200: #ff0;
  --md-yellow-A400: #ffea00;
  --md-yellow-A700: #ffd600;
  --md-amber-50: #fff8e1;
  --md-amber-100: #ffecb3;
  --md-amber-200: #ffe082;
  --md-amber-300: #ffd54f;
  --md-amber-400: #ffca28;
  --md-amber-500: #ffc107;
  --md-amber-600: #ffb300;
  --md-amber-700: #ffa000;
  --md-amber-800: #ff8f00;
  --md-amber-900: #ff6f00;
  --md-amber-A100: #ffe57f;
  --md-amber-A200: #ffd740;
  --md-amber-A400: #ffc400;
  --md-amber-A700: #ffab00;
  --md-orange-50: #fff3e0;
  --md-orange-100: #ffe0b2;
  --md-orange-200: #ffcc80;
  --md-orange-300: #ffb74d;
  --md-orange-400: #ffa726;
  --md-orange-500: #ff9800;
  --md-orange-600: #fb8c00;
  --md-orange-700: #f57c00;
  --md-orange-800: #ef6c00;
  --md-orange-900: #e65100;
  --md-orange-A100: #ffd180;
  --md-orange-A200: #ffab40;
  --md-orange-A400: #ff9100;
  --md-orange-A700: #ff6d00;
  --md-deep-orange-50: #fbe9e7;
  --md-deep-orange-100: #ffccbc;
  --md-deep-orange-200: #ffab91;
  --md-deep-orange-300: #ff8a65;
  --md-deep-orange-400: #ff7043;
  --md-deep-orange-500: #ff5722;
  --md-deep-orange-600: #f4511e;
  --md-deep-orange-700: #e64a19;
  --md-deep-orange-800: #d84315;
  --md-deep-orange-900: #bf360c;
  --md-deep-orange-A100: #ff9e80;
  --md-deep-orange-A200: #ff6e40;
  --md-deep-orange-A400: #ff3d00;
  --md-deep-orange-A700: #dd2c00;
  --md-brown-50: #efebe9;
  --md-brown-100: #d7ccc8;
  --md-brown-200: #bcaaa4;
  --md-brown-300: #a1887f;
  --md-brown-400: #8d6e63;
  --md-brown-500: #795548;
  --md-brown-600: #6d4c41;
  --md-brown-700: #5d4037;
  --md-brown-800: #4e342e;
  --md-brown-900: #3e2723;
  --md-grey-50: #fafafa;
  --md-grey-100: #f5f5f5;
  --md-grey-200: #eee;
  --md-grey-300: #e0e0e0;
  --md-grey-400: #bdbdbd;
  --md-grey-500: #9e9e9e;
  --md-grey-600: #757575;
  --md-grey-700: #616161;
  --md-grey-800: #424242;
  --md-grey-900: #212121;
  --md-blue-grey-50: #eceff1;
  --md-blue-grey-100: #cfd8dc;
  --md-blue-grey-200: #b0bec5;
  --md-blue-grey-300: #90a4ae;
  --md-blue-grey-400: #78909c;
  --md-blue-grey-500: #607d8b;
  --md-blue-grey-600: #546e7a;
  --md-blue-grey-700: #455a64;
  --md-blue-grey-800: #37474f;
  --md-blue-grey-900: #263238;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2017, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| RenderedText
|----------------------------------------------------------------------------*/

:root {
  /* This is the padding value to fill the gaps between lines containing spans with background color. */
  --jp-private-code-span-padding: calc(
    (var(--jp-code-line-height) - 1) * var(--jp-code-font-size) / 2
  );
}

.jp-RenderedText {
  text-align: left;
  padding-left: var(--jp-code-padding);
  line-height: var(--jp-code-line-height);
  font-family: var(--jp-code-font-family);
}

.jp-RenderedText pre,
.jp-RenderedJavaScript pre,
.jp-RenderedHTMLCommon pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
  border: none;
  margin: 0;
  padding: 0;
}

.jp-RenderedText pre a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedText pre a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* console foregrounds and backgrounds */
.jp-RenderedText pre .ansi-black-fg {
  color: #3e424d;
}

.jp-RenderedText pre .ansi-red-fg {
  color: #e75c58;
}

.jp-RenderedText pre .ansi-green-fg {
  color: #00a250;
}

.jp-RenderedText pre .ansi-yellow-fg {
  color: #ddb62b;
}

.jp-RenderedText pre .ansi-blue-fg {
  color: #208ffb;
}

.jp-RenderedText pre .ansi-magenta-fg {
  color: #d160c4;
}

.jp-RenderedText pre .ansi-cyan-fg {
  color: #60c6c8;
}

.jp-RenderedText pre .ansi-white-fg {
  color: #c5c1b4;
}

.jp-RenderedText pre .ansi-black-bg {
  background-color: #3e424d;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-bg {
  background-color: #e75c58;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-bg {
  background-color: #00a250;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-bg {
  background-color: #ddb62b;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-bg {
  background-color: #208ffb;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-bg {
  background-color: #d160c4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-bg {
  background-color: #60c6c8;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-bg {
  background-color: #c5c1b4;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-black-intense-fg {
  color: #282c36;
}

.jp-RenderedText pre .ansi-red-intense-fg {
  color: #b22b31;
}

.jp-RenderedText pre .ansi-green-intense-fg {
  color: #007427;
}

.jp-RenderedText pre .ansi-yellow-intense-fg {
  color: #b27d12;
}

.jp-RenderedText pre .ansi-blue-intense-fg {
  color: #0065ca;
}

.jp-RenderedText pre .ansi-magenta-intense-fg {
  color: #a03196;
}

.jp-RenderedText pre .ansi-cyan-intense-fg {
  color: #258f8f;
}

.jp-RenderedText pre .ansi-white-intense-fg {
  color: #a1a6b2;
}

.jp-RenderedText pre .ansi-black-intense-bg {
  background-color: #282c36;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-red-intense-bg {
  background-color: #b22b31;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-green-intense-bg {
  background-color: #007427;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-yellow-intense-bg {
  background-color: #b27d12;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-blue-intense-bg {
  background-color: #0065ca;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-magenta-intense-bg {
  background-color: #a03196;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-cyan-intense-bg {
  background-color: #258f8f;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-white-intense-bg {
  background-color: #a1a6b2;
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-default-inverse-fg {
  color: var(--jp-ui-inverse-font-color0);
}

.jp-RenderedText pre .ansi-default-inverse-bg {
  background-color: var(--jp-inverse-layout-color0);
  padding: var(--jp-private-code-span-padding) 0;
}

.jp-RenderedText pre .ansi-bold {
  font-weight: bold;
}

.jp-RenderedText pre .ansi-underline {
  text-decoration: underline;
}

.jp-RenderedText[data-mime-type='application/vnd.jupyter.stderr'] {
  background: var(--jp-rendermime-error-background);
  padding-top: var(--jp-code-padding);
}

/*-----------------------------------------------------------------------------
| RenderedLatex
|----------------------------------------------------------------------------*/

.jp-RenderedLatex {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);
}

/* Left-justify outputs.*/
.jp-OutputArea-output.jp-RenderedLatex {
  padding: var(--jp-code-padding);
  text-align: left;
}

/*-----------------------------------------------------------------------------
| RenderedHTML
|----------------------------------------------------------------------------*/

.jp-RenderedHTMLCommon {
  color: var(--jp-content-font-color1);
  font-family: var(--jp-content-font-family);
  font-size: var(--jp-content-font-size1);
  line-height: var(--jp-content-line-height);

  /* Give a bit more R padding on Markdown text to keep line lengths reasonable */
  padding-right: 20px;
}

.jp-RenderedHTMLCommon em {
  font-style: italic;
}

.jp-RenderedHTMLCommon strong {
  font-weight: bold;
}

.jp-RenderedHTMLCommon u {
  text-decoration: underline;
}

.jp-RenderedHTMLCommon a:link {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:hover {
  text-decoration: underline;
  color: var(--jp-content-link-color);
}

.jp-RenderedHTMLCommon a:visited {
  text-decoration: none;
  color: var(--jp-content-link-color);
}

/* Headings */

.jp-RenderedHTMLCommon h1,
.jp-RenderedHTMLCommon h2,
.jp-RenderedHTMLCommon h3,
.jp-RenderedHTMLCommon h4,
.jp-RenderedHTMLCommon h5,
.jp-RenderedHTMLCommon h6 {
  line-height: var(--jp-content-heading-line-height);
  font-weight: var(--jp-content-heading-font-weight);
  font-style: normal;
  margin: var(--jp-content-heading-margin-top) 0
    var(--jp-content-heading-margin-bottom) 0;
}

.jp-RenderedHTMLCommon h1:first-child,
.jp-RenderedHTMLCommon h2:first-child,
.jp-RenderedHTMLCommon h3:first-child,
.jp-RenderedHTMLCommon h4:first-child,
.jp-RenderedHTMLCommon h5:first-child,
.jp-RenderedHTMLCommon h6:first-child {
  margin-top: calc(0.5 * var(--jp-content-heading-margin-top));
}

.jp-RenderedHTMLCommon h1:last-child,
.jp-RenderedHTMLCommon h2:last-child,
.jp-RenderedHTMLCommon h3:last-child,
.jp-RenderedHTMLCommon h4:last-child,
.jp-RenderedHTMLCommon h5:last-child,
.jp-RenderedHTMLCommon h6:last-child {
  margin-bottom: calc(0.5 * var(--jp-content-heading-margin-bottom));
}

.jp-RenderedHTMLCommon h1 {
  font-size: var(--jp-content-font-size5);
}

.jp-RenderedHTMLCommon h2 {
  font-size: var(--jp-content-font-size4);
}

.jp-RenderedHTMLCommon h3 {
  font-size: var(--jp-content-font-size3);
}

.jp-RenderedHTMLCommon h4 {
  font-size: var(--jp-content-font-size2);
}

.jp-RenderedHTMLCommon h5 {
  font-size: var(--jp-content-font-size1);
}

.jp-RenderedHTMLCommon h6 {
  font-size: var(--jp-content-font-size0);
}

/* Lists */

/* stylelint-disable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon ul:not(.list-inline),
.jp-RenderedHTMLCommon ol:not(.list-inline) {
  padding-left: 2em;
}

.jp-RenderedHTMLCommon ul {
  list-style: disc;
}

.jp-RenderedHTMLCommon ul ul {
  list-style: square;
}

.jp-RenderedHTMLCommon ul ul ul {
  list-style: circle;
}

.jp-RenderedHTMLCommon ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol ol {
  list-style: upper-alpha;
}

.jp-RenderedHTMLCommon ol ol ol {
  list-style: lower-alpha;
}

.jp-RenderedHTMLCommon ol ol ol ol {
  list-style: lower-roman;
}

.jp-RenderedHTMLCommon ol ol ol ol ol {
  list-style: decimal;
}

.jp-RenderedHTMLCommon ol,
.jp-RenderedHTMLCommon ul {
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon ul ul,
.jp-RenderedHTMLCommon ul ol,
.jp-RenderedHTMLCommon ol ul,
.jp-RenderedHTMLCommon ol ol {
  margin-bottom: 0;
}

/* stylelint-enable selector-max-type, selector-max-compound-selectors */

.jp-RenderedHTMLCommon hr {
  color: var(--jp-border-color2);
  background-color: var(--jp-border-color1);
  margin-top: 1em;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon > pre {
  margin: 1.5em 2em;
}

.jp-RenderedHTMLCommon pre,
.jp-RenderedHTMLCommon code {
  border: 0;
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  line-height: var(--jp-code-line-height);
  padding: 0;
  white-space: pre-wrap;
}

.jp-RenderedHTMLCommon :not(pre) > code {
  background-color: var(--jp-layout-color2);
  padding: 1px 5px;
}

/* Tables */

.jp-RenderedHTMLCommon table {
  border-collapse: collapse;
  border-spacing: 0;
  border: none;
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  table-layout: fixed;
  margin-left: auto;
  margin-bottom: 1em;
  margin-right: auto;
}

.jp-RenderedHTMLCommon thead {
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  vertical-align: bottom;
}

.jp-RenderedHTMLCommon td,
.jp-RenderedHTMLCommon th,
.jp-RenderedHTMLCommon tr {
  vertical-align: middle;
  padding: 0.5em;
  line-height: normal;
  white-space: normal;
  max-width: none;
  border: none;
}

.jp-RenderedMarkdown.jp-RenderedHTMLCommon td,
.jp-RenderedMarkdown.jp-RenderedHTMLCommon th {
  max-width: none;
}

:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon td,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon th,
:not(.jp-RenderedMarkdown).jp-RenderedHTMLCommon tr {
  text-align: right;
}

.jp-RenderedHTMLCommon th {
  font-weight: bold;
}

.jp-RenderedHTMLCommon tbody tr:nth-child(odd) {
  background: var(--jp-layout-color0);
}

.jp-RenderedHTMLCommon tbody tr:nth-child(even) {
  background: var(--jp-rendermime-table-row-background);
}

.jp-RenderedHTMLCommon tbody tr:hover {
  background: var(--jp-rendermime-table-row-hover-background);
}

.jp-RenderedHTMLCommon p {
  text-align: left;
  margin: 0;
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon img {
  -moz-force-broken-image-icon: 1;
}

/* Restrict to direct children as other images could be nested in other content. */
.jp-RenderedHTMLCommon > img {
  display: block;
  margin-left: 0;
  margin-right: 0;
  margin-bottom: 1em;
}

/* Change color behind transparent images if they need it... */
[data-jp-theme-light='false'] .jp-RenderedImage img.jp-needs-light-background {
  background-color: var(--jp-inverse-layout-color1);
}

[data-jp-theme-light='true'] .jp-RenderedImage img.jp-needs-dark-background {
  background-color: var(--jp-inverse-layout-color1);
}

.jp-RenderedHTMLCommon img,
.jp-RenderedImage img,
.jp-RenderedHTMLCommon svg,
.jp-RenderedSVG svg {
  max-width: 100%;
  height: auto;
}

.jp-RenderedHTMLCommon img.jp-mod-unconfined,
.jp-RenderedImage img.jp-mod-unconfined,
.jp-RenderedHTMLCommon svg.jp-mod-unconfined,
.jp-RenderedSVG svg.jp-mod-unconfined {
  max-width: none;
}

.jp-RenderedHTMLCommon .alert {
  padding: var(--jp-notebook-padding);
  border: var(--jp-border-width) solid transparent;
  border-radius: var(--jp-border-radius);
  margin-bottom: 1em;
}

.jp-RenderedHTMLCommon .alert-info {
  color: var(--jp-info-color0);
  background-color: var(--jp-info-color3);
  border-color: var(--jp-info-color2);
}

.jp-RenderedHTMLCommon .alert-info hr {
  border-color: var(--jp-info-color3);
}

.jp-RenderedHTMLCommon .alert-info > p:last-child,
.jp-RenderedHTMLCommon .alert-info > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-warning {
  color: var(--jp-warn-color0);
  background-color: var(--jp-warn-color3);
  border-color: var(--jp-warn-color2);
}

.jp-RenderedHTMLCommon .alert-warning hr {
  border-color: var(--jp-warn-color3);
}

.jp-RenderedHTMLCommon .alert-warning > p:last-child,
.jp-RenderedHTMLCommon .alert-warning > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-success {
  color: var(--jp-success-color0);
  background-color: var(--jp-success-color3);
  border-color: var(--jp-success-color2);
}

.jp-RenderedHTMLCommon .alert-success hr {
  border-color: var(--jp-success-color3);
}

.jp-RenderedHTMLCommon .alert-success > p:last-child,
.jp-RenderedHTMLCommon .alert-success > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon .alert-danger {
  color: var(--jp-error-color0);
  background-color: var(--jp-error-color3);
  border-color: var(--jp-error-color2);
}

.jp-RenderedHTMLCommon .alert-danger hr {
  border-color: var(--jp-error-color3);
}

.jp-RenderedHTMLCommon .alert-danger > p:last-child,
.jp-RenderedHTMLCommon .alert-danger > ul:last-child {
  margin-bottom: 0;
}

.jp-RenderedHTMLCommon blockquote {
  margin: 1em 2em;
  padding: 0 1em;
  border-left: 5px solid var(--jp-border-color2);
}

a.jp-InternalAnchorLink {
  visibility: hidden;
  margin-left: 8px;
  color: var(--md-blue-800);
}

h1:hover .jp-InternalAnchorLink,
h2:hover .jp-InternalAnchorLink,
h3:hover .jp-InternalAnchorLink,
h4:hover .jp-InternalAnchorLink,
h5:hover .jp-InternalAnchorLink,
h6:hover .jp-InternalAnchorLink {
  visibility: visible;
}

.jp-RenderedHTMLCommon kbd {
  background-color: var(--jp-rendermime-table-row-background);
  border: 1px solid var(--jp-border-color0);
  border-bottom-color: var(--jp-border-color2);
  border-radius: 3px;
  box-shadow: inset 0 -1px 0 rgba(0, 0, 0, 0.25);
  display: inline-block;
  font-size: var(--jp-ui-font-size0);
  line-height: 1em;
  padding: 0.2em 0.5em;
}

/* Most direct children of .jp-RenderedHTMLCommon have a margin-bottom of 1.0.
 * At the bottom of cells this is a bit too much as there is also spacing
 * between cells. Going all the way to 0 gets too tight between markdown and
 * code cells.
 */
.jp-RenderedHTMLCommon > *:last-child {
  margin-bottom: 0.5em;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Copyright (c) 2014-2017, PhosphorJS Contributors
|
| Distributed under the terms of the BSD 3-Clause License.
|
| The full license is in the file LICENSE, distributed with this software.
|----------------------------------------------------------------------------*/

.lm-cursor-backdrop {
  position: fixed;
  width: 200px;
  height: 200px;
  margin-top: -100px;
  margin-left: -100px;
  will-change: transform;
  z-index: 100;
}

.lm-mod-drag-image {
  will-change: transform;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-lineFormSearch {
  padding: 4px 12px;
  background-color: var(--jp-layout-color2);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
  font-size: var(--jp-ui-font-size1);
}

.jp-lineFormCaption {
  font-size: var(--jp-ui-font-size0);
  line-height: var(--jp-ui-font-size1);
  margin-top: 4px;
  color: var(--jp-ui-font-color0);
}

.jp-baseLineForm {
  border: none;
  border-radius: 0;
  position: absolute;
  background-size: 16px;
  background-repeat: no-repeat;
  background-position: center;
  outline: none;
}

.jp-lineFormButtonContainer {
  top: 4px;
  right: 8px;
  height: 24px;
  padding: 0 12px;
  width: 12px;
}

.jp-lineFormButtonIcon {
  top: 0;
  right: 0;
  background-color: var(--jp-brand-color1);
  height: 100%;
  width: 100%;
  box-sizing: border-box;
  padding: 4px 6px;
}

.jp-lineFormButton {
  top: 0;
  right: 0;
  background-color: transparent;
  height: 100%;
  width: 100%;
  box-sizing: border-box;
}

.jp-lineFormWrapper {
  overflow: hidden;
  padding: 0 8px;
  border: 1px solid var(--jp-border-color0);
  background-color: var(--jp-input-active-background);
  height: 22px;
}

.jp-lineFormWrapperFocusWithin {
  border: var(--jp-border-width) solid var(--md-blue-500);
  box-shadow: inset 0 0 4px var(--md-blue-300);
}

.jp-lineFormInput {
  background: transparent;
  width: 200px;
  height: 100%;
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  line-height: 28px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) 2014-2016, Jupyter Development Team.
|
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-JSONEditor {
  display: flex;
  flex-direction: column;
  width: 100%;
}

.jp-JSONEditor-host {
  flex: 1 1 auto;
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  border-radius: 0;
  background: var(--jp-layout-color0);
  min-height: 50px;
  padding: 1px;
}

.jp-JSONEditor.jp-mod-error .jp-JSONEditor-host {
  border-color: red;
  outline-color: red;
}

.jp-JSONEditor-header {
  display: flex;
  flex: 1 0 auto;
  padding: 0 0 0 12px;
}

.jp-JSONEditor-header label {
  flex: 0 0 auto;
}

.jp-JSONEditor-commitButton {
  height: 16px;
  width: 16px;
  background-size: 18px;
  background-repeat: no-repeat;
  background-position: center;
}

.jp-JSONEditor-host.jp-mod-focused {
  background-color: var(--jp-input-active-background);
  border: 1px solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

.jp-Editor.jp-mod-dropTarget {
  border: var(--jp-border-width) solid var(--jp-input-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/
.jp-DocumentSearch-input {
  border: none;
  outline: none;
  color: var(--jp-ui-font-color0);
  font-size: var(--jp-ui-font-size1);
  background-color: var(--jp-layout-color0);
  font-family: var(--jp-ui-font-family);
  padding: 2px 1px;
  resize: none;
}

.jp-DocumentSearch-overlay {
  position: absolute;
  background-color: var(--jp-toolbar-background);
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  border-left: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  top: 0;
  right: 0;
  z-index: 7;
  min-width: 405px;
  padding: 2px;
  font-size: var(--jp-ui-font-size1);

  --jp-private-document-search-button-height: 20px;
}

.jp-DocumentSearch-overlay button {
  background-color: var(--jp-toolbar-background);
  outline: 0;
}

.jp-DocumentSearch-overlay button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-overlay button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-overlay-row {
  display: flex;
  align-items: center;
  margin-bottom: 2px;
}

.jp-DocumentSearch-button-content {
  display: inline-block;
  cursor: pointer;
  box-sizing: border-box;
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-button-content svg {
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-input-wrapper {
  border: var(--jp-border-width) solid var(--jp-border-color0);
  display: flex;
  background-color: var(--jp-layout-color0);
  margin: 2px;
}

.jp-DocumentSearch-input-wrapper:focus-within {
  border-color: var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper {
  all: initial;
  overflow: hidden;
  display: inline-block;
  border: none;
  box-sizing: border-box;
}

.jp-DocumentSearch-toggle-wrapper {
  width: 14px;
  height: 14px;
}

.jp-DocumentSearch-button-wrapper {
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
}

.jp-DocumentSearch-toggle-wrapper:focus,
.jp-DocumentSearch-button-wrapper:focus {
  outline: var(--jp-border-width) solid
    var(--jp-cell-editor-active-border-color);
  outline-offset: -1px;
}

.jp-DocumentSearch-toggle-wrapper,
.jp-DocumentSearch-button-wrapper,
.jp-DocumentSearch-button-content:focus {
  outline: none;
}

.jp-DocumentSearch-toggle-placeholder {
  width: 5px;
}

.jp-DocumentSearch-input-button::before {
  display: block;
  padding-top: 100%;
}

.jp-DocumentSearch-input-button-off {
  opacity: var(--jp-search-toggle-off-opacity);
}

.jp-DocumentSearch-input-button-off:hover {
  opacity: var(--jp-search-toggle-hover-opacity);
}

.jp-DocumentSearch-input-button-on {
  opacity: var(--jp-search-toggle-on-opacity);
}

.jp-DocumentSearch-index-counter {
  padding-left: 10px;
  padding-right: 10px;
  user-select: none;
  min-width: 35px;
  display: inline-block;
}

.jp-DocumentSearch-up-down-wrapper {
  display: inline-block;
  padding-right: 2px;
  margin-left: auto;
  white-space: nowrap;
}

.jp-DocumentSearch-spacer {
  margin-left: auto;
}

.jp-DocumentSearch-up-down-wrapper button {
  outline: 0;
  border: none;
  width: var(--jp-private-document-search-button-height);
  height: var(--jp-private-document-search-button-height);
  vertical-align: middle;
  margin: 1px 5px 2px;
}

.jp-DocumentSearch-up-down-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-up-down-button:active {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-filter-button {
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-filter-button:hover {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled {
  background-color: var(--jp-layout-color2);
}

.jp-DocumentSearch-filter-button-enabled:hover {
  background-color: var(--jp-layout-color3);
}

.jp-DocumentSearch-search-options {
  padding: 0 8px;
  margin-left: 3px;
  width: 100%;
  display: grid;
  justify-content: start;
  grid-template-columns: 1fr 1fr;
  align-items: center;
  justify-items: stretch;
}

.jp-DocumentSearch-search-filter-disabled {
  color: var(--jp-ui-font-color2);
}

.jp-DocumentSearch-search-filter {
  display: flex;
  align-items: center;
  user-select: none;
}

.jp-DocumentSearch-regex-error {
  color: var(--jp-error-color0);
}

.jp-DocumentSearch-replace-button-wrapper {
  overflow: hidden;
  display: inline-block;
  box-sizing: border-box;
  border: var(--jp-border-width) solid var(--jp-border-color0);
  margin: auto 2px;
  padding: 1px 4px;
  height: calc(var(--jp-private-document-search-button-height) + 2px);
}

.jp-DocumentSearch-replace-button-wrapper:focus {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
}

.jp-DocumentSearch-replace-button {
  display: inline-block;
  text-align: center;
  cursor: pointer;
  box-sizing: border-box;
  color: var(--jp-ui-font-color1);

  /* height - 2 * (padding of wrapper) */
  line-height: calc(var(--jp-private-document-search-button-height) - 2px);
  width: 100%;
  height: 100%;
}

.jp-DocumentSearch-replace-button:focus {
  outline: none;
}

.jp-DocumentSearch-replace-wrapper-class {
  margin-left: 14px;
  display: flex;
}

.jp-DocumentSearch-replace-toggle {
  border: none;
  background-color: var(--jp-toolbar-background);
  border-radius: var(--jp-border-radius);
}

.jp-DocumentSearch-replace-toggle:hover {
  background-color: var(--jp-layout-color2);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.cm-editor {
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  border: 0;
  border-radius: 0;
  height: auto;

  /* Changed to auto to autogrow */
}

.cm-editor pre {
  padding: 0 var(--jp-code-padding);
}

.jp-CodeMirrorEditor[data-type='inline'] .cm-dialog {
  background-color: var(--jp-layout-color0);
  color: var(--jp-content-font-color1);
}

.jp-CodeMirrorEditor {
  cursor: text;
}

/* When zoomed out 67% and 33% on a screen of 1440 width x 900 height */
@media screen and (min-width: 2138px) and (max-width: 4319px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width1) solid
      var(--jp-editor-cursor-color);
  }
}

/* When zoomed out less than 33% */
@media screen and (min-width: 4320px) {
  .jp-CodeMirrorEditor[data-type='inline'] .cm-cursor {
    border-left: var(--jp-code-cursor-width2) solid
      var(--jp-editor-cursor-color);
  }
}

.cm-editor.jp-mod-readOnly .cm-cursor {
  display: none;
}

.jp-CollaboratorCursor {
  border-left: 5px solid transparent;
  border-right: 5px solid transparent;
  border-top: none;
  border-bottom: 3px solid;
  background-clip: content-box;
  margin-left: -5px;
  margin-right: -5px;
}

.cm-searching,
.cm-searching span {
  /* `.cm-searching span`: we need to override syntax highlighting */
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.cm-searching::selection,
.cm-searching span::selection {
  background-color: var(--jp-search-unselected-match-background-color);
  color: var(--jp-search-unselected-match-color);
}

.jp-current-match > .cm-searching,
.jp-current-match > .cm-searching span,
.cm-searching > .jp-current-match,
.cm-searching > .jp-current-match span {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.jp-current-match > .cm-searching::selection,
.cm-searching > .jp-current-match::selection,
.jp-current-match > .cm-searching span::selection {
  background-color: var(--jp-search-selected-match-background-color);
  color: var(--jp-search-selected-match-color);
}

.cm-trailingspace {
  background-image: url(data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAgAAAAFCAYAAAB4ka1VAAAAsElEQVQIHQGlAFr/AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA7+r3zKmT0/+pk9P/7+r3zAAAAAAAAAAABAAAAAAAAAAA6OPzM+/q9wAAAAAA6OPzMwAAAAAAAAAAAgAAAAAAAAAAGR8NiRQaCgAZIA0AGR8NiQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAQyoYJ/SY80UAAAAASUVORK5CYII=);
  background-position: center left;
  background-repeat: repeat-x;
}

.jp-CollaboratorCursor-hover {
  position: absolute;
  z-index: 1;
  transform: translateX(-50%);
  color: white;
  border-radius: 3px;
  padding-left: 4px;
  padding-right: 4px;
  padding-top: 1px;
  padding-bottom: 1px;
  text-align: center;
  font-size: var(--jp-ui-font-size1);
  white-space: nowrap;
}

.jp-CodeMirror-ruler {
  border-left: 1px dashed var(--jp-border-color2);
}

/* Styles for shared cursors (remote cursor locations and selected ranges) */
.jp-CodeMirrorEditor .cm-ySelectionCaret {
  position: relative;
  border-left: 1px solid black;
  margin-left: -1px;
  margin-right: -1px;
  box-sizing: border-box;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret > .cm-ySelectionInfo {
  white-space: nowrap;
  position: absolute;
  top: -1.15em;
  padding-bottom: 0.05em;
  left: -1px;
  font-size: 0.95em;
  font-family: var(--jp-ui-font-family);
  font-weight: bold;
  line-height: normal;
  user-select: none;
  color: white;
  padding-left: 2px;
  padding-right: 2px;
  z-index: 101;
  transition: opacity 0.3s ease-in-out;
}

.jp-CodeMirrorEditor .cm-ySelectionInfo {
  transition-delay: 0.7s;
  opacity: 0;
}

.jp-CodeMirrorEditor .cm-ySelectionCaret:hover > .cm-ySelectionInfo {
  opacity: 1;
  transition-delay: 0s;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-MimeDocument {
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-filebrowser-button-height: 28px;
  --jp-private-filebrowser-button-width: 48px;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-FileBrowser .jp-SidePanel-content {
  display: flex;
  flex-direction: column;
}

.jp-FileBrowser-toolbar.jp-Toolbar {
  flex-wrap: wrap;
  row-gap: 12px;
  border-bottom: none;
  height: auto;
  margin: 8px 12px 0;
  box-shadow: none;
  padding: 0;
  justify-content: flex-start;
}

.jp-FileBrowser-Panel {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
}

.jp-BreadCrumbs {
  flex: 0 0 auto;
  margin: 8px 12px;
}

.jp-BreadCrumbs-item {
  margin: 0 2px;
  padding: 0 2px;
  border-radius: var(--jp-border-radius);
  cursor: pointer;
}

.jp-BreadCrumbs-item:hover {
  background-color: var(--jp-layout-color2);
}

.jp-BreadCrumbs-item:first-child {
  margin-left: 0;
}

.jp-BreadCrumbs-item.jp-mod-dropTarget {
  background-color: var(--jp-brand-color2);
  opacity: 0.7;
}

/*-----------------------------------------------------------------------------
| Buttons
|----------------------------------------------------------------------------*/

.jp-FileBrowser-toolbar > .jp-Toolbar-item {
  flex: 0 0 auto;
  padding-left: 0;
  padding-right: 2px;
  align-items: center;
  height: unset;
}

.jp-FileBrowser-toolbar > .jp-Toolbar-item .jp-ToolbarButtonComponent {
  width: 40px;
}

/*-----------------------------------------------------------------------------
| Other styles
|----------------------------------------------------------------------------*/

.jp-FileDialog.jp-mod-conflict input {
  color: var(--jp-error-color1);
}

.jp-FileDialog .jp-new-name-title {
  margin-top: 12px;
}

.jp-LastModified-hidden {
  display: none;
}

.jp-FileSize-hidden {
  display: none;
}

.jp-FileBrowser .lm-AccordionPanel > h3:first-child {
  display: none;
}

/*-----------------------------------------------------------------------------
| DirListing
|----------------------------------------------------------------------------*/

.jp-DirListing {
  flex: 1 1 auto;
  display: flex;
  flex-direction: column;
  outline: 0;
}

.jp-DirListing-header {
  flex: 0 0 auto;
  display: flex;
  flex-direction: row;
  align-items: center;
  overflow: hidden;
  border-top: var(--jp-border-width) solid var(--jp-border-color2);
  border-bottom: var(--jp-border-width) solid var(--jp-border-color1);
  box-shadow: var(--jp-toolbar-box-shadow);
  z-index: 2;
}

.jp-DirListing-headerItem {
  padding: 4px 12px 2px;
  font-weight: 500;
}

.jp-DirListing-headerItem:hover {
  background: var(--jp-layout-color2);
}

.jp-DirListing-headerItem.jp-id-name {
  flex: 1 0 84px;
}

.jp-DirListing-headerItem.jp-id-modified {
  flex: 0 0 112px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-DirListing-headerItem.jp-id-filesize {
  flex: 0 0 75px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
}

.jp-id-narrow {
  display: none;
  flex: 0 0 5px;
  padding: 4px;
  border-left: var(--jp-border-width) solid var(--jp-border-color2);
  text-align: right;
  color: var(--jp-border-color2);
}

.jp-DirListing-narrow .jp-id-narrow {
  display: block;
}

.jp-DirListing-narrow .jp-id-modified,
.jp-DirListing-narrow .jp-DirListing-itemModified {
  display: none;
}

.jp-DirListing-headerItem.jp-mod-selected {
  font-weight: 600;
}

/* increase specificity to override bundled default */
.jp-DirListing-content {
  flex: 1 1 auto;
  margin: 0;
  padding: 0;
  list-style-type: none;
  overflow: auto;
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-content mark {
  color: var(--jp-ui-font-color0);
  background-color: transparent;
  font-weight: bold;
}

.jp-DirListing-content .jp-DirListing-item.jp-mod-selected mark {
  color: var(--jp-ui-inverse-font-color0);
}

/* Style the directory listing content when a user drops a file to upload */
.jp-DirListing.jp-mod-native-drop .jp-DirListing-content {
  outline: 5px dashed rgba(128, 128, 128, 0.5);
  outline-offset: -10px;
  cursor: copy;
}

.jp-DirListing-item {
  display: flex;
  flex-direction: row;
  align-items: center;
  padding: 4px 12px;
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-DirListing-checkboxWrapper {
  /* Increases hit area of checkbox. */
  padding: 4px;
}

.jp-DirListing-header
  .jp-DirListing-checkboxWrapper
  + .jp-DirListing-headerItem {
  padding-left: 4px;
}

.jp-DirListing-content .jp-DirListing-checkboxWrapper {
  position: relative;
  left: -4px;
  margin: -4px 0 -4px -8px;
}

.jp-DirListing-checkboxWrapper.jp-mod-visible {
  visibility: visible;
}

/* For devices that support hovering, hide checkboxes until hovered, selected...
*/
@media (hover: hover) {
  .jp-DirListing-checkboxWrapper {
    visibility: hidden;
  }

  .jp-DirListing-item:hover .jp-DirListing-checkboxWrapper,
  .jp-DirListing-item.jp-mod-selected .jp-DirListing-checkboxWrapper {
    visibility: visible;
  }
}

.jp-DirListing-item[data-is-dot] {
  opacity: 75%;
}

.jp-DirListing-item.jp-mod-selected {
  color: var(--jp-ui-inverse-font-color1);
  background: var(--jp-brand-color1);
}

.jp-DirListing-item.jp-mod-dropTarget {
  background: var(--jp-brand-color3);
}

.jp-DirListing-item:hover:not(.jp-mod-selected) {
  background: var(--jp-layout-color2);
}

.jp-DirListing-itemIcon {
  flex: 0 0 20px;
  margin-right: 4px;
}

.jp-DirListing-itemText {
  flex: 1 0 64px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  user-select: none;
}

.jp-DirListing-itemText:focus {
  outline-width: 2px;
  outline-color: var(--jp-inverse-layout-color1);
  outline-style: solid;
  outline-offset: 1px;
}

.jp-DirListing-item.jp-mod-selected .jp-DirListing-itemText:focus {
  outline-color: var(--jp-layout-color1);
}

.jp-DirListing-itemModified {
  flex: 0 0 125px;
  text-align: right;
}

.jp-DirListing-itemFileSize {
  flex: 0 0 90px;
  text-align: right;
}

.jp-DirListing-editor {
  flex: 1 0 64px;
  outline: none;
  border: none;
  color: var(--jp-ui-font-color1);
  background-color: var(--jp-layout-color1);
}

.jp-DirListing-item.jp-mod-running .jp-DirListing-itemIcon::before {
  color: var(--jp-success-color1);
  content: '\25CF';
  font-size: 8px;
  position: absolute;
  left: -8px;
}

.jp-DirListing-item.jp-mod-running.jp-mod-selected
  .jp-DirListing-itemIcon::before {
  color: var(--jp-ui-inverse-font-color1);
}

.jp-DirListing-item.lm-mod-drag-image,
.jp-DirListing-item.jp-mod-selected.lm-mod-drag-image {
  font-size: var(--jp-ui-font-size1);
  padding-left: 4px;
  margin-left: 4px;
  width: 160px;
  background-color: var(--jp-ui-inverse-font-color2);
  box-shadow: var(--jp-elevation-z2);
  border-radius: 0;
  color: var(--jp-ui-font-color1);
  transform: translateX(-40%) translateY(-58%);
}

.jp-Document {
  min-width: 120px;
  min-height: 120px;
  outline: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Main OutputArea
| OutputArea has a list of Outputs
|----------------------------------------------------------------------------*/

.jp-OutputArea {
  overflow-y: auto;
}

.jp-OutputArea-child {
  display: table;
  table-layout: fixed;
  width: 100%;
  overflow: hidden;
}

.jp-OutputPrompt {
  width: var(--jp-cell-prompt-width);
  color: var(--jp-cell-outprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
  opacity: var(--jp-cell-prompt-opacity);

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-OutputArea-prompt {
  display: table-cell;
  vertical-align: top;
}

.jp-OutputArea-output {
  display: table-cell;
  width: 100%;
  height: auto;
  overflow: auto;
  user-select: text;
  -moz-user-select: text;
  -webkit-user-select: text;
  -ms-user-select: text;
}

.jp-OutputArea .jp-RenderedText {
  padding-left: 1ch;
}

/**
 * Prompt overlay.
 */

.jp-OutputArea-promptOverlay {
  position: absolute;
  top: 0;
  width: var(--jp-cell-prompt-width);
  height: 100%;
  opacity: 0.5;
}

.jp-OutputArea-promptOverlay:hover {
  background: var(--jp-layout-color2);
  box-shadow: inset 0 0 1px var(--jp-inverse-layout-color0);
  cursor: zoom-out;
}

.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay:hover {
  cursor: zoom-in;
}

/**
 * Isolated output.
 */
.jp-OutputArea-output.jp-mod-isolated {
  width: 100%;
  display: block;
}

/*
When drag events occur, `lm-mod-override-cursor` is added to the body.
Because iframes steal all cursor events, the following two rules are necessary
to suppress pointer events while resize drags are occurring. There may be a
better solution to this problem.
*/
body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated {
  position: relative;
}

body.lm-mod-override-cursor .jp-OutputArea-output.jp-mod-isolated::before {
  content: '';
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: transparent;
}

/* pre */

.jp-OutputArea-output pre {
  border: none;
  margin: 0;
  padding: 0;
  overflow-x: auto;
  overflow-y: auto;
  word-break: break-all;
  word-wrap: break-word;
  white-space: pre-wrap;
}

/* tables */

.jp-OutputArea-output.jp-RenderedHTMLCommon table {
  margin-left: 0;
  margin-right: 0;
}

/* description lists */

.jp-OutputArea-output dl,
.jp-OutputArea-output dt,
.jp-OutputArea-output dd {
  display: block;
}

.jp-OutputArea-output dl {
  width: 100%;
  overflow: hidden;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dt {
  font-weight: bold;
  float: left;
  width: 20%;
  padding: 0;
  margin: 0;
}

.jp-OutputArea-output dd {
  float: left;
  width: 80%;
  padding: 0;
  margin: 0;
}

.jp-TrimmedOutputs pre {
  background: var(--jp-layout-color3);
  font-size: calc(var(--jp-code-font-size) * 1.4);
  text-align: center;
  text-transform: uppercase;
}

/* Hide the gutter in case of
 *  - nested output areas (e.g. in the case of output widgets)
 *  - mirrored output areas
 */
.jp-OutputArea .jp-OutputArea .jp-OutputArea-prompt {
  display: none;
}

/* Hide empty lines in the output area, for instance due to cleared widgets */
.jp-OutputArea-prompt:empty {
  padding: 0;
  border: 0;
}

/*-----------------------------------------------------------------------------
| executeResult is added to any Output-result for the display of the object
| returned by a cell
|----------------------------------------------------------------------------*/

.jp-OutputArea-output.jp-OutputArea-executeResult {
  margin-left: 0;
  width: 100%;
}

/* Text output with the Out[] prompt needs a top padding to match the
 * alignment of the Out[] prompt itself.
 */
.jp-OutputArea-executeResult .jp-RenderedText.jp-OutputArea-output {
  padding-top: var(--jp-code-padding);
  border-top: var(--jp-border-width) solid transparent;
}

/*-----------------------------------------------------------------------------
| The Stdin output
|----------------------------------------------------------------------------*/

.jp-Stdin-prompt {
  color: var(--jp-content-font-color0);
  padding-right: var(--jp-code-padding);
  vertical-align: baseline;
  flex: 0 0 auto;
}

.jp-Stdin-input {
  font-family: var(--jp-code-font-family);
  font-size: inherit;
  color: inherit;
  background-color: inherit;
  width: 42%;
  min-width: 200px;

  /* make sure input baseline aligns with prompt */
  vertical-align: baseline;

  /* padding + margin = 0.5em between prompt and cursor */
  padding: 0 0.25em;
  margin: 0 0.25em;
  flex: 0 0 70%;
}

.jp-Stdin-input::placeholder {
  opacity: 0;
}

.jp-Stdin-input:focus {
  box-shadow: none;
}

.jp-Stdin-input:focus::placeholder {
  opacity: 1;
}

/*-----------------------------------------------------------------------------
| Output Area View
|----------------------------------------------------------------------------*/

.jp-LinkedOutputView .jp-OutputArea {
  height: 100%;
  display: block;
}

.jp-LinkedOutputView .jp-OutputArea-output:only-child {
  height: 100%;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

@media print {
  .jp-OutputArea-child {
    break-inside: avoid-page;
  }
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (max-width: 760px) {
  .jp-OutputPrompt {
    display: table-row;
    text-align: left;
  }

  .jp-OutputArea-child .jp-OutputArea-output {
    display: table-row;
    margin-left: var(--jp-notebook-padding);
  }
}

/* Trimmed outputs warning */
.jp-TrimmedOutputs > a {
  margin: 10px;
  text-decoration: none;
  cursor: pointer;
}

.jp-TrimmedOutputs > a:hover {
  text-decoration: none;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Table of Contents
|----------------------------------------------------------------------------*/

:root {
  --jp-private-toc-active-width: 4px;
}

.jp-TableOfContents {
  display: flex;
  flex-direction: column;
  background: var(--jp-layout-color1);
  color: var(--jp-ui-font-color1);
  font-size: var(--jp-ui-font-size1);
  height: 100%;
}

.jp-TableOfContents-placeholder {
  text-align: center;
}

.jp-TableOfContents-placeholderContent {
  color: var(--jp-content-font-color2);
  padding: 8px;
}

.jp-TableOfContents-placeholderContent > h3 {
  margin-bottom: var(--jp-content-heading-margin-bottom);
}

.jp-TableOfContents .jp-SidePanel-content {
  overflow-y: auto;
}

.jp-TableOfContents-tree {
  margin: 4px;
}

.jp-TableOfContents ol {
  list-style-type: none;
}

/* stylelint-disable-next-line selector-max-type */
.jp-TableOfContents li > ol {
  /* Align left border with triangle icon center */
  padding-left: 11px;
}

.jp-TableOfContents-content {
  /* left margin for the active heading indicator */
  margin: 0 0 0 var(--jp-private-toc-active-width);
  padding: 0;
  background-color: var(--jp-layout-color1);
}

.jp-tocItem {
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.jp-tocItem-heading {
  display: flex;
  cursor: pointer;
}

.jp-tocItem-heading:hover {
  background-color: var(--jp-layout-color2);
}

.jp-tocItem-content {
  display: block;
  padding: 4px 0;
  white-space: nowrap;
  text-overflow: ellipsis;
  overflow-x: hidden;
}

.jp-tocItem-collapser {
  height: 20px;
  margin: 2px 2px 0;
  padding: 0;
  background: none;
  border: none;
  cursor: pointer;
}

.jp-tocItem-collapser:hover {
  background-color: var(--jp-layout-color3);
}

/* Active heading indicator */

.jp-tocItem-heading::before {
  content: ' ';
  background: transparent;
  width: var(--jp-private-toc-active-width);
  height: 24px;
  position: absolute;
  left: 0;
  border-radius: var(--jp-border-radius);
}

.jp-tocItem-heading.jp-tocItem-active::before {
  background-color: var(--jp-brand-color1);
}

.jp-tocItem-heading:hover.jp-tocItem-active::before {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

.jp-Collapser {
  flex: 0 0 var(--jp-cell-collapser-width);
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
  border-radius: var(--jp-border-radius);
  opacity: 1;
}

.jp-Collapser-child {
  display: block;
  width: 100%;
  box-sizing: border-box;

  /* height: 100% doesn't work because the height of its parent is computed from content */
  position: absolute;
  top: 0;
  bottom: 0;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Hiding collapsers in print mode.

Note: input and output wrappers have "display: block" propery in print mode.
*/

@media print {
  .jp-Collapser {
    display: none;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Header/Footer
|----------------------------------------------------------------------------*/

/* Hidden by zero height by default */
.jp-CellHeader,
.jp-CellFooter {
  height: 0;
  width: 100%;
  padding: 0;
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Input
|----------------------------------------------------------------------------*/

/* All input areas */
.jp-InputArea {
  display: table;
  table-layout: fixed;
  width: 100%;
  overflow: hidden;
}

.jp-InputArea-editor {
  display: table-cell;
  overflow: hidden;
  vertical-align: top;

  /* This is the non-active, default styling */
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  background: var(--jp-cell-editor-background);
}

.jp-InputPrompt {
  display: table-cell;
  vertical-align: top;
  width: var(--jp-cell-prompt-width);
  color: var(--jp-cell-inprompt-font-color);
  font-family: var(--jp-cell-prompt-font-family);
  padding: var(--jp-code-padding);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  opacity: var(--jp-cell-prompt-opacity);
  line-height: var(--jp-code-line-height);
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;

  /* Right align prompt text, don't wrap to handle large prompt numbers */
  text-align: right;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;

  /* Disable text selection */
  -webkit-user-select: none;
  -moz-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

/*-----------------------------------------------------------------------------
| Mobile
|----------------------------------------------------------------------------*/
@media only screen and (max-width: 760px) {
  .jp-InputArea-editor {
    display: table-row;
    margin-left: var(--jp-notebook-padding);
  }

  .jp-InputPrompt {
    display: table-row;
    text-align: left;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Placeholder {
  display: table;
  table-layout: fixed;
  width: 100%;
}

.jp-Placeholder-prompt {
  display: table-cell;
  box-sizing: border-box;
}

.jp-Placeholder-content {
  display: table-cell;
  padding: 4px 6px;
  border: 1px solid transparent;
  border-radius: 0;
  background: none;
  box-sizing: border-box;
  cursor: pointer;
}

.jp-Placeholder-contentContainer {
  display: flex;
}

.jp-Placeholder-content:hover,
.jp-InputPlaceholder > .jp-Placeholder-content:hover {
  border-color: var(--jp-layout-color3);
}

.jp-Placeholder-content .jp-MoreHorizIcon {
  width: 32px;
  height: 16px;
  border: 1px solid transparent;
  border-radius: var(--jp-border-radius);
}

.jp-Placeholder-content .jp-MoreHorizIcon:hover {
  border: 1px solid var(--jp-border-color1);
  box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.25);
  background-color: var(--jp-layout-color0);
}

.jp-PlaceholderText {
  white-space: nowrap;
  overflow-x: hidden;
  color: var(--jp-inverse-layout-color3);
  font-family: var(--jp-code-font-family);
}

.jp-InputPlaceholder > .jp-Placeholder-content {
  border-color: var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Private CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-private-cell-scrolling-output-offset: 5px;
}

/*-----------------------------------------------------------------------------
| Cell
|----------------------------------------------------------------------------*/

.jp-Cell {
  padding: var(--jp-cell-padding);
  margin: 0;
  border: none;
  outline: none;
  background: transparent;
}

/*-----------------------------------------------------------------------------
| Common input/output
|----------------------------------------------------------------------------*/

.jp-Cell-inputWrapper,
.jp-Cell-outputWrapper {
  display: flex;
  flex-direction: row;
  padding: 0;
  margin: 0;

  /* Added to reveal the box-shadow on the input and output collapsers. */
  overflow: visible;
}

/* Only input/output areas inside cells */
.jp-Cell-inputArea,
.jp-Cell-outputArea {
  flex: 1 1 auto;
}

/*-----------------------------------------------------------------------------
| Collapser
|----------------------------------------------------------------------------*/

/* Make the output collapser disappear when there is not output, but do so
 * in a manner that leaves it in the layout and preserves its width.
 */
.jp-Cell.jp-mod-noOutputs .jp-Cell-outputCollapser {
  border: none !important;
  background: transparent !important;
}

.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputCollapser {
  min-height: var(--jp-cell-collapser-min-height);
}

/*-----------------------------------------------------------------------------
| Output
|----------------------------------------------------------------------------*/

/* Put a space between input and output when there IS output */
.jp-Cell:not(.jp-mod-noOutputs) .jp-Cell-outputWrapper {
  margin-top: 5px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea {
  overflow-y: auto;
  max-height: 24em;
  margin-left: var(--jp-private-cell-scrolling-output-offset);
  resize: vertical;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea[style*='height'] {
  max-height: unset;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-Cell-outputArea::after {
  content: ' ';
  box-shadow: inset 0 0 6px 2px rgb(0 0 0 / 30%);
  width: 100%;
  height: 100%;
  position: sticky;
  bottom: 0;
  top: 0;
  margin-top: -50%;
  float: left;
  display: block;
  pointer-events: none;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-child {
  padding-top: 6px;
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-prompt {
  width: calc(
    var(--jp-cell-prompt-width) - var(--jp-private-cell-scrolling-output-offset)
  );
}

.jp-CodeCell.jp-mod-outputsScrolled .jp-OutputArea-promptOverlay {
  left: calc(-1 * var(--jp-private-cell-scrolling-output-offset));
}

/*-----------------------------------------------------------------------------
| CodeCell
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| MarkdownCell
|----------------------------------------------------------------------------*/

.jp-MarkdownOutput {
  display: table-cell;
  width: 100%;
  margin-top: 0;
  margin-bottom: 0;
  padding-left: var(--jp-code-padding);
}

.jp-MarkdownOutput.jp-RenderedHTMLCommon {
  overflow: auto;
}

/* collapseHeadingButton (show always if hiddenCellsButton is _not_ shown) */
.jp-collapseHeadingButton {
  display: flex;
  min-height: var(--jp-cell-collapser-min-height);
  font-size: var(--jp-code-font-size);
  position: absolute;
  background-color: transparent;
  background-size: 25px;
  background-repeat: no-repeat;
  background-position-x: center;
  background-position-y: top;
  background-image: var(--jp-icon-caret-down);
  right: 0;
  top: 0;
  bottom: 0;
}

.jp-collapseHeadingButton.jp-mod-collapsed {
  background-image: var(--jp-icon-caret-right);
}

/*
 set the container font size to match that of content
 so that the nested collapse buttons have the right size
*/
.jp-MarkdownCell .jp-InputPrompt {
  font-size: var(--jp-content-font-size1);
}

/*
  Align collapseHeadingButton with cell top header
  The font sizes are identical to the ones in packages/rendermime/style/base.css
*/
.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='1'] {
  font-size: var(--jp-content-font-size5);
  background-position-y: calc(0.3 * var(--jp-content-font-size5));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='2'] {
  font-size: var(--jp-content-font-size4);
  background-position-y: calc(0.3 * var(--jp-content-font-size4));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='3'] {
  font-size: var(--jp-content-font-size3);
  background-position-y: calc(0.3 * var(--jp-content-font-size3));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='4'] {
  font-size: var(--jp-content-font-size2);
  background-position-y: calc(0.3 * var(--jp-content-font-size2));
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='5'] {
  font-size: var(--jp-content-font-size1);
  background-position-y: top;
}

.jp-mod-rendered .jp-collapseHeadingButton[data-heading-level='6'] {
  font-size: var(--jp-content-font-size0);
  background-position-y: top;
}

/* collapseHeadingButton (show only on (hover,active) if hiddenCellsButton is shown) */
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-collapseHeadingButton {
  display: none;
}

.jp-Notebook.jp-mod-showHiddenCellsButton
  :is(.jp-MarkdownCell:hover, .jp-mod-active)
  .jp-collapseHeadingButton {
  display: flex;
}

/* showHiddenCellsButton (only show if jp-mod-showHiddenCellsButton is set, which
is a consequence of the showHiddenCellsButton option in Notebook Settings)*/
.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton {
  margin-left: calc(var(--jp-cell-prompt-width) + 2 * var(--jp-code-padding));
  margin-top: var(--jp-code-padding);
  border: 1px solid var(--jp-border-color2);
  background-color: var(--jp-border-color3) !important;
  color: var(--jp-content-font-color0) !important;
  display: flex;
}

.jp-Notebook.jp-mod-showHiddenCellsButton .jp-showHiddenCellsButton:hover {
  background-color: var(--jp-border-color2) !important;
}

.jp-showHiddenCellsButton {
  display: none;
}

/*-----------------------------------------------------------------------------
| Printing
|----------------------------------------------------------------------------*/

/*
Using block instead of flex to allow the use of the break-inside CSS property for
cell outputs.
*/

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

:root {
  --jp-notebook-toolbar-padding: 2px 5px 2px 2px;
}

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-NotebookPanel-toolbar {
  padding: var(--jp-notebook-toolbar-padding);

  /* disable paint containment from lumino 2.0 default strict CSS containment */
  contain: style size !important;
}

.jp-Toolbar-item.jp-Notebook-toolbarCellType .jp-select-wrapper.jp-mod-focused {
  border: none;
  box-shadow: none;
}

.jp-Notebook-toolbarCellTypeDropdown select {
  height: 24px;
  font-size: var(--jp-ui-font-size1);
  line-height: 14px;
  border-radius: 0;
  display: block;
}

.jp-Notebook-toolbarCellTypeDropdown span {
  top: 5px !important;
}

.jp-Toolbar-responsive-popup {
  position: absolute;
  height: fit-content;
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
  justify-content: flex-end;
  border-bottom: var(--jp-border-width) solid var(--jp-toolbar-border-color);
  box-shadow: var(--jp-toolbar-box-shadow);
  background: var(--jp-toolbar-background);
  min-height: var(--jp-toolbar-micro-height);
  padding: var(--jp-notebook-toolbar-padding);
  z-index: 1;
  right: 0;
  top: 0;
}

.jp-Toolbar > .jp-Toolbar-responsive-opener {
  margin-left: auto;
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Variables
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------

/*-----------------------------------------------------------------------------
| Styles
|----------------------------------------------------------------------------*/

.jp-Notebook-ExecutionIndicator {
  position: relative;
  display: inline-block;
  height: 100%;
  z-index: 9997;
}

.jp-Notebook-ExecutionIndicator-tooltip {
  visibility: hidden;
  height: auto;
  width: max-content;
  width: -moz-max-content;
  background-color: var(--jp-layout-color2);
  color: var(--jp-ui-font-color1);
  text-align: justify;
  border-radius: 6px;
  padding: 0 5px;
  position: fixed;
  display: table;
}

.jp-Notebook-ExecutionIndicator-tooltip.up {
  transform: translateX(-50%) translateY(-100%) translateY(-32px);
}

.jp-Notebook-ExecutionIndicator-tooltip.down {
  transform: translateX(calc(-100% + 16px)) translateY(5px);
}

.jp-Notebook-ExecutionIndicator-tooltip.hidden {
  display: none;
}

.jp-Notebook-ExecutionIndicator:hover .jp-Notebook-ExecutionIndicator-tooltip {
  visibility: visible;
}

.jp-Notebook-ExecutionIndicator span {
  font-size: var(--jp-ui-font-size1);
  font-family: var(--jp-ui-font-family);
  color: var(--jp-ui-font-color1);
  line-height: 24px;
  display: block;
}

.jp-Notebook-ExecutionIndicator-progress-bar {
  display: flex;
  justify-content: center;
  height: 100%;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

/*
 * Execution indicator
 */
.jp-tocItem-content::after {
  content: '';

  /* Must be identical to form a circle */
  width: 12px;
  height: 12px;
  background: none;
  border: none;
  position: absolute;
  right: 0;
}

.jp-tocItem-content[data-running='0']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background: none;
}

.jp-tocItem-content[data-running='1']::after {
  border-radius: 50%;
  border: var(--jp-border-width) solid var(--jp-inverse-layout-color3);
  background-color: var(--jp-inverse-layout-color3);
}

.jp-tocItem-content[data-running='0'],
.jp-tocItem-content[data-running='1'] {
  margin-right: 12px;
}

/*
 * Copyright (c) Jupyter Development Team.
 * Distributed under the terms of the Modified BSD License.
 */

.jp-Notebook-footer {
  height: 27px;
  margin-left: calc(
    var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
      var(--jp-cell-padding)
  );
  width: calc(
    100% -
      (
        var(--jp-cell-prompt-width) + var(--jp-cell-collapser-width) +
          var(--jp-cell-padding) + var(--jp-cell-padding)
      )
  );
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  color: var(--jp-ui-font-color3);
  margin-top: 6px;
  background: none;
  cursor: pointer;
}

.jp-Notebook-footer:focus {
  border-color: var(--jp-cell-editor-active-border-color);
}

/* For devices that support hovering, hide footer until hover */
@media (hover: hover) {
  .jp-Notebook-footer {
    opacity: 0;
  }

  .jp-Notebook-footer:focus,
  .jp-Notebook-footer:hover {
    opacity: 1;
  }
}

/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| Imports
|----------------------------------------------------------------------------*/

/*-----------------------------------------------------------------------------
| CSS variables
|----------------------------------------------------------------------------*/

:root {
  --jp-side-by-side-output-size: 1fr;
  --jp-side-by-side-resized-cell: var(--jp-side-by-side-output-size);
  --jp-private-notebook-dragImage-width: 304px;
  --jp-private-notebook-dragImage-height: 36px;
  --jp-private-notebook-selected-color: var(--md-blue-400);
  --jp-private-notebook-active-color: var(--md-green-400);
}

/*-----------------------------------------------------------------------------
| Notebook
|----------------------------------------------------------------------------*/

/* stylelint-disable selector-max-class */

.jp-NotebookPanel {
  display: block;
  height: 100%;
}

.jp-NotebookPanel.jp-Document {
  min-width: 240px;
  min-height: 120px;
}

.jp-Notebook {
  padding: var(--jp-notebook-padding);
  outline: none;
  overflow: auto;
  background: var(--jp-layout-color0);
}

.jp-Notebook.jp-mod-scrollPastEnd::after {
  display: block;
  content: '';
  min-height: var(--jp-notebook-scroll-padding);
}

.jp-MainAreaWidget-ContainStrict .jp-Notebook * {
  contain: strict;
}

.jp-Notebook .jp-Cell {
  overflow: visible;
}

.jp-Notebook .jp-Cell .jp-InputPrompt {
  cursor: move;
}

/*-----------------------------------------------------------------------------
| Notebook state related styling
|
| The notebook and cells each have states, here are the possibilities:
|
| - Notebook
|   - Command
|   - Edit
| - Cell
|   - None
|   - Active (only one can be active)
|   - Selected (the cells actions are applied to)
|   - Multiselected (when multiple selected, the cursor)
|   - No outputs
|----------------------------------------------------------------------------*/

/* Command or edit modes */

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-InputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

.jp-Notebook .jp-Cell:not(.jp-mod-active) .jp-OutputPrompt {
  opacity: var(--jp-cell-prompt-not-active-opacity);
  color: var(--jp-cell-prompt-not-active-font-color);
}

/* cell is active */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser {
  background: var(--jp-brand-color1);
}

/* cell is dirty */
.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt {
  color: var(--jp-warn-color1);
}

.jp-Notebook .jp-Cell.jp-mod-dirty .jp-InputPrompt::before {
  color: var(--jp-warn-color1);
  content: '•';
}

.jp-Notebook .jp-Cell.jp-mod-active.jp-mod-dirty .jp-Collapser {
  background: var(--jp-warn-color1);
}

/* collapser is hovered */
.jp-Notebook .jp-Cell .jp-Collapser:hover {
  box-shadow: var(--jp-elevation-z2);
  background: var(--jp-brand-color1);
  opacity: var(--jp-cell-collapser-not-active-hover-opacity);
}

/* cell is active and collapser is hovered */
.jp-Notebook .jp-Cell.jp-mod-active .jp-Collapser:hover {
  background: var(--jp-brand-color0);
  opacity: 1;
}

/* Command mode */

.jp-Notebook.jp-mod-commandMode .jp-Cell.jp-mod-selected {
  background: var(--jp-notebook-multiselected-color);
}

.jp-Notebook.jp-mod-commandMode
  .jp-Cell.jp-mod-active.jp-mod-selected:not(.jp-mod-multiSelected) {
  background: transparent;
}

/* Edit mode */

.jp-Notebook.jp-mod-editMode .jp-Cell.jp-mod-active .jp-InputArea-editor {
  border: var(--jp-border-width) solid var(--jp-cell-editor-active-border-color);
  box-shadow: var(--jp-input-box-shadow);
  background-color: var(--jp-cell-editor-active-background);
}

/*-----------------------------------------------------------------------------
| Notebook drag and drop
|----------------------------------------------------------------------------*/

.jp-Notebook-cell.jp-mod-dropSource {
  opacity: 0.5;
}

.jp-Notebook-cell.jp-mod-dropTarget,
.jp-Notebook.jp-mod-commandMode
  .jp-Notebook-cell.jp-mod-active.jp-mod-selected.jp-mod-dropTarget {
  border-top-color: var(--jp-private-notebook-selected-color);
  border-top-style: solid;
  border-top-width: 2px;
}

.jp-dragImage {
  display: block;
  flex-direction: row;
  width: var(--jp-private-notebook-dragImage-width);
  height: var(--jp-private-notebook-dragImage-height);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background);
  overflow: visible;
}

.jp-dragImage-singlePrompt {
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

.jp-dragImage .jp-dragImage-content {
  flex: 1 1 auto;
  z-index: 2;
  font-size: var(--jp-code-font-size);
  font-family: var(--jp-code-font-family);
  line-height: var(--jp-code-line-height);
  padding: var(--jp-code-padding);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  background: var(--jp-cell-editor-background-color);
  color: var(--jp-content-font-color3);
  text-align: left;
  margin: 4px 4px 4px 0;
}

.jp-dragImage .jp-dragImage-prompt {
  flex: 0 0 auto;
  min-width: 36px;
  color: var(--jp-cell-inprompt-font-color);
  padding: var(--jp-code-padding);
  padding-left: 12px;
  font-family: var(--jp-cell-prompt-font-family);
  letter-spacing: var(--jp-cell-prompt-letter-spacing);
  line-height: 1.9;
  font-size: var(--jp-code-font-size);
  border: var(--jp-border-width) solid transparent;
}

.jp-dragImage-multipleBack {
  z-index: -1;
  position: absolute;
  height: 32px;
  width: 300px;
  top: 8px;
  left: 8px;
  background: var(--jp-layout-color2);
  border: var(--jp-border-width) solid var(--jp-input-border-color);
  box-shadow: 2px 2px 4px 0 rgba(0, 0, 0, 0.12);
}

/*-----------------------------------------------------------------------------
| Cell toolbar
|----------------------------------------------------------------------------*/

.jp-NotebookTools {
  display: block;
  min-width: var(--jp-sidebar-min-width);
  color: var(--jp-ui-font-color1);
  background: var(--jp-layout-color1);

  /* This is needed so that all font sizing of children done in ems is
    * relative to this base size */
  font-size: var(--jp-ui-font-size1);
  overflow: auto;
}

.jp-ActiveCellTool {
  padding: 12px 0;
  display: flex;
}

.jp-ActiveCellTool-Content {
  flex: 1 1 auto;
}

.jp-ActiveCellTool .jp-ActiveCellTool-CellContent {
  background: var(--jp-cell-editor-background);
  border: var(--jp-border-width) solid var(--jp-cell-editor-border-color);
  border-radius: 0;
  min-height: 29px;
}

.jp-ActiveCellTool .jp-InputPrompt {
  min-width: calc(var(--jp-cell-prompt-width) * 0.75);
}

.jp-ActiveCellTool-CellContent > pre {
  padding: 5px 4px;
  margin: 0;
  white-space: normal;
}

.jp-MetadataEditorTool {
  flex-direction: column;
  padding: 12px 0;
}

.jp-RankedPanel > :not(:first-child) {
  margin-top: 12px;
}

.jp-KeySelector select.jp-mod-styled {
  font-size: var(--jp-ui-font-size1);
  color: var(--jp-ui-font-color0);
  border: var(--jp-border-width) solid var(--jp-border-color1);
}

.jp-KeySelector label,
.jp-MetadataEditorTool label,
.jp-NumberSetter label {
  line-height: 1.4;
}

.jp-NotebookTools .jp-select-wrapper {
  margin-top: 4px;
  margin-bottom: 0;
}

.jp-NumberSetter input {
  width: 100%;
  margin-top: 4px;
}

.jp-NotebookTools .jp-Collapse {
  margin-top: 16px;
}

/*-----------------------------------------------------------------------------
| Presentation Mode (.jp-mod-presentationMode)
|----------------------------------------------------------------------------*/

.jp-mod-presentationMode .jp-Notebook {
  --jp-content-font-size1: var(--jp-content-presentation-font-size1);
  --jp-code-font-size: var(--jp-code-presentation-font-size);
}

.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-InputPrompt,
.jp-mod-presentationMode .jp-Notebook .jp-Cell .jp-OutputPrompt {
  flex: 0 0 110px;
}

/*-----------------------------------------------------------------------------
| Side-by-side Mode (.jp-mod-sideBySide)
|----------------------------------------------------------------------------*/
.jp-mod-sideBySide.jp-Notebook .jp-Notebook-cell {
  margin-top: 3em;
  margin-bottom: 3em;
  margin-left: 5%;
  margin-right: 5%;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell {
  display: grid;
  grid-template-columns: minmax(0, 1fr) min-content minmax(
      0,
      var(--jp-side-by-side-output-size)
    );
  grid-template-rows: auto minmax(0, 1fr) auto;
  grid-template-areas:
    'header header header'
    'input handle output'
    'footer footer footer';
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell.jp-mod-resizedCell {
  grid-template-columns: minmax(0, 1fr) min-content minmax(
      0,
      var(--jp-side-by-side-resized-cell)
    );
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellHeader {
  grid-area: header;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-inputWrapper {
  grid-area: input;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-Cell-outputWrapper {
  /* overwrite the default margin (no vertical separation needed in side by side move */
  margin-top: 0;
  grid-area: output;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellFooter {
  grid-area: footer;
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle {
  grid-area: handle;
  user-select: none;
  display: block;
  height: 100%;
  cursor: ew-resize;
  padding: 0 var(--jp-cell-padding);
}

.jp-mod-sideBySide.jp-Notebook .jp-CodeCell .jp-CellResizeHandle::after {
  content: '';
  display: block;
  background: var(--jp-border-color2);
  height: 100%;
  width: 5px;
}

.jp-mod-sideBySide.jp-Notebook
  .jp-CodeCell.jp-mod-resizedCell
  .jp-CellResizeHandle::after {
  background: var(--jp-border-color0);
}

.jp-CellResizeHandle {
  display: none;
}

/*-----------------------------------------------------------------------------
| Placeholder
|----------------------------------------------------------------------------*/

.jp-Cell-Placeholder {
  padding-left: 55px;
}

.jp-Cell-Placeholder-wrapper {
  background: #fff;
  border: 1px solid;
  border-color: #e5e6e9 #dfe0e4 #d0d1d5;
  border-radius: 4px;
  -webkit-border-radius: 4px;
  margin: 10px 15px;
}

.jp-Cell-Placeholder-wrapper-inner {
  padding: 15px;
  position: relative;
}

.jp-Cell-Placeholder-wrapper-body {
  background-repeat: repeat;
  background-size: 50% auto;
}

.jp-Cell-Placeholder-wrapper-body div {
  background: #f6f7f8;
  background-image: -webkit-linear-gradient(
    left,
    #f6f7f8 0%,
    #edeef1 20%,
    #f6f7f8 40%,
    #f6f7f8 100%
  );
  background-repeat: no-repeat;
  background-size: 800px 104px;
  height: 104px;
  position: absolute;
  right: 15px;
  left: 15px;
  top: 15px;
}

div.jp-Cell-Placeholder-h1 {
  top: 20px;
  height: 20px;
  left: 15px;
  width: 150px;
}

div.jp-Cell-Placeholder-h2 {
  left: 15px;
  top: 50px;
  height: 10px;
  width: 100px;
}

div.jp-Cell-Placeholder-content-1,
div.jp-Cell-Placeholder-content-2,
div.jp-Cell-Placeholder-content-3 {
  left: 15px;
  right: 15px;
  height: 10px;
}

div.jp-Cell-Placeholder-content-1 {
  top: 100px;
}

div.jp-Cell-Placeholder-content-2 {
  top: 120px;
}

div.jp-Cell-Placeholder-content-3 {
  top: 140px;
}

</style>
<style type="text/css">
/*-----------------------------------------------------------------------------
| Copyright (c) Jupyter Development Team.
| Distributed under the terms of the Modified BSD License.
|----------------------------------------------------------------------------*/

/*
The following CSS variables define the main, public API for styling JupyterLab.
These variables should be used by all plugins wherever possible. In other
words, plugins should not define custom colors, sizes, etc unless absolutely
necessary. This enables users to change the visual theme of JupyterLab
by changing these variables.

Many variables appear in an ordered sequence (0,1,2,3). These sequences
are designed to work well together, so for example, `--jp-border-color1` should
be used with `--jp-layout-color1`. The numbers have the following meanings:

* 0: super-primary, reserved for special emphasis
* 1: primary, most important under normal situations
* 2: secondary, next most important under normal situations
* 3: tertiary, next most important under normal situations

Throughout JupyterLab, we are mostly following principles from Google's
Material Design when selecting colors. We are not, however, following
all of MD as it is not optimized for dense, information rich UIs.
*/

:root {
  /* Elevation
   *
   * We style box-shadows using Material Design's idea of elevation. These particular numbers are taken from here:
   *
   * https://github.com/material-components/material-components-web
   * https://material-components-web.appspot.com/elevation.html
   */

  --jp-shadow-base-lightness: 0;
  --jp-shadow-umbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.2
  );
  --jp-shadow-penumbra-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.14
  );
  --jp-shadow-ambient-color: rgba(
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    var(--jp-shadow-base-lightness),
    0.12
  );
  --jp-elevation-z0: none;
  --jp-elevation-z1: 0 2px 1px -1px var(--jp-shadow-umbra-color),
    0 1px 1px 0 var(--jp-shadow-penumbra-color),
    0 1px 3px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z2: 0 3px 1px -2px var(--jp-shadow-umbra-color),
    0 2px 2px 0 var(--jp-shadow-penumbra-color),
    0 1px 5px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z4: 0 2px 4px -1px var(--jp-shadow-umbra-color),
    0 4px 5px 0 var(--jp-shadow-penumbra-color),
    0 1px 10px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z6: 0 3px 5px -1px var(--jp-shadow-umbra-color),
    0 6px 10px 0 var(--jp-shadow-penumbra-color),
    0 1px 18px 0 var(--jp-shadow-ambient-color);
  --jp-elevation-z8: 0 5px 5px -3px var(--jp-shadow-umbra-color),
    0 8px 10px 1px var(--jp-shadow-penumbra-color),
    0 3px 14px 2px var(--jp-shadow-ambient-color);
  --jp-elevation-z12: 0 7px 8px -4px var(--jp-shadow-umbra-color),
    0 12px 17px 2px var(--jp-shadow-penumbra-color),
    0 5px 22px 4px var(--jp-shadow-ambient-color);
  --jp-elevation-z16: 0 8px 10px -5px var(--jp-shadow-umbra-color),
    0 16px 24px 2px var(--jp-shadow-penumbra-color),
    0 6px 30px 5px var(--jp-shadow-ambient-color);
  --jp-elevation-z20: 0 10px 13px -6px var(--jp-shadow-umbra-color),
    0 20px 31px 3px var(--jp-shadow-penumbra-color),
    0 8px 38px 7px var(--jp-shadow-ambient-color);
  --jp-elevation-z24: 0 11px 15px -7px var(--jp-shadow-umbra-color),
    0 24px 38px 3px var(--jp-shadow-penumbra-color),
    0 9px 46px 8px var(--jp-shadow-ambient-color);

  /* Borders
   *
   * The following variables, specify the visual styling of borders in JupyterLab.
   */

  --jp-border-width: 1px;
  --jp-border-color0: var(--md-grey-400);
  --jp-border-color1: var(--md-grey-400);
  --jp-border-color2: var(--md-grey-300);
  --jp-border-color3: var(--md-grey-200);
  --jp-inverse-border-color: var(--md-grey-600);
  --jp-border-radius: 2px;

  /* UI Fonts
   *
   * The UI font CSS variables are used for the typography all of the JupyterLab
   * user interface elements that are not directly user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-ui-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-ui-font-scale-factor: 1.2;
  --jp-ui-font-size0: 0.83333em;
  --jp-ui-font-size1: 13px; /* Base font size */
  --jp-ui-font-size2: 1.2em;
  --jp-ui-font-size3: 1.44em;
  --jp-ui-font-family: system-ui, -apple-system, blinkmacsystemfont, 'Segoe UI',
    helvetica, arial, sans-serif, 'Apple Color Emoji', 'Segoe UI Emoji',
    'Segoe UI Symbol';

  /*
   * Use these font colors against the corresponding main layout colors.
   * In a light theme, these go from dark to light.
   */

  /* Defaults use Material Design specification */
  --jp-ui-font-color0: rgba(0, 0, 0, 1);
  --jp-ui-font-color1: rgba(0, 0, 0, 0.87);
  --jp-ui-font-color2: rgba(0, 0, 0, 0.54);
  --jp-ui-font-color3: rgba(0, 0, 0, 0.38);

  /*
   * Use these against the brand/accent/warn/error colors.
   * These will typically go from light to darker, in both a dark and light theme.
   */

  --jp-ui-inverse-font-color0: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color1: rgba(255, 255, 255, 1);
  --jp-ui-inverse-font-color2: rgba(255, 255, 255, 0.7);
  --jp-ui-inverse-font-color3: rgba(255, 255, 255, 0.5);

  /* Content Fonts
   *
   * Content font variables are used for typography of user generated content.
   *
   * The font sizing here is done assuming that the body font size of --jp-content-font-size1
   * is applied to a parent element. When children elements, such as headings, are sized
   * in em all things will be computed relative to that body size.
   */

  --jp-content-line-height: 1.6;
  --jp-content-font-scale-factor: 1.2;
  --jp-content-font-size0: 0.83333em;
  --jp-content-font-size1: 14px; /* Base font size */
  --jp-content-font-size2: 1.2em;
  --jp-content-font-size3: 1.44em;
  --jp-content-font-size4: 1.728em;
  --jp-content-font-size5: 2.0736em;

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-content-presentation-font-size1: 17px;
  --jp-content-heading-line-height: 1;
  --jp-content-heading-margin-top: 1.2em;
  --jp-content-heading-margin-bottom: 0.8em;
  --jp-content-heading-font-weight: 500;

  /* Defaults use Material Design specification */
  --jp-content-font-color0: rgba(0, 0, 0, 1);
  --jp-content-font-color1: rgba(0, 0, 0, 0.87);
  --jp-content-font-color2: rgba(0, 0, 0, 0.54);
  --jp-content-font-color3: rgba(0, 0, 0, 0.38);
  --jp-content-link-color: var(--md-blue-900);
  --jp-content-font-family: system-ui, -apple-system, blinkmacsystemfont,
    'Segoe UI', helvetica, arial, sans-serif, 'Apple Color Emoji',
    'Segoe UI Emoji', 'Segoe UI Symbol';

  /*
   * Code Fonts
   *
   * Code font variables are used for typography of code and other monospaces content.
   */

  --jp-code-font-size: 13px;
  --jp-code-line-height: 1.3077; /* 17px for 13px base */
  --jp-code-padding: 5px; /* 5px for 13px base, codemirror highlighting needs integer px value */
  --jp-code-font-family-default: menlo, consolas, 'DejaVu Sans Mono', monospace;
  --jp-code-font-family: var(--jp-code-font-family-default);

  /* This gives a magnification of about 125% in presentation mode over normal. */
  --jp-code-presentation-font-size: 16px;

  /* may need to tweak cursor width if you change font size */
  --jp-code-cursor-width0: 1.4px;
  --jp-code-cursor-width1: 2px;
  --jp-code-cursor-width2: 4px;

  /* Layout
   *
   * The following are the main layout colors use in JupyterLab. In a light
   * theme these would go from light to dark.
   */

  --jp-layout-color0: white;
  --jp-layout-color1: white;
  --jp-layout-color2: var(--md-grey-200);
  --jp-layout-color3: var(--md-grey-400);
  --jp-layout-color4: var(--md-grey-600);

  /* Inverse Layout
   *
   * The following are the inverse layout colors use in JupyterLab. In a light
   * theme these would go from dark to light.
   */

  --jp-inverse-layout-color0: #111;
  --jp-inverse-layout-color1: var(--md-grey-900);
  --jp-inverse-layout-color2: var(--md-grey-800);
  --jp-inverse-layout-color3: var(--md-grey-700);
  --jp-inverse-layout-color4: var(--md-grey-600);

  /* Brand/accent */

  --jp-brand-color0: var(--md-blue-900);
  --jp-brand-color1: var(--md-blue-700);
  --jp-brand-color2: var(--md-blue-300);
  --jp-brand-color3: var(--md-blue-100);
  --jp-brand-color4: var(--md-blue-50);
  --jp-accent-color0: var(--md-green-900);
  --jp-accent-color1: var(--md-green-700);
  --jp-accent-color2: var(--md-green-300);
  --jp-accent-color3: var(--md-green-100);

  /* State colors (warn, error, success, info) */

  --jp-warn-color0: var(--md-orange-900);
  --jp-warn-color1: var(--md-orange-700);
  --jp-warn-color2: var(--md-orange-300);
  --jp-warn-color3: var(--md-orange-100);
  --jp-error-color0: var(--md-red-900);
  --jp-error-color1: var(--md-red-700);
  --jp-error-color2: var(--md-red-300);
  --jp-error-color3: var(--md-red-100);
  --jp-success-color0: var(--md-green-900);
  --jp-success-color1: var(--md-green-700);
  --jp-success-color2: var(--md-green-300);
  --jp-success-color3: var(--md-green-100);
  --jp-info-color0: var(--md-cyan-900);
  --jp-info-color1: var(--md-cyan-700);
  --jp-info-color2: var(--md-cyan-300);
  --jp-info-color3: var(--md-cyan-100);

  /* Cell specific styles */

  --jp-cell-padding: 5px;
  --jp-cell-collapser-width: 8px;
  --jp-cell-collapser-min-height: 20px;
  --jp-cell-collapser-not-active-hover-opacity: 0.6;
  --jp-cell-editor-background: var(--md-grey-100);
  --jp-cell-editor-border-color: var(--md-grey-300);
  --jp-cell-editor-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-cell-editor-active-background: var(--jp-layout-color0);
  --jp-cell-editor-active-border-color: var(--jp-brand-color1);
  --jp-cell-prompt-width: 64px;
  --jp-cell-prompt-font-family: var(--jp-code-font-family-default);
  --jp-cell-prompt-letter-spacing: 0;
  --jp-cell-prompt-opacity: 1;
  --jp-cell-prompt-not-active-opacity: 0.5;
  --jp-cell-prompt-not-active-font-color: var(--md-grey-700);

  /* A custom blend of MD grey and blue 600
   * See https://meyerweb.com/eric/tools/color-blend/#546E7A:1E88E5:5:hex */
  --jp-cell-inprompt-font-color: #307fc1;

  /* A custom blend of MD grey and orange 600
   * https://meyerweb.com/eric/tools/color-blend/#546E7A:F4511E:5:hex */
  --jp-cell-outprompt-font-color: #bf5b3d;

  /* Notebook specific styles */

  --jp-notebook-padding: 10px;
  --jp-notebook-select-background: var(--jp-layout-color1);
  --jp-notebook-multiselected-color: var(--md-blue-50);

  /* The scroll padding is calculated to fill enough space at the bottom of the
  notebook to show one single-line cell (with appropriate padding) at the top
  when the notebook is scrolled all the way to the bottom. We also subtract one
  pixel so that no scrollbar appears if we have just one single-line cell in the
  notebook. This padding is to enable a 'scroll past end' feature in a notebook.
  */
  --jp-notebook-scroll-padding: calc(
    100% - var(--jp-code-font-size) * var(--jp-code-line-height) -
      var(--jp-code-padding) - var(--jp-cell-padding) - 1px
  );

  /* Rendermime styles */

  --jp-rendermime-error-background: #fdd;
  --jp-rendermime-table-row-background: var(--md-grey-100);
  --jp-rendermime-table-row-hover-background: var(--md-light-blue-50);

  /* Dialog specific styles */

  --jp-dialog-background: rgba(0, 0, 0, 0.25);

  /* Console specific styles */

  --jp-console-padding: 10px;

  /* Toolbar specific styles */

  --jp-toolbar-border-color: var(--jp-border-color1);
  --jp-toolbar-micro-height: 8px;
  --jp-toolbar-background: var(--jp-layout-color1);
  --jp-toolbar-box-shadow: 0 0 2px 0 rgba(0, 0, 0, 0.24);
  --jp-toolbar-header-margin: 4px 4px 0 4px;
  --jp-toolbar-active-background: var(--md-grey-300);

  /* Statusbar specific styles */

  --jp-statusbar-height: 24px;

  /* Input field styles */

  --jp-input-box-shadow: inset 0 0 2px var(--md-blue-300);
  --jp-input-active-background: var(--jp-layout-color1);
  --jp-input-hover-background: var(--jp-layout-color1);
  --jp-input-background: var(--md-grey-100);
  --jp-input-border-color: var(--jp-inverse-border-color);
  --jp-input-active-border-color: var(--jp-brand-color1);
  --jp-input-active-box-shadow-color: rgba(19, 124, 189, 0.3);

  /* General editor styles */

  --jp-editor-selected-background: #d9d9d9;
  --jp-editor-selected-focused-background: #d7d4f0;
  --jp-editor-cursor-color: var(--jp-ui-font-color0);

  /* Code mirror specific styles */

  --jp-mirror-editor-keyword-color: #008000;
  --jp-mirror-editor-atom-color: #88f;
  --jp-mirror-editor-number-color: #080;
  --jp-mirror-editor-def-color: #00f;
  --jp-mirror-editor-variable-color: var(--md-grey-900);
  --jp-mirror-editor-variable-2-color: rgb(0, 54, 109);
  --jp-mirror-editor-variable-3-color: #085;
  --jp-mirror-editor-punctuation-color: #05a;
  --jp-mirror-editor-property-color: #05a;
  --jp-mirror-editor-operator-color: #a2f;
  --jp-mirror-editor-comment-color: #408080;
  --jp-mirror-editor-string-color: #ba2121;
  --jp-mirror-editor-string-2-color: #708;
  --jp-mirror-editor-meta-color: #a2f;
  --jp-mirror-editor-qualifier-color: #555;
  --jp-mirror-editor-builtin-color: #008000;
  --jp-mirror-editor-bracket-color: #997;
  --jp-mirror-editor-tag-color: #170;
  --jp-mirror-editor-attribute-color: #00c;
  --jp-mirror-editor-header-color: blue;
  --jp-mirror-editor-quote-color: #090;
  --jp-mirror-editor-link-color: #00c;
  --jp-mirror-editor-error-color: #f00;
  --jp-mirror-editor-hr-color: #999;

  /*
    RTC user specific colors.
    These colors are used for the cursor, username in the editor,
    and the icon of the user.
  */

  --jp-collaborator-color1: #ffad8e;
  --jp-collaborator-color2: #dac83d;
  --jp-collaborator-color3: #72dd76;
  --jp-collaborator-color4: #00e4d0;
  --jp-collaborator-color5: #45d4ff;
  --jp-collaborator-color6: #e2b1ff;
  --jp-collaborator-color7: #ff9de6;

  /* Vega extension styles */

  --jp-vega-background: white;

  /* Sidebar-related styles */

  --jp-sidebar-min-width: 250px;

  /* Search-related styles */

  --jp-search-toggle-off-opacity: 0.5;
  --jp-search-toggle-hover-opacity: 0.8;
  --jp-search-toggle-on-opacity: 1;
  --jp-search-selected-match-background-color: rgb(245, 200, 0);
  --jp-search-selected-match-color: black;
  --jp-search-unselected-match-background-color: var(
    --jp-inverse-layout-color0
  );
  --jp-search-unselected-match-color: var(--jp-ui-inverse-font-color0);

  /* Icon colors that work well with light or dark backgrounds */
  --jp-icon-contrast-color0: var(--md-purple-600);
  --jp-icon-contrast-color1: var(--md-green-600);
  --jp-icon-contrast-color2: var(--md-pink-600);
  --jp-icon-contrast-color3: var(--md-blue-600);

  /* Button colors */
  --jp-accept-color-normal: var(--md-blue-700);
  --jp-accept-color-hover: var(--md-blue-800);
  --jp-accept-color-active: var(--md-blue-900);
  --jp-warn-color-normal: var(--md-red-700);
  --jp-warn-color-hover: var(--md-red-800);
  --jp-warn-color-active: var(--md-red-900);
  --jp-reject-color-normal: var(--md-grey-600);
  --jp-reject-color-hover: var(--md-grey-700);
  --jp-reject-color-active: var(--md-grey-800);

  /* File or activity icons and switch semantic variables */
  --jp-jupyter-icon-color: #f37626;
  --jp-notebook-icon-color: #f37626;
  --jp-json-icon-color: var(--md-orange-700);
  --jp-console-icon-background-color: var(--md-blue-700);
  --jp-console-icon-color: white;
  --jp-terminal-icon-background-color: var(--md-grey-800);
  --jp-terminal-icon-color: var(--md-grey-200);
  --jp-text-editor-icon-color: var(--md-grey-700);
  --jp-inspector-icon-color: var(--md-grey-700);
  --jp-switch-color: var(--md-grey-400);
  --jp-switch-true-position-color: var(--md-orange-900);
}
</style>
<style type="text/css">
/* Force rendering true colors when outputing to pdf */
* {
  -webkit-print-color-adjust: exact;
}

/* Misc */
a.anchor-link {
  display: none;
}

/* Input area styling */
.jp-InputArea {
  overflow: hidden;
}

.jp-InputArea-editor {
  overflow: hidden;
}

.cm-editor.cm-s-jupyter .highlight pre {
/* weird, but --jp-code-padding defined to be 5px but 4px horizontal padding is hardcoded for pre.cm-line */
  padding: var(--jp-code-padding) 4px;
  margin: 0;

  font-family: inherit;
  font-size: inherit;
  line-height: inherit;
  color: inherit;

}

.jp-OutputArea-output pre {
  line-height: inherit;
  font-family: inherit;
}

.jp-RenderedText pre {
  color: var(--jp-content-font-color1);
  font-size: var(--jp-code-font-size);
}

/* Hiding the collapser by default */
.jp-Collapser {
  display: none;
}

@page {
    margin: 0.5in; /* Margin for each printed piece of paper */
}

@media print {
  .jp-Cell-inputWrapper,
  .jp-Cell-outputWrapper {
    display: block;
  }
}
</style>
<!-- Load mathjax -->
<script src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.7/latest.js?config=TeX-AMS_CHTML-full,Safe"> </script>
<!-- MathJax configuration -->
<script type="text/x-mathjax-config">
    init_mathjax = function() {
        if (window.MathJax) {
        // MathJax loaded
            MathJax.Hub.Config({
                TeX: {
                    equationNumbers: {
                    autoNumber: "AMS",
                    useLabelIds: true
                    }
                },
                tex2jax: {
                    inlineMath: [ ['$','$'], ["\\(","\\)"] ],
                    displayMath: [ ['$$','$$'], ["\\[","\\]"] ],
                    processEscapes: true,
                    processEnvironments: true
                },
                displayAlign: 'center',
                messageStyle: 'none',
                CommonHTML: {
                    linebreaks: {
                    automatic: true
                    }
                }
            });

            MathJax.Hub.Queue(["Typeset", MathJax.Hub]);
        }
    }
    init_mathjax();
    </script>
<!-- End of mathjax configuration --><script type="module">
  document.addEventListener("DOMContentLoaded", async () => {
    const diagrams = document.querySelectorAll(".jp-Mermaid > pre.mermaid");
    // do not load mermaidjs if not needed
    if (!diagrams.length) {
      return;
    }
    const mermaid = (await import("https://cdnjs.cloudflare.com/ajax/libs/mermaid/11.10.0/mermaid.esm.min.mjs")).default;
    const elkUrl = "https://cdnjs.cloudflare.com/ajax/libs/mermaid-layout-elk/0.1.9/mermaid-layout-elk.esm.min.mjs";
    if(elkUrl) {
      const elkLayouts = (await import(elkUrl)).default;
      mermaid.registerLayoutLoaders(elkLayouts);
    }
    const parser = new DOMParser();

    mermaid.initialize({
      maxTextSize: 100000,
      maxEdges: 100000,
      startOnLoad: false,
      fontFamily: window
        .getComputedStyle(document.body)
        .getPropertyValue("--jp-ui-font-family"),
      theme: document.querySelector("body[data-jp-theme-light='true']")
        ? "default"
        : "dark",
    });

    let _nextMermaidId = 0;

    function makeMermaidImage(svg) {
      const img = document.createElement("img");
      const doc = parser.parseFromString(svg, "image/svg+xml");
      const svgEl = doc.querySelector("svg");
      const { maxWidth } = svgEl?.style || {};
      const firstTitle = doc.querySelector("title");
      const firstDesc = doc.querySelector("desc");

      img.setAttribute("src", `data:image/svg+xml,${encodeURIComponent(svg)}`);
      if (maxWidth) {
        img.width = parseInt(maxWidth);
      }
      if (firstTitle) {
        img.setAttribute("alt", firstTitle.textContent);
      }
      if (firstDesc) {
        const caption = document.createElement("figcaption");
        caption.className = "sr-only";
        caption.textContent = firstDesc.textContent;
        return [img, caption];
      }
      return [img];
    }

    async function makeMermaidError(text) {
      let errorMessage = "";
      try {
        await mermaid.parse(text);
      } catch (err) {
        errorMessage = `${err}`;
      }

      const result = document.createElement("details");
      result.className = 'jp-RenderedMermaid-Details';
      const summary = document.createElement("summary");
      summary.className = 'jp-RenderedMermaid-Summary';
      const pre = document.createElement("pre");
      const code = document.createElement("code");
      code.innerText = text;
      pre.appendChild(code);
      summary.appendChild(pre);
      result.appendChild(summary);

      const warning = document.createElement("pre");
      warning.innerText = errorMessage;
      result.appendChild(warning);
      return [result];
    }

    async function renderOneMarmaid(src) {
      const id = `jp-mermaid-${_nextMermaidId++}`;
      const parent = src.parentNode;
      let raw = src.textContent.trim();
      const el = document.createElement("div");
      el.style.visibility = "hidden";
      document.body.appendChild(el);
      let results = null;
      let output = null;
      try {
        let { svg } = await mermaid.render(id, raw, el);
        svg = cleanMermaidSvg(svg);
        results = makeMermaidImage(svg);
        output = document.createElement("figure");
        results.map(output.appendChild, output);
      } catch (err) {
        parent.classList.add("jp-mod-warning");
        results = await makeMermaidError(raw);
        output = results[0];
      } finally {
        el.remove();
      }
      parent.classList.add("jp-RenderedMermaid");
      parent.appendChild(output);
    }


    /**
     * Post-process to ensure mermaid diagrams contain only valid SVG and XHTML.
     */
    function cleanMermaidSvg(svg) {
      svg = svg.replace(RE_VOID_ELEMENT, replaceVoidElement);
      return `${SVG_XML_HEADER}${svg}`;
    }


    /**
     * A regular expression for all void elements, which may include attributes and
     * a slash.
     *
     * @see https://developer.mozilla.org/en-US/docs/Glossary/Void_element
     *
     * Of these, only `<br>` is generated by Mermaid in place of `\n`,
     * but _any_ "malformed" tag will break the SVG rendering entirely.
     */
    const RE_VOID_ELEMENT =
      /<\s*(area|base|br|col|embed|hr|img|input|link|meta|param|source|track|wbr)\s*([^>]*?)\s*>/gi;

    /**
     * Ensure a void element is closed with a slash, preserving any attributes.
     */
    function replaceVoidElement(match, tag, rest) {
      rest = rest.trim();
      if (!rest.endsWith('/')) {
        rest = `${rest} /`;
      }
      return `<${tag} ${rest}>`;
    }


  /**
   * Named HTML entities with their decimal equivalent codes.
   *
   * @see https://www.w3.org/TR/WD-html40-970708/sgml/entities.html
   * */
  const HTML_ENTITIES = `<!ENTITY Aacute "&#193;">
<!ENTITY aacute "&#225;">
<!ENTITY Acirc "&#194;">
<!ENTITY acirc "&#226;">
<!ENTITY acute "&#180;">
<!ENTITY AElig "&#198;">
<!ENTITY aelig "&#230;">
<!ENTITY Agrave "&#192;">
<!ENTITY agrave "&#224;">
<!ENTITY alefsym "&#8501;">
<!ENTITY Alpha "&#913;">
<!ENTITY alpha "&#945;">
<!ENTITY amp "&#38;">
<!ENTITY and "&#8869;">
<!ENTITY ang "&#8736;">
<!ENTITY Aring "&#197;">
<!ENTITY aring "&#229;">
<!ENTITY asymp "&#8776;">
<!ENTITY Atilde "&#195;">
<!ENTITY atilde "&#227;">
<!ENTITY Auml "&#196;">
<!ENTITY auml "&#228;">
<!ENTITY bdquo "&#8222;">
<!ENTITY Beta "&#914;">
<!ENTITY beta "&#946;">
<!ENTITY brvbar "&#166;">
<!ENTITY bull "&#8226;">
<!ENTITY cap "&#8745;">
<!ENTITY Ccedil "&#199;">
<!ENTITY ccedil "&#231;">
<!ENTITY cedil "&#184;">
<!ENTITY cent "&#162;">
<!ENTITY Chi "&#935;">
<!ENTITY chi "&#967;">
<!ENTITY circ "&#710;">
<!ENTITY clubs "&#9827;">
<!ENTITY cong "&#8773;">
<!ENTITY copy "&#169;">
<!ENTITY crarr "&#8629;">
<!ENTITY cup "&#8746;">
<!ENTITY curren "&#164;">
<!ENTITY dagger "&#8224;">
<!ENTITY Dagger "&#8225;">
<!ENTITY darr "&#8595;">
<!ENTITY dArr "&#8659;">
<!ENTITY deg "&#176;">
<!ENTITY Delta "&#916;">
<!ENTITY delta "&#948;">
<!ENTITY diams "&#9830;">
<!ENTITY divide "&#247;">
<!ENTITY Eacute "&#201;">
<!ENTITY eacute "&#233;">
<!ENTITY Ecirc "&#202;">
<!ENTITY ecirc "&#234;">
<!ENTITY Egrave "&#200;">
<!ENTITY egrave "&#232;">
<!ENTITY empty "&#8709;">
<!ENTITY emsp "&#8195;">
<!ENTITY ensp "&#8194;">
<!ENTITY epsilon "&#949;">
<!ENTITY Epsilon "&#917;">
<!ENTITY equiv "&#8801;">
<!ENTITY Eta "&#919;">
<!ENTITY eta "&#951;">
<!ENTITY ETH "&#208;">
<!ENTITY eth "&#240;">
<!ENTITY Euml "&#203;">
<!ENTITY euml "&#235;">
<!ENTITY exist "&#8707;">
<!ENTITY fnof "&#402;">
<!ENTITY forall "&#8704;">
<!ENTITY frac12 "&#189;">
<!ENTITY frac14 "&#188;">
<!ENTITY frac34 "&#190;">
<!ENTITY frasl "&#8260;">
<!ENTITY Gamma "&#915;">
<!ENTITY gamma "&#947;">
<!ENTITY ge "&#8805;">
<!ENTITY gt "&#62;">
<!ENTITY harr "&#8596;">
<!ENTITY hArr "&#8660;">
<!ENTITY hearts "&#9829;">
<!ENTITY hellip "&#8230;">
<!ENTITY Iacute "&#205;">
<!ENTITY iacute "&#237;">
<!ENTITY Icirc "&#206;">
<!ENTITY icirc "&#238;">
<!ENTITY iexcl "&#161;">
<!ENTITY Igrave "&#204;">
<!ENTITY igrave "&#236;">
<!ENTITY image "&#8465;">
<!ENTITY infin "&#8734;">
<!ENTITY int "&#8747;">
<!ENTITY Iota "&#921;">
<!ENTITY iota "&#953;">
<!ENTITY iquest "&#191;">
<!ENTITY isin "&#8712;">
<!ENTITY Iuml "&#207;">
<!ENTITY iuml "&#239;">
<!ENTITY Kappa "&#922;">
<!ENTITY kappa "&#954;">
<!ENTITY Lambda "&#923;">
<!ENTITY lambda "&#955;">
<!ENTITY lang "&#9001;">
<!ENTITY laquo "&#171;">
<!ENTITY larr "&#8592;">
<!ENTITY lArr "&#8656;">
<!ENTITY lceil "&#8968;">
<!ENTITY ldquo "&#8220;">
<!ENTITY le "&#8804;">
<!ENTITY lfloor "&#8970;">
<!ENTITY lowast "&#8727;">
<!ENTITY loz "&#9674;">
<!ENTITY lrm "&#8206;">
<!ENTITY lsaquo "&#8249;">
<!ENTITY lsquo "&#8216;">
<!ENTITY lt "&#60;">
<!ENTITY macr "&#175;">
<!ENTITY mdash "&#8212;">
<!ENTITY micro "&#181;">
<!ENTITY middot "&#183;">
<!ENTITY minus "&#8722;">
<!ENTITY Mu "&#924;">
<!ENTITY mu "&#956;">
<!ENTITY nabla "&#8711;">
<!ENTITY nbsp "&#160;">
<!ENTITY ndash "&#8211;">
<!ENTITY ne "&#8800;">
<!ENTITY ni "&#8715;">
<!ENTITY not "&#172;">
<!ENTITY notin "&#8713;">
<!ENTITY nsub "&#8836;">
<!ENTITY Ntilde "&#209;">
<!ENTITY ntilde "&#241;">
<!ENTITY Nu "&#925;">
<!ENTITY nu "&#957;">
<!ENTITY Oacute "&#211;">
<!ENTITY oacute "&#243;">
<!ENTITY Ocirc "&#212;">
<!ENTITY ocirc "&#244;">
<!ENTITY OElig "&#338;">
<!ENTITY oelig "&#339;">
<!ENTITY Ograve "&#210;">
<!ENTITY ograve "&#242;">
<!ENTITY oline "&#8254;">
<!ENTITY Omega "&#937;">
<!ENTITY omega "&#969;">
<!ENTITY Omicron "&#927;">
<!ENTITY omicron "&#959;">
<!ENTITY oplus "&#8853;">
<!ENTITY or "&#8870;">
<!ENTITY ordf "&#170;">
<!ENTITY ordm "&#186;">
<!ENTITY Oslash "&#216;">
<!ENTITY oslash "&#248;">
<!ENTITY Otilde "&#213;">
<!ENTITY otilde "&#245;">
<!ENTITY otimes "&#8855;">
<!ENTITY Ouml "&#214;">
<!ENTITY ouml "&#246;">
<!ENTITY para "&#182;">
<!ENTITY part "&#8706;">
<!ENTITY permil "&#8240;">
<!ENTITY perp "&#8869;">
<!ENTITY Phi "&#934;">
<!ENTITY phi "&#966;">
<!ENTITY Pi "&#928;">
<!ENTITY pi "&#960;">
<!ENTITY piv "&#982;">
<!ENTITY plusmn "&#177;">
<!ENTITY pound "&#163;">
<!ENTITY prime "&#8242;">
<!ENTITY Prime "&#8243;">
<!ENTITY prod "&#8719;">
<!ENTITY prop "&#8733;">
<!ENTITY Psi "&#936;">
<!ENTITY psi "&#968;">
<!ENTITY quot "&#34;">
<!ENTITY radic "&#8730;">
<!ENTITY rang "&#9002;">
<!ENTITY raquo "&#187;">
<!ENTITY rarr "&#8594;">
<!ENTITY rArr "&#8658;">
<!ENTITY rceil "&#8969;">
<!ENTITY rdquo "&#8221;">
<!ENTITY real "&#8476;">
<!ENTITY reg "&#174;">
<!ENTITY rfloor "&#8971;">
<!ENTITY Rho "&#929;">
<!ENTITY rho "&#961;">
<!ENTITY rlm "&#8207;">
<!ENTITY rsaquo "&#8250;">
<!ENTITY rsquo "&#8217;">
<!ENTITY sbquo "&#8218;">
<!ENTITY Scaron "&#352;">
<!ENTITY scaron "&#353;">
<!ENTITY sdot "&#8901;">
<!ENTITY sect "&#167;">
<!ENTITY shy "&#173;">
<!ENTITY Sigma "&#931;">
<!ENTITY sigma "&#963;">
<!ENTITY sigmaf "&#962;">
<!ENTITY sim "&#8764;">
<!ENTITY spades "&#9824;">
<!ENTITY sub "&#8834;">
<!ENTITY sube "&#8838;">
<!ENTITY sum "&#8721;">
<!ENTITY sup "&#8835;">
<!ENTITY sup1 "&#185;">
<!ENTITY sup2 "&#178;">
<!ENTITY sup3 "&#179;">
<!ENTITY supe "&#8839;">
<!ENTITY szlig "&#223;">
<!ENTITY Tau "&#932;">
<!ENTITY tau "&#964;">
<!ENTITY there4 "&#8756;">
<!ENTITY Theta "&#920;">
<!ENTITY theta "&#952;">
<!ENTITY thetasym "&#977;">
<!ENTITY thinsp "&#8201;">
<!ENTITY THORN "&#222;">
<!ENTITY thorn "&#254;">
<!ENTITY tilde "&#732;">
<!ENTITY times "&#215;">
<!ENTITY trade "&#8482;">
<!ENTITY Uacute "&#218;">
<!ENTITY uacute "&#250;">
<!ENTITY uarr "&#8593;">
<!ENTITY uArr "&#8657;">
<!ENTITY Ucirc "&#219;">
<!ENTITY ucirc "&#251;">
<!ENTITY Ugrave "&#217;">
<!ENTITY ugrave "&#249;">
<!ENTITY uml "&#168;">
<!ENTITY upsih "&#978;">
<!ENTITY Upsilon "&#933;">
<!ENTITY upsilon "&#965;">
<!ENTITY Uuml "&#220;">
<!ENTITY uuml "&#252;">
<!ENTITY weierp "&#8472;">
<!ENTITY Xi "&#926;">
<!ENTITY xi "&#958;">
<!ENTITY Yacute "&#221;">
<!ENTITY yacute "&#253;">
<!ENTITY yen "&#165;">
<!ENTITY Yuml "&#376;">
<!ENTITY yuml "&#255;">
<!ENTITY Zeta "&#918;">
<!ENTITY zeta "&#950;">
<!ENTITY zwj "&#8205;">
<!ENTITY zwnj "&#8204;">`.replace(/\n/g, ' ');

  /**
   * A reasonably strict xml declaration.
   */
  const XML_DECL = '<?xml version="1.0" standalone="no"?>';

  /**
   * The beginning of the XML doctype declaration.
   */
  const DOCTYPE_START = `<!DOCTYPE svg PUBLIC "-//W3C//DTD SVG 1.1//EN" "http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd" [`;

  /**
   * The end of the XML docype declaration.
   */
  const DOCTYPE_END = ']>';

  /**
   * A full header for an SVG XML document.
   */
  const SVG_XML_HEADER = `${XML_DECL}
    ${DOCTYPE_START}${HTML_ENTITIES}${DOCTYPE_END}`;

    void Promise.all([...diagrams].map(renderOneMarmaid));
  });
</script>
<style>
  .jp-Mermaid:not(.jp-RenderedMermaid) {
    display: none;
  }

  .jp-RenderedMermaid {
    overflow: auto;
    display: flex;
  }

  .jp-RenderedMermaid.jp-mod-warning {
    width: auto;
    padding: 0.5em;
    margin-top: 0.5em;
    border: var(--jp-border-width) solid var(--jp-warn-color2);
    border-radius: var(--jp-border-radius);
    color: var(--jp-ui-font-color1);
    font-size: var(--jp-ui-font-size1);
    white-space: pre-wrap;
    word-wrap: break-word;
  }

  .jp-RenderedMermaid figure {
    margin: 0;
    overflow: auto;
    max-width: 100%;
  }

  .jp-RenderedMermaid img {
    max-width: 100%;
  }

  .jp-RenderedMermaid-Details > pre {
    margin-top: 1em;
  }

  .jp-RenderedMermaid-Summary {
    color: var(--jp-warn-color2);
  }

  .jp-RenderedMermaid:not(.jp-mod-warning) pre {
    display: none;
  }

  .jp-RenderedMermaid-Summary > pre {
    display: inline-block;
    white-space: normal;
  }
</style>
<!-- End of mermaid configuration --></head>
<body class="jp-Notebook" data-jp-theme-light="true" data-jp-theme-name="JupyterLab Light">
<main>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>1D CNN MODEL</p>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>DATA LOADING AND PREPROCESSING</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">import</span><span class="w"> </span><span class="nn">pandas</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">pd</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">sklearn.preprocessing</span><span class="w"> </span><span class="kn">import</span> <span class="n">StandardScaler</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Load data</span>
<span class="n">train_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">'/Users/mounikapolamreddy/Desktop/train.csv'</span><span class="p">)</span>
<span class="n">labels_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">read_csv</span><span class="p">(</span><span class="s1">'/Users/mounikapolamreddy/Desktop/train_labels.csv'</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># 1. Fill missing values (common in multi-exchange data)</span>
<span class="n">train_df</span> <span class="o">=</span> <span class="n">train_df</span><span class="o">.</span><span class="n">sort_values</span><span class="p">(</span><span class="s1">'date_id'</span><span class="p">)</span><span class="o">.</span><span class="n">ffill</span><span class="p">()</span><span class="o">.</span><span class="n">bfill</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># 2. Scale features (CNNs need normalized inputs)</span>
<span class="n">feature_cols</span> <span class="o">=</span> <span class="p">[</span><span class="n">c</span> <span class="k">for</span> <span class="n">c</span> <span class="ow">in</span> <span class="n">train_df</span><span class="o">.</span><span class="n">columns</span> <span class="k">if</span> <span class="n">c</span> <span class="ow">not</span> <span class="ow">in</span> <span class="p">[</span><span class="s1">'date_id'</span><span class="p">]]</span>
<span class="n">scaler</span> <span class="o">=</span> <span class="n">StandardScaler</span><span class="p">()</span>
<span class="n">train_df</span><span class="p">[</span><span class="n">feature_cols</span><span class="p">]</span> <span class="o">=</span> <span class="n">scaler</span><span class="o">.</span><span class="n">fit_transform</span><span class="p">(</span><span class="n">train_df</span><span class="p">[</span><span class="n">feature_cols</span><span class="p">])</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># 3. Create Sliding Windows</span>
<span class="k">def</span><span class="w"> </span><span class="nf">create_windows</span><span class="p">(</span><span class="n">data</span><span class="p">,</span> <span class="n">labels</span><span class="p">,</span> <span class="n">window_size</span><span class="o">=</span><span class="mi">30</span><span class="p">):</span>
    <span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="p">[],</span> <span class="p">[]</span>
    <span class="c1"># We loop through the data to create (window_size) chunks</span>
    <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">window_size</span><span class="p">,</span> <span class="nb">len</span><span class="p">(</span><span class="n">data</span><span class="p">)):</span>
        <span class="n">X</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">data</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">i</span><span class="o">-</span><span class="n">window_size</span><span class="p">:</span><span class="n">i</span><span class="p">][</span><span class="n">feature_cols</span><span class="p">]</span><span class="o">.</span><span class="n">values</span><span class="p">)</span>
        <span class="n">y</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">labels</span><span class="o">.</span><span class="n">iloc</span><span class="p">[</span><span class="n">i</span><span class="p">][</span><span class="mi">1</span><span class="p">:]</span><span class="o">.</span><span class="n">values</span><span class="p">)</span> <span class="c1"># Skip date_id in labels</span>
    <span class="k">return</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">X</span><span class="p">),</span> <span class="n">np</span><span class="o">.</span><span class="n">array</span><span class="p">(</span><span class="n">y</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># clean the labels</span>
<span class="c1"># This fixes the 86k NaNs we found in your y data</span>
<span class="n">labels_df</span><span class="o">.</span><span class="n">iloc</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">:]</span> <span class="o">=</span> <span class="n">labels_df</span><span class="o">.</span><span class="n">iloc</span><span class="p">[:,</span> <span class="mi">1</span><span class="p">:]</span><span class="o">.</span><span class="n">fillna</span><span class="p">(</span><span class="mi">0</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># define window size</span>
<span class="n">WINDOW_SIZE</span> <span class="o">=</span> <span class="mi">30</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># create windows</span>
<span class="n">X</span><span class="p">,</span> <span class="n">y</span> <span class="o">=</span> <span class="n">create_windows</span><span class="p">(</span><span class="n">train_df</span><span class="p">,</span> <span class="n">labels_df</span><span class="p">,</span> <span class="n">window_size</span><span class="o">=</span><span class="n">WINDOW_SIZE</span><span class="p">)</span>
<span class="n">X</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">nan_to_num</span><span class="p">(</span><span class="n">X</span><span class="p">)</span>
<span class="n">y</span> <span class="o">=</span> <span class="n">np</span><span class="o">.</span><span class="n">nan_to_num</span><span class="p">(</span><span class="n">y</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Validate that there are no NaNs</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Final check - Any NaNs in X? </span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">isnan</span><span class="p">(</span><span class="n">X</span><span class="p">)</span><span class="o">.</span><span class="n">any</span><span class="p">()</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Final check - Any NaNs in y? </span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">isnan</span><span class="p">(</span><span class="n">y</span><span class="p">)</span><span class="o">.</span><span class="n">any</span><span class="p">()</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="s2">"NaNs in X:"</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">isnan</span><span class="p">(</span><span class="n">X</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">())</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"NaNs in y:"</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">isnan</span><span class="p">(</span><span class="n">y</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">())</span>
<span class="nb">print</span><span class="p">(</span><span class="s2">"Infs in X:"</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">isinf</span><span class="p">(</span><span class="n">X</span><span class="p">)</span><span class="o">.</span><span class="n">sum</span><span class="p">())</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Final check - Any NaNs in X? False
Final check - Any NaNs in y? False
NaNs in X: 0
NaNs in y: 0
Infs in X: 0
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Input Shape (Samples, TimeSteps, Features): </span><span class="si">{</span><span class="n">X</span><span class="o">.</span><span class="n">shape</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Input Shape (Samples, TimeSteps, Features): (1931, 30, 557)
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Output Shape (Samples, Targets): </span><span class="si">{</span><span class="n">y</span><span class="o">.</span><span class="n">shape</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Output Shape (Samples, Targets): (1931, 424)
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>BUILDING 1D CNN MODEL</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">import</span><span class="w"> </span><span class="nn">tensorflow</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">tf</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">tensorflow.keras</span><span class="w"> </span><span class="kn">import</span> <span class="n">layers</span><span class="p">,</span> <span class="n">models</span><span class="p">,</span> <span class="n">regularizers</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="k">def</span><span class="w"> </span><span class="nf">build_refined_model</span><span class="p">(</span><span class="n">input_shape</span><span class="p">,</span> <span class="n">output_dim</span><span class="p">):</span>
    <span class="n">model</span> <span class="o">=</span> <span class="n">models</span><span class="o">.</span><span class="n">Sequential</span><span class="p">([</span>
        <span class="c1"># Layer 1: Smaller kernels often work better for daily financial data</span>
        <span class="n">layers</span><span class="o">.</span><span class="n">Conv1D</span><span class="p">(</span><span class="mi">32</span><span class="p">,</span> <span class="n">kernel_size</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">padding</span><span class="o">=</span><span class="s1">'same'</span><span class="p">,</span> <span class="n">input_shape</span><span class="o">=</span><span class="n">input_shape</span><span class="p">,</span>
                      <span class="n">kernel_regularizer</span><span class="o">=</span><span class="n">regularizers</span><span class="o">.</span><span class="n">l2</span><span class="p">(</span><span class="mf">0.001</span><span class="p">)),</span>
        <span class="n">layers</span><span class="o">.</span><span class="n">LeakyReLU</span><span class="p">(</span><span class="n">alpha</span><span class="o">=</span><span class="mf">0.1</span><span class="p">),</span>
        <span class="n">layers</span><span class="o">.</span><span class="n">BatchNormalization</span><span class="p">(),</span>
        <span class="n">layers</span><span class="o">.</span><span class="n">MaxPooling1D</span><span class="p">(</span><span class="n">pool_size</span><span class="o">=</span><span class="mi">2</span><span class="p">),</span>
        
        <span class="c1"># Layer 2: Deeper feature extraction</span>
        <span class="n">layers</span><span class="o">.</span><span class="n">Conv1D</span><span class="p">(</span><span class="mi">64</span><span class="p">,</span> <span class="n">kernel_size</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">padding</span><span class="o">=</span><span class="s1">'same'</span><span class="p">,</span> 
                      <span class="n">kernel_regularizer</span><span class="o">=</span><span class="n">regularizers</span><span class="o">.</span><span class="n">l2</span><span class="p">(</span><span class="mf">0.001</span><span class="p">)),</span>
        <span class="n">layers</span><span class="o">.</span><span class="n">LeakyReLU</span><span class="p">(</span><span class="n">alpha</span><span class="o">=</span><span class="mf">0.1</span><span class="p">),</span>
        <span class="n">layers</span><span class="o">.</span><span class="n">Dropout</span><span class="p">(</span><span class="mf">0.4</span><span class="p">),</span> <span class="c1"># Increased dropout because 1,931 samples is a small dataset</span>
        
        <span class="n">layers</span><span class="o">.</span><span class="n">GlobalAveragePooling1D</span><span class="p">(),</span>
        
        <span class="c1"># Dense Layers</span>
        <span class="n">layers</span><span class="o">.</span><span class="n">Dense</span><span class="p">(</span><span class="mi">128</span><span class="p">),</span>
        <span class="n">layers</span><span class="o">.</span><span class="n">LeakyReLU</span><span class="p">(</span><span class="n">alpha</span><span class="o">=</span><span class="mf">0.1</span><span class="p">),</span>
        <span class="n">layers</span><span class="o">.</span><span class="n">Dense</span><span class="p">(</span><span class="n">output_dim</span><span class="p">,</span> <span class="n">activation</span><span class="o">=</span><span class="s1">'linear'</span><span class="p">)</span> 
    <span class="p">])</span>
    
    <span class="c1"># Lower learning rate is usually safer for volatile commodity returns</span>
    <span class="n">optimizer</span> <span class="o">=</span> <span class="n">tf</span><span class="o">.</span><span class="n">keras</span><span class="o">.</span><span class="n">optimizers</span><span class="o">.</span><span class="n">Adam</span><span class="p">(</span><span class="n">learning_rate</span><span class="o">=</span><span class="mf">0.0005</span><span class="p">)</span>
    
    <span class="n">model</span><span class="o">.</span><span class="n">compile</span><span class="p">(</span><span class="n">optimizer</span><span class="o">=</span><span class="n">optimizer</span><span class="p">,</span> <span class="n">loss</span><span class="o">=</span><span class="s1">'mse'</span><span class="p">,</span> <span class="n">metrics</span><span class="o">=</span><span class="p">[</span><span class="s1">'mae'</span><span class="p">])</span>
    <span class="k">return</span> <span class="n">model</span>
    
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># Re-initialize with the refined architecture</span>
<span class="n">model</span> <span class="o">=</span> <span class="n">build_refined_model</span><span class="p">((</span><span class="n">WINDOW_SIZE</span><span class="p">,</span> <span class="n">X</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">2</span><span class="p">]),</span> <span class="n">y</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">])</span>
<span class="n">model</span><span class="o">.</span><span class="n">summary</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="application/vnd.jupyter.stderr" tabindex="0">
<pre>/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/keras/src/layers/convolutional/base_conv.py:113: UserWarning: Do not pass an `input_shape`/`input_dim` argument to a layer. When using Sequential models, prefer using an `Input(shape)` object as the first layer in the model instead.
  super().__init__(activity_regularizer=activity_regularizer, **kwargs)
/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages/keras/src/layers/activations/leaky_relu.py:41: UserWarning: Argument `alpha` is deprecated. Use `negative_slope` instead.
  warnings.warn(
</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output" data-mime-type="text/html" tabindex="0">
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold">Model: "sequential"</span>
</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output" data-mime-type="text/html" tabindex="0">
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> Layer (type)                    </span>┃<span style="font-weight: bold"> Output Shape           </span>┃<span style="font-weight: bold">       Param # </span>┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ conv1d (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │        <span style="color: #00af00; text-decoration-color: #00af00">53,504</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>)         │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">30</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │           <span style="color: #00af00; text-decoration-color: #00af00">128</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">BatchNormalization</span>)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling1d (<span style="color: #0087ff; text-decoration-color: #0087ff">MaxPooling1D</span>)    │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">15</span>, <span style="color: #00af00; text-decoration-color: #00af00">32</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv1d_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Conv1D</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">15</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │         <span style="color: #00af00; text-decoration-color: #00af00">6,208</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">15</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (<span style="color: #0087ff; text-decoration-color: #0087ff">Dropout</span>)               │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">15</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)         │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling1d        │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">64</span>)             │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
│ (<span style="color: #0087ff; text-decoration-color: #0087ff">GlobalAveragePooling1D</span>)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                   │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │         <span style="color: #00af00; text-decoration-color: #00af00">8,320</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ leaky_re_lu_2 (<span style="color: #0087ff; text-decoration-color: #0087ff">LeakyReLU</span>)       │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">128</span>)            │             <span style="color: #00af00; text-decoration-color: #00af00">0</span> │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (<span style="color: #0087ff; text-decoration-color: #0087ff">Dense</span>)                 │ (<span style="color: #00d7ff; text-decoration-color: #00d7ff">None</span>, <span style="color: #00af00; text-decoration-color: #00af00">424</span>)            │        <span style="color: #00af00; text-decoration-color: #00af00">54,696</span> │
└─────────────────────────────────┴────────────────────────┴───────────────┘
</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output" data-mime-type="text/html" tabindex="0">
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Total params: </span><span style="color: #00af00; text-decoration-color: #00af00">122,856</span> (479.91 KB)
</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output" data-mime-type="text/html" tabindex="0">
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">122,792</span> (479.66 KB)
</pre>
</div>
</div>
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedHTMLCommon jp-RenderedHTML jp-OutputArea-output" data-mime-type="text/html" tabindex="0">
<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace"><span style="font-weight: bold"> Non-trainable params: </span><span style="color: #00af00; text-decoration-color: #00af00">64</span> (256.00 B)
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>TRAINING AND VALIDATION</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">from</span><span class="w"> </span><span class="nn">tensorflow.keras.callbacks</span><span class="w"> </span><span class="kn">import</span> <span class="n">EarlyStopping</span><span class="p">,</span> <span class="n">ReduceLROnPlateau</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># 1. Chronological Split (No shuffling!)</span>
<span class="n">split_idx</span> <span class="o">=</span> <span class="nb">int</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">X</span><span class="p">)</span> <span class="o">*</span> <span class="mf">0.8</span><span class="p">)</span>
<span class="n">X_train</span><span class="p">,</span> <span class="n">X_val</span> <span class="o">=</span> <span class="n">X</span><span class="p">[:</span><span class="n">split_idx</span><span class="p">],</span> <span class="n">X</span><span class="p">[</span><span class="n">split_idx</span><span class="p">:]</span>
<span class="n">y_train</span><span class="p">,</span> <span class="n">y_val</span> <span class="o">=</span> <span class="n">y</span><span class="p">[:</span><span class="n">split_idx</span><span class="p">],</span> <span class="n">y</span><span class="p">[</span><span class="n">split_idx</span><span class="p">:]</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># 2. Callbacks for better training</span>
<span class="n">callbacks</span> <span class="o">=</span> <span class="p">[</span>
    <span class="c1"># Stops training if val_loss doesn't improve for 5 epochs</span>
    <span class="n">EarlyStopping</span><span class="p">(</span><span class="n">monitor</span><span class="o">=</span><span class="s1">'val_loss'</span><span class="p">,</span> <span class="n">patience</span><span class="o">=</span><span class="mi">5</span><span class="p">,</span> <span class="n">restore_best_weights</span><span class="o">=</span><span class="kc">True</span><span class="p">),</span>
    <span class="c1"># Lowers learning rate if the model plateaus</span>
    <span class="n">ReduceLROnPlateau</span><span class="p">(</span><span class="n">monitor</span><span class="o">=</span><span class="s1">'val_loss'</span><span class="p">,</span> <span class="n">factor</span><span class="o">=</span><span class="mf">0.5</span><span class="p">,</span> <span class="n">patience</span><span class="o">=</span><span class="mi">3</span><span class="p">,</span> <span class="n">min_lr</span><span class="o">=</span><span class="mf">1e-6</span><span class="p">)</span>
<span class="p">]</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># 3. Train</span>
<span class="n">history</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">fit</span><span class="p">(</span>
    <span class="n">X_train</span><span class="p">,</span> <span class="n">y_train</span><span class="p">,</span>
    <span class="n">epochs</span><span class="o">=</span><span class="mi">50</span><span class="p">,</span> <span class="c1"># Set high, EarlyStopping will cut it short</span>
    <span class="n">batch_size</span><span class="o">=</span><span class="mi">32</span><span class="p">,</span>
    <span class="n">validation_data</span><span class="o">=</span><span class="p">(</span><span class="n">X_val</span><span class="p">,</span> <span class="n">y_val</span><span class="p">),</span>
    <span class="n">callbacks</span><span class="o">=</span><span class="n">callbacks</span><span class="p">,</span>
    <span class="n">verbose</span><span class="o">=</span><span class="mi">1</span>
<span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Epoch 1/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">2s</span> 13ms/step - loss: 0.0969 - mae: 0.0565 - val_loss: 0.0914 - val_mae: 0.1058 - learning_rate: 5.0000e-04
Epoch 2/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 7ms/step - loss: 0.0598 - mae: 0.0263 - val_loss: 0.0491 - val_mae: 0.0512 - learning_rate: 5.0000e-04
Epoch 3/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 0.0362 - mae: 0.0230 - val_loss: 0.0282 - val_mae: 0.0328 - learning_rate: 5.0000e-04
Epoch 4/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 0.0216 - mae: 0.0214 - val_loss: 0.0165 - val_mae: 0.0245 - learning_rate: 5.0000e-04
Epoch 5/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 0.0130 - mae: 0.0205 - val_loss: 0.0098 - val_mae: 0.0211 - learning_rate: 5.0000e-04
Epoch 6/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 10ms/step - loss: 0.0080 - mae: 0.0201 - val_loss: 0.0060 - val_mae: 0.0199 - learning_rate: 5.0000e-04
Epoch 7/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 0.0050 - mae: 0.0198 - val_loss: 0.0037 - val_mae: 0.0184 - learning_rate: 5.0000e-04
Epoch 8/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">1s</span> 10ms/step - loss: 0.0033 - mae: 0.0197 - val_loss: 0.0024 - val_mae: 0.0178 - learning_rate: 5.0000e-04
Epoch 9/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 10ms/step - loss: 0.0023 - mae: 0.0197 - val_loss: 0.0017 - val_mae: 0.0176 - learning_rate: 5.0000e-04
Epoch 10/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 0.0017 - mae: 0.0196 - val_loss: 0.0012 - val_mae: 0.0173 - learning_rate: 5.0000e-04
Epoch 11/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 0.0014 - mae: 0.0195 - val_loss: 9.9767e-04 - val_mae: 0.0173 - learning_rate: 5.0000e-04
Epoch 12/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 0.0012 - mae: 0.0195 - val_loss: 8.5389e-04 - val_mae: 0.0173 - learning_rate: 5.0000e-04
Epoch 13/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 10ms/step - loss: 0.0011 - mae: 0.0195 - val_loss: 7.7258e-04 - val_mae: 0.0173 - learning_rate: 5.0000e-04
Epoch 14/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">1s</span> 10ms/step - loss: 0.0010 - mae: 0.0195 - val_loss: 7.2549e-04 - val_mae: 0.0172 - learning_rate: 5.0000e-04
Epoch 15/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 0.0010 - mae: 0.0195 - val_loss: 6.9989e-04 - val_mae: 0.0173 - learning_rate: 5.0000e-04
Epoch 16/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 9.8612e-04 - mae: 0.0195 - val_loss: 6.8540e-04 - val_mae: 0.0173 - learning_rate: 5.0000e-04
Epoch 17/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">1s</span> 10ms/step - loss: 9.7453e-04 - mae: 0.0195 - val_loss: 6.7649e-04 - val_mae: 0.0173 - learning_rate: 5.0000e-04
Epoch 18/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 9.6905e-04 - mae: 0.0195 - val_loss: 6.7394e-04 - val_mae: 0.0172 - learning_rate: 2.5000e-04
Epoch 19/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">1s</span> 14ms/step - loss: 9.6663e-04 - mae: 0.0195 - val_loss: 6.7148e-04 - val_mae: 0.0172 - learning_rate: 2.5000e-04
Epoch 20/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 9.6492e-04 - mae: 0.0195 - val_loss: 6.7086e-04 - val_mae: 0.0172 - learning_rate: 2.5000e-04
Epoch 21/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 9.6356e-04 - mae: 0.0195 - val_loss: 6.7002e-04 - val_mae: 0.0172 - learning_rate: 1.2500e-04
Epoch 22/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 9.6301e-04 - mae: 0.0195 - val_loss: 6.6912e-04 - val_mae: 0.0172 - learning_rate: 1.2500e-04
Epoch 23/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">1s</span> 10ms/step - loss: 9.6244e-04 - mae: 0.0195 - val_loss: 6.6878e-04 - val_mae: 0.0172 - learning_rate: 1.2500e-04
Epoch 24/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">1s</span> 10ms/step - loss: 9.6204e-04 - mae: 0.0195 - val_loss: 6.6859e-04 - val_mae: 0.0172 - learning_rate: 6.2500e-05
Epoch 25/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 9.6173e-04 - mae: 0.0195 - val_loss: 6.6846e-04 - val_mae: 0.0172 - learning_rate: 6.2500e-05
Epoch 26/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">1s</span> 9ms/step - loss: 9.6158e-04 - mae: 0.0195 - val_loss: 6.6807e-04 - val_mae: 0.0172 - learning_rate: 6.2500e-05
Epoch 27/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 10ms/step - loss: 9.6132e-04 - mae: 0.0195 - val_loss: 6.6799e-04 - val_mae: 0.0172 - learning_rate: 3.1250e-05
Epoch 28/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 9.6128e-04 - mae: 0.0195 - val_loss: 6.6798e-04 - val_mae: 0.0172 - learning_rate: 3.1250e-05
Epoch 29/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">1s</span> 10ms/step - loss: 9.6115e-04 - mae: 0.0195 - val_loss: 6.6785e-04 - val_mae: 0.0172 - learning_rate: 3.1250e-05
Epoch 30/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 10ms/step - loss: 9.6109e-04 - mae: 0.0195 - val_loss: 6.6781e-04 - val_mae: 0.0172 - learning_rate: 1.5625e-05
Epoch 31/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 9.6104e-04 - mae: 0.0195 - val_loss: 6.6776e-04 - val_mae: 0.0172 - learning_rate: 1.5625e-05
Epoch 32/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 9.6095e-04 - mae: 0.0195 - val_loss: 6.6768e-04 - val_mae: 0.0172 - learning_rate: 1.5625e-05
Epoch 33/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 9.6093e-04 - mae: 0.0195 - val_loss: 6.6768e-04 - val_mae: 0.0172 - learning_rate: 7.8125e-06
Epoch 34/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 9.6086e-04 - mae: 0.0195 - val_loss: 6.6765e-04 - val_mae: 0.0172 - learning_rate: 7.8125e-06
Epoch 35/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 9.6089e-04 - mae: 0.0195 - val_loss: 6.6767e-04 - val_mae: 0.0172 - learning_rate: 7.8125e-06
Epoch 36/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 9.6085e-04 - mae: 0.0195 - val_loss: 6.6764e-04 - val_mae: 0.0172 - learning_rate: 3.9063e-06
Epoch 37/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 9.6082e-04 - mae: 0.0195 - val_loss: 6.6763e-04 - val_mae: 0.0172 - learning_rate: 3.9063e-06
Epoch 38/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 9.6081e-04 - mae: 0.0195 - val_loss: 6.6761e-04 - val_mae: 0.0172 - learning_rate: 3.9063e-06
Epoch 39/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 9.6083e-04 - mae: 0.0195 - val_loss: 6.6761e-04 - val_mae: 0.0172 - learning_rate: 1.9531e-06
Epoch 40/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 10ms/step - loss: 9.6081e-04 - mae: 0.0195 - val_loss: 6.6761e-04 - val_mae: 0.0172 - learning_rate: 1.9531e-06
Epoch 41/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 9.6076e-04 - mae: 0.0195 - val_loss: 6.6760e-04 - val_mae: 0.0172 - learning_rate: 1.9531e-06
Epoch 42/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 9.6078e-04 - mae: 0.0195 - val_loss: 6.6759e-04 - val_mae: 0.0172 - learning_rate: 1.0000e-06
Epoch 43/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 9.6075e-04 - mae: 0.0195 - val_loss: 6.6759e-04 - val_mae: 0.0172 - learning_rate: 1.0000e-06
Epoch 44/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 9.6077e-04 - mae: 0.0195 - val_loss: 6.6759e-04 - val_mae: 0.0172 - learning_rate: 1.0000e-06
Epoch 45/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 9.6075e-04 - mae: 0.0195 - val_loss: 6.6759e-04 - val_mae: 0.0172 - learning_rate: 1.0000e-06
Epoch 46/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 9.6071e-04 - mae: 0.0195 - val_loss: 6.6758e-04 - val_mae: 0.0172 - learning_rate: 1.0000e-06
Epoch 47/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 9.6071e-04 - mae: 0.0195 - val_loss: 6.6758e-04 - val_mae: 0.0172 - learning_rate: 1.0000e-06
Epoch 48/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 8ms/step - loss: 9.6075e-04 - mae: 0.0195 - val_loss: 6.6757e-04 - val_mae: 0.0172 - learning_rate: 1.0000e-06
Epoch 49/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step - loss: 9.6073e-04 - mae: 0.0195 - val_loss: 6.6757e-04 - val_mae: 0.0172 - learning_rate: 1.0000e-06
Epoch 50/50
<span class="ansi-bold">49/49</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">1s</span> 15ms/step - loss: 9.6071e-04 - mae: 0.0195 - val_loss: 6.6756e-04 - val_mae: 0.0172 - learning_rate: 1.0000e-06
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>ENSURING DATA WAS PROCESSED WELL THROUGH VISUALIZATION</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">import</span><span class="w"> </span><span class="nn">matplotlib.pyplot</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">plt</span>

<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">history</span><span class="o">.</span><span class="n">history</span><span class="p">[</span><span class="s1">'loss'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'Train Loss'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">history</span><span class="o">.</span><span class="n">history</span><span class="p">[</span><span class="s1">'val_loss'</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'Val Loss'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="s1">'Model Loss (MSE)'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAiwAAAGzCAYAAAAMr0ziAAAAOnRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjEwLjgsIGh0dHBzOi8vbWF0cGxvdGxpYi5vcmcvwVt1zgAAAAlwSFlzAAAPYQAAD2EBqD+naQAASGhJREFUeJzt3Ql8VNX9/vEnewhLggQIO6jIIgiyClrQiqKigktFagsurX93FLUuRdBai9VKXUCptopaqYgKVaQoIuICyiY/RRFFEZAtoGQjZJ//63smMyQQIMtsCZ/3q9M7c+fO5OYmMk/O+Z5zojwej0cAAAARLDrcJwAAAHA4BBYAABDxCCwAACDiEVgAAEDEI7AAAICIR2ABAAARj8ACAAAiHoEFAABEPAILAACIeAQWAIqKitK9995b5Svxww8/uNdOnz69Tl3Fhx56SJ07d1ZJSYki0UknnaQ//OEP4T4NIKQILECEsA99+/C320cffXTA87aKRps2bdzz5557rmqT999/3533q6++qkiXlZWlv/71r7rjjjsUHb3vn0jfz+Z3v/tdha/74x//6D9m165d5Z578803NXjwYDVr1kxJSUk6+uijdckll2j+/PkHhL+D3R588EH/sXZuU6dO1fbt24NyDYBIFBvuEwBQXmJiombMmKFTTjml3P7Fixfrxx9/VEJCApcsiJ599lkVFRVp1KhRFf5sXnvtNT355JOKj48v99x//vMf93xeXl65/X/72990++23u8By1113ucCyfv16vfvuu3r55Zd11llnlTvevu4555xzwNc+8cQT/feHDx+uRo0aufP405/+FIDvGoh8BBYgwtiH1axZs/T4448rNnbff6IWYnr37n3AX+8IrOeee07nn3++Cx/7s3Dxxhtv6H//+58LDT5LlizRhg0bdNFFF7lA42PB5/7779cZZ5yhd95554D3S09PP2Bfr1699Jvf/OaQ52gtPxdffLFeeOEF3Xfffa4FBqjr6BICIoz9hf3TTz9pwYIF/n0FBQWuO+XXv/51ha/Zs2ePbr31VtdlZC0wnTp1cn/Z778Ye35+vm655RY1bdpUDRs2dB/M1mpTkS1btujKK69U8+bN3Xsef/zxrvUhmL7//nv96le/0lFHHeVaIqxW46233jrguCeeeMKdjx3TuHFj9enTxwU6n+zsbN18881q3769O3frirHQsGrVqkN+fQsdn3/+uYYMGVLh861atdKgQYPKfS3z0ksvqXv37urWrVu5/RYurYvp5JNPrvD97Lyqy76fjRs3avXq1dV+D6A2IbAAEcY+ZAcMGOC6GHzsL/rMzExdeumlBxxvocSCx9///nfXAjB58mQXWKwbYty4ceWOtfqLRx99VGeeeaariYiLi9OwYcMOeM8dO3a4sGDdFjfccIMee+wxHXvssbrqqqvc64PBvubAgQP19ttv67rrrtMDDzzgulfse5s9e7b/uGeeeUY33XSTunbt6s7FWhh69uypTz/91H/MNddco6eeesq1eFi3yW233aZ69epp7dq1hzwHaynxtXIcjIVGq0nJycnxt6JYi1hFYdICiX1dO/7nn3+u1HXIzc11QWf/m32dsqy1zXz88ceVel+g1vMAiAjPPfecNYd4li9f7pkyZYqnYcOGntzcXPfcr371K89pp53m7rdr184zbNgw/+vmzJnjXvfnP/+53PtdfPHFnqioKM/69evd49WrV7vjrrvuunLH/frXv3b7J06c6N931VVXeVq0aOHZtWtXuWMvvfRST3Jysv+8NmzY4F5r534oixYtcsfNmjXroMfcfPPN7pgPP/zQvy87O9vToUMHT/v27T3FxcVu3/Dhwz3HH3/8Ib+eneP111/vqarx48e7c7Cvuz/bb+/5888/e+Lj4z0vvvii2//WW2+56/zDDz+4a2jH7dy50/+6CRMmuH3169f3nH322Z4HHnjAs3LlygPe33ctD3ZbunTpAa+x87j22mur/H0CtREtLEAEshEke/fu1dy5c133hm0P1h00b948xcTEuFaHsqyLyD5nrXXGd5zZ/zjrOinLXmN1GOedd567X/av/KFDh7qWnsN1rVSHnV+/fv3KFRs3aNBAV199tRtB89VXX7l9KSkprhtr+fLlB30vO8ZaXLZu3Vqlc7CuOKsbsq97MNYFZS1ZvhYw6x6ylqF27dpVeLy1ANkxVjRrrUc2mshaR6wVp6IWH/t+rTtw/5u1KFV0LtQ04UhB0S0QgazGxOoo7IPOugiKi4tdkWVFrI6hZcuWrialrC5duvif922tWPOYY44pd5x1H5W1c+dOZWRk6Omnn3a3ilRULFpTdn79+/c/YH/Z78NqRGxIr3VVWbixbirr3rIwV7ZOxOZRGTNmjKvpsXBghcyj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"/>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>EVALUATE PERFORMANCE</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># 1. Get predictions on the validation set</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_val</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre><span class="ansi-bold">13/13</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 9ms/step
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># 2. Pick a random target (e.g., Target 0) and a small slice of time</span>
<span class="n">target_idx</span> <span class="o">=</span> <span class="mi">0</span> 
<span class="n">plt</span><span class="o">.</span><span class="n">figure</span><span class="p">(</span><span class="n">figsize</span><span class="o">=</span><span class="p">(</span><span class="mi">12</span><span class="p">,</span> <span class="mi">5</span><span class="p">))</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">y_val</span><span class="p">[:</span><span class="mi">100</span><span class="p">,</span> <span class="n">target_idx</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'Actual Return'</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=</span><span class="mf">0.7</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">plot</span><span class="p">(</span><span class="n">y_pred</span><span class="p">[:</span><span class="mi">100</span><span class="p">,</span> <span class="n">target_idx</span><span class="p">],</span> <span class="n">label</span><span class="o">=</span><span class="s1">'Predicted Return'</span><span class="p">,</span> <span class="n">alpha</span><span class="o">=</span><span class="mf">0.7</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">title</span><span class="p">(</span><span class="sa">f</span><span class="s1">'Target </span><span class="si">{</span><span class="n">target_idx</span><span class="si">}</span><span class="s1">: Actual vs Predicted'</span><span class="p">)</span>
<span class="n">plt</span><span class="o">.</span><span class="n">legend</span><span class="p">()</span>
<span class="n">plt</span><span class="o">.</span><span class="n">show</span><span class="p">()</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedImage jp-OutputArea-output" tabindex="0">
<img alt="No description has been provided for this image" class="" src="data:image/png;base64,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"/>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># 3. Calculate Overall Correlation (The competition metric)</span>
<span class="kn">from</span><span class="w"> </span><span class="nn">scipy.stats</span><span class="w"> </span><span class="kn">import</span> <span class="n">spearmanr</span>
<span class="n">correlations</span> <span class="o">=</span> <span class="p">[]</span>
<span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="n">y_val</span><span class="o">.</span><span class="n">shape</span><span class="p">[</span><span class="mi">1</span><span class="p">]):</span>
    <span class="n">corr</span><span class="p">,</span> <span class="n">_</span> <span class="o">=</span> <span class="n">spearmanr</span><span class="p">(</span><span class="n">y_val</span><span class="p">[:,</span> <span class="n">i</span><span class="p">],</span> <span class="n">y_pred</span><span class="p">[:,</span> <span class="n">i</span><span class="p">])</span>
    <span class="n">correlations</span><span class="o">.</span><span class="n">append</span><span class="p">(</span><span class="n">corr</span><span class="p">)</span>

<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Average Spearman Correlation across all 424 targets: </span><span class="si">{</span><span class="n">np</span><span class="o">.</span><span class="n">mean</span><span class="p">(</span><span class="n">correlations</span><span class="p">)</span><span class="si">:</span><span class="s2">.4f</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>Average Spearman Correlation across all 424 targets: 0.0069
</pre>
</div>
</div>
</div>
</div>
</div>
<div class="jp-Cell jp-MarkdownCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea"><div class="jp-InputPrompt jp-InputArea-prompt">
</div><div class="jp-RenderedHTMLCommon jp-RenderedMarkdown jp-MarkdownOutput" data-mime-type="text/markdown">
<p>TESTING</p>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="c1"># 1. Generate predictions for the validation set</span>
<span class="n">y_pred</span> <span class="o">=</span> <span class="n">model</span><span class="o">.</span><span class="n">predict</span><span class="p">(</span><span class="n">X_val</span><span class="p">)</span>

<span class="c1"># 2. Create the 'solution' DataFrame (The Ground Truth)</span>
<span class="c1"># We need to add 'date_id' back as a dummy index for the score function to work</span>
<span class="n">target_col_names</span> <span class="o">=</span> <span class="p">[</span><span class="sa">f</span><span class="s1">'target_</span><span class="si">{</span><span class="n">i</span><span class="si">}</span><span class="s1">'</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="mi">424</span><span class="p">)]</span>
<span class="n">solution_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">y_val</span><span class="p">,</span> <span class="n">columns</span><span class="o">=</span><span class="n">target_col_names</span><span class="p">)</span>
<span class="n">solution_df</span><span class="p">[</span><span class="s1">'date_id'</span><span class="p">]</span> <span class="o">=</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">solution_df</span><span class="p">))</span>

<span class="c1"># 3. Create the 'submission' DataFrame (The Model Predictions)</span>
<span class="n">submission_df</span> <span class="o">=</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">(</span><span class="n">y_pred</span><span class="p">,</span> <span class="n">columns</span><span class="o">=</span><span class="n">target_col_names</span><span class="p">)</span>
<span class="n">submission_df</span><span class="p">[</span><span class="s1">'date_id'</span><span class="p">]</span> <span class="o">=</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">submission_df</span><span class="p">))</span>

<span class="c1"># 4. Define the row_id column required by the score function</span>
<span class="n">ROW_ID_NAME</span> <span class="o">=</span> <span class="s1">'date_id'</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre><span class="ansi-bold">13/13</span> <span class="ansi-green-fg">━━━━━━━━━━━━━━━━━━━━</span> <span class="ansi-bold">0s</span> 5ms/step 
</pre>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell jp-mod-noOutputs">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="kn">import</span><span class="w"> </span><span class="nn">numpy</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">np</span>
<span class="kn">import</span><span class="w"> </span><span class="nn">pandas</span><span class="w"> </span><span class="k">as</span><span class="w"> </span><span class="nn">pd</span>

<span class="n">SOLUTION_NULL_FILLER</span> <span class="o">=</span> <span class="o">-</span><span class="mi">999999</span>

<span class="k">def</span><span class="w"> </span><span class="nf">rank_correlation_sharpe_ratio</span><span class="p">(</span><span class="n">merged_df</span><span class="p">:</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">float</span><span class="p">:</span>
    <span class="n">prediction_cols</span> <span class="o">=</span> <span class="p">[</span><span class="n">col</span> <span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">merged_df</span><span class="o">.</span><span class="n">columns</span> <span class="k">if</span> <span class="n">col</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s1">'prediction_'</span><span class="p">)]</span>
    <span class="n">target_cols</span> <span class="o">=</span> <span class="p">[</span><span class="n">col</span> <span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">merged_df</span><span class="o">.</span><span class="n">columns</span> <span class="k">if</span> <span class="n">col</span><span class="o">.</span><span class="n">startswith</span><span class="p">(</span><span class="s1">'target_'</span><span class="p">)]</span>

    <span class="k">def</span><span class="w"> </span><span class="nf">_compute_rank_correlation</span><span class="p">(</span><span class="n">row</span><span class="p">):</span>
        <span class="c1"># Identify non-null targets</span>
        <span class="n">non_null_targets</span> <span class="o">=</span> <span class="p">[</span><span class="n">col</span> <span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">target_cols</span> <span class="k">if</span> <span class="ow">not</span> <span class="n">pd</span><span class="o">.</span><span class="n">isnull</span><span class="p">(</span><span class="n">row</span><span class="p">[</span><span class="n">col</span><span class="p">])]</span>
        <span class="n">matching_predictions</span> <span class="o">=</span> <span class="p">[</span><span class="n">col</span> <span class="k">for</span> <span class="n">col</span> <span class="ow">in</span> <span class="n">prediction_cols</span> <span class="k">if</span> <span class="n">col</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="s1">'prediction'</span><span class="p">,</span> <span class="s1">'target'</span><span class="p">)</span> <span class="ow">in</span> <span class="n">non_null_targets</span><span class="p">]</span>
        
        <span class="k">if</span> <span class="ow">not</span> <span class="n">non_null_targets</span><span class="p">:</span>
            <span class="k">return</span> <span class="mf">0.0</span>
        
        <span class="c1"># Calculate standard deviation to check for constant values</span>
        <span class="n">target_std</span> <span class="o">=</span> <span class="n">row</span><span class="p">[</span><span class="n">non_null_targets</span><span class="p">]</span><span class="o">.</span><span class="n">std</span><span class="p">(</span><span class="n">ddof</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
        <span class="n">pred_std</span> <span class="o">=</span> <span class="n">row</span><span class="p">[</span><span class="n">matching_predictions</span><span class="p">]</span><span class="o">.</span><span class="n">std</span><span class="p">(</span><span class="n">ddof</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
        
        <span class="k">if</span> <span class="n">target_std</span> <span class="o">==</span> <span class="mi">0</span> <span class="ow">or</span> <span class="n">pred_std</span> <span class="o">==</span> <span class="mi">0</span><span class="p">:</span>
            <span class="k">return</span> <span class="mf">0.0</span> <span class="c1"># Return 0 correlation if there's no variance</span>
            
        <span class="c1"># Compute Spearman Rank Correlation</span>
        <span class="k">return</span> <span class="n">row</span><span class="p">[</span><span class="n">matching_predictions</span><span class="p">]</span><span class="o">.</span><span class="n">rank</span><span class="p">()</span><span class="o">.</span><span class="n">corr</span><span class="p">(</span><span class="n">row</span><span class="p">[</span><span class="n">non_null_targets</span><span class="p">]</span><span class="o">.</span><span class="n">rank</span><span class="p">())</span>

    <span class="n">daily_rank_corrs</span> <span class="o">=</span> <span class="n">merged_df</span><span class="o">.</span><span class="n">apply</span><span class="p">(</span><span class="n">_compute_rank_correlation</span><span class="p">,</span> <span class="n">axis</span><span class="o">=</span><span class="mi">1</span><span class="p">)</span>
    
    <span class="n">std_dev</span> <span class="o">=</span> <span class="n">daily_rank_corrs</span><span class="o">.</span><span class="n">std</span><span class="p">(</span><span class="n">ddof</span><span class="o">=</span><span class="mi">0</span><span class="p">)</span>
    <span class="k">if</span> <span class="n">std_dev</span> <span class="o">==</span> <span class="mi">0</span> <span class="ow">or</span> <span class="n">np</span><span class="o">.</span><span class="n">isnan</span><span class="p">(</span><span class="n">std_dev</span><span class="p">):</span>
        <span class="k">return</span> <span class="mf">0.0</span>
        
    <span class="n">sharpe_ratio</span> <span class="o">=</span> <span class="n">daily_rank_corrs</span><span class="o">.</span><span class="n">mean</span><span class="p">()</span> <span class="o">/</span> <span class="n">std_dev</span>
    <span class="k">return</span> <span class="nb">float</span><span class="p">(</span><span class="n">sharpe_ratio</span><span class="p">)</span>

<span class="k">def</span><span class="w"> </span><span class="nf">score</span><span class="p">(</span><span class="n">solution</span><span class="p">:</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">,</span> <span class="n">submission</span><span class="p">:</span> <span class="n">pd</span><span class="o">.</span><span class="n">DataFrame</span><span class="p">,</span> <span class="n">row_id_column_name</span><span class="p">:</span> <span class="nb">str</span><span class="p">)</span> <span class="o">-&gt;</span> <span class="nb">float</span><span class="p">:</span>
    <span class="c1"># Work on copies to avoid modifying original DataFrames</span>
    <span class="n">sol</span> <span class="o">=</span> <span class="n">solution</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
    <span class="n">sub</span> <span class="o">=</span> <span class="n">submission</span><span class="o">.</span><span class="n">copy</span><span class="p">()</span>
    
    <span class="k">del</span> <span class="n">sol</span><span class="p">[</span><span class="n">row_id_column_name</span><span class="p">]</span>
    <span class="k">del</span> <span class="n">sub</span><span class="p">[</span><span class="n">row_id_column_name</span><span class="p">]</span>
    
    <span class="c1"># Rename submission columns to prediction_X</span>
    <span class="n">sub</span><span class="o">.</span><span class="n">columns</span> <span class="o">=</span> <span class="p">[</span><span class="sa">f</span><span class="s2">"prediction_</span><span class="si">{</span><span class="n">i</span><span class="si">}</span><span class="s2">"</span> <span class="k">for</span> <span class="n">i</span> <span class="ow">in</span> <span class="nb">range</span><span class="p">(</span><span class="nb">len</span><span class="p">(</span><span class="n">sub</span><span class="o">.</span><span class="n">columns</span><span class="p">))]</span>
    
    <span class="c1"># Replace the competition's null filler</span>
    <span class="n">sol</span> <span class="o">=</span> <span class="n">sol</span><span class="o">.</span><span class="n">replace</span><span class="p">(</span><span class="n">SOLUTION_NULL_FILLER</span><span class="p">,</span> <span class="n">np</span><span class="o">.</span><span class="n">nan</span><span class="p">)</span>
    
    <span class="k">return</span> <span class="n">rank_correlation_sharpe_ratio</span><span class="p">(</span><span class="n">pd</span><span class="o">.</span><span class="n">concat</span><span class="p">([</span><span class="n">sol</span><span class="p">,</span> <span class="n">sub</span><span class="p">],</span> <span class="n">axis</span><span class="o">=</span><span class="s1">'columns'</span><span class="p">))</span>
</pre></div>
</div>
</div>
</div>
</div>
</div><div class="jp-Cell jp-CodeCell jp-Notebook-cell">
<div class="jp-Cell-inputWrapper" tabindex="0">
<div class="jp-Collapser jp-InputCollapser jp-Cell-inputCollapser">
</div>
<div class="jp-InputArea jp-Cell-inputArea">
<div class="jp-InputPrompt jp-InputArea-prompt">In [ ]:</div>
<div class="jp-CodeMirrorEditor jp-Editor jp-InputArea-editor" data-type="inline">
<div class="cm-editor cm-s-jupyter">
<div class="highlight hl-ipython3"><pre><span></span><span class="n">final_score</span> <span class="o">=</span> <span class="n">score</span><span class="p">(</span><span class="n">solution_df</span><span class="p">,</span> <span class="n">submission_df</span><span class="p">,</span> <span class="n">ROW_ID_NAME</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"--- Final Evaluation ---"</span><span class="p">)</span>
<span class="nb">print</span><span class="p">(</span><span class="sa">f</span><span class="s2">"Sharpe Ratio of Rank Correlation: </span><span class="si">{</span><span class="n">final_score</span><span class="si">:</span><span class="s2">.4f</span><span class="si">}</span><span class="s2">"</span><span class="p">)</span>
</pre></div>
</div>
</div>
</div>
</div>
<div class="jp-Cell-outputWrapper">
<div class="jp-Collapser jp-OutputCollapser jp-Cell-outputCollapser">
</div>
<div class="jp-OutputArea jp-Cell-outputArea">
<div class="jp-OutputArea-child">
<div class="jp-OutputPrompt jp-OutputArea-prompt"></div>
<div class="jp-RenderedText jp-OutputArea-output" data-mime-type="text/plain" tabindex="0">
<pre>--- Final Evaluation ---
Sharpe Ratio of Rank Correlation: 0.0000
</pre>
</div>
</div>
</div>
</div>
</div>
</main>
</body>
</html>
