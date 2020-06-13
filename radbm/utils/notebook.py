import numpy as np
from IPython.core.display import display, clear_output, HTML

html_caption = '<caption style="text-align:left">{}</caption>'
html_table = '<table style="float:left;width:50%">{}</table>'
html_row = '<tr>{}</tr>'
html_item = '<th style="text-align:center">{}</th>'
html_bold_item = '<th style="text-align:center"><b>{}</b></th>'

def notebook_html_table(table, title=None, cols_name=None, rows_name=None):
    rows = list()
    for n, row in enumerate(table):
        row = ['{:.4f}'.format(e) for e in row]
        row = list(map(html_item.format, row))
        if rows_name: row = [html_bold_item.format(rows_name[n])] + row
        rows.append(html_row.format(''.join(row)))
    if cols_name:
        if rows_name: cols_name = [''] + cols_name
        header = list(map(html_bold_item.format, cols_name))
        header = html_row.format(''.join(header))
        rows = [header] + rows
    if title: rows = [html_caption.format(title)] + rows
    table = html_table.format(''.join(rows))
    clear_output(wait=True)
    display(HTML(table))
    return table