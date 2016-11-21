
##########
# Import #
##########
import csv

def f_or_none(f, row_el):
    """
    applies the parser f to a row element, will return None if parser error
    """
    try:
        return f(row_el)
    except:
        return None

def parse_row(parsers, input_row):
    """ apply a list of parsers to elements in input_row. Parsers may be
    None which returns element
    """
    return [f_or_none(parser,el) if parser is not None else el 
            for parser,el in zip(parsers, input_row)]

def parse_rows_with(reader, parsers):
    """
    apply parsers to each row in reader object
    """
    for row in reader:
        yield parse_row(parsers, row)


data = []

def main(filename, parsers, delete_None_rows=False):
    """ 
    Open csv file and apply parsers list of parser functions to each row. If
    delete_None_rows, rows containing Nones will be deleted.
    """

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        # call our parse_rows_with wrapper for the reader
        for row in parse_rows_with(reader, parsers):
            data.append(row)
        
    if delete_None_rows:
        # if delete is true delete any rows containing None(s)
        for idx, row in enumerate(data):
            if any(x is None for x in row):
                del data[idx]
    
    return data

