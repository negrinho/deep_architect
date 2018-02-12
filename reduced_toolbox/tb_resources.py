def convert_between_byte_units(x, src_units='b', dst_units='mb'):
    units = ['b', 'kb', 'mb', 'gb', 'tb']
    assert (src_units in units) and (dst_units in units)
    return x / float(
        2 ** (10 * (units.index(dst_units) - units.index(src_units))))