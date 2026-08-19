from collections import namedtuple
import struct
import numpy as np
import pandas as pd

# Local import
from .read_illumina import readstr


ManifestRow = namedtuple('ManifestRow',
                         'ilmnid name design_strand alleles assay_type_id norm_id '
                         'addressA_id alleleA_probe_sequence addressB_id alleleB_probe_sequence '
                         'genome_version chromosome mapinfo ploidy species '
                         'source source_version source_strand source_sequence top_genomic_sequence '
                         'customer_strand genomic_strand')

assay_type_map = { 0 : 'Infinium II',
                   1 : 'Infinium I red channel',
                   2 : 'Infinium I green channel' }


IDATData = namedtuple('IDATData', 'filename version snp_count illumina_ids sds means bead_counts '
                                  'midblock red_green manifest barcode format label opa sampleid '
                                  'descr plate well runinfo')

IDATData2 = namedtuple('IDATData2', 'filename version snp_count idat_values '
                                  'midblock red_green manifest barcode format label opa sampleid '
                                  'descr plate well runinfo')


IDAT_FIELD_CODES = { 1000 : 'snp_count',
                      102 : 'illumina_ids',
                      103 : 'sds',
                      104 : 'means',
                      107 : 'bead_counts',
                      200 : 'midblock',
                      300 : 'runinfo',
                      400 : 'red_green',
                      401 : 'manifest',
                      402 : 'barcode',
                      403 : 'format',
                      404 : 'label',
                      405 : 'opa',
                      406 : 'sampleid',
                      407 : 'descr',
                      408 : 'plate',
                      409 : 'well',
                      510 : 'unknown' }
                    

def read_idat(filename: str):
    """
    """
    

    idat = open(filename,"rb")

    if idat.read(4) != b'IDAT':
        raise ValueError('Invalid IDAT file signature')

    version, = struct.unpack('<L', idat.read(4))

    if version != 3:
        raise ValueError('Invalid IDAT version')

    unknown0, field_count = struct.unpack('<LL', idat.read(8))

    fields = {}
    for i in range(field_count):
        field_code, field_offset = struct.unpack('<HQ', idat.read(10))
        field_name = IDAT_FIELD_CODES.get(field_code,'Unknown')

        if field_name in fields:
            raise ValueError('Invalid duplicated field %s in IDAT file' % field_name)

        fields[field_name] = field_offset

    
    if 'snp_count' in fields:
        idat.seek(fields['snp_count'])
        snp_count, = struct.unpack('<L',idat.read(4))

    if 'illumina_ids' in fields:
        idat.seek(fields['illumina_ids'])
        illumina_ids = np.fromfile(idat, dtype='<u4', count=snp_count)

    # Construct DataFrame
    idat_values = pd.DataFrame(np.zeros((snp_count, 3)),
                                columns = ["means","sds","bead_counts"],
                                index = illumina_ids)


    if 'sds' in fields:
        idat.seek(fields['sds'])
        idat_values.loc[:,"sds"] = np.fromfile(idat, dtype='<u2', count=snp_count)

    if 'means' in fields:
        idat.seek(fields['means'])
        idat_values.loc[:,"means"] = np.fromfile(idat, dtype='<u2', count=snp_count)

    if 'bead_counts' in fields:
        idat.seek(fields['bead_counts'])
        idat_values.loc[:,"bead_counts"] = np.fromfile(idat, dtype='b', count=snp_count)
    
    idat.close()
    
    return idat_values


def get_idat_type(filename: str) -> str:
    """
    Determine version of Methylation array from IDAT file

    Parameters
    ----------
        filename : str

    Returns
    -------
        array_type : str
    """
    
    idat = open(filename,"rb")

    if idat.read(4) != b'IDAT':
        raise ValueError('Invalid IDAT file signature')

    version, = struct.unpack('<L', idat.read(4))

    if version != 3:
        raise ValueError('Invalid IDAT version')

    unknown0, field_count = struct.unpack('<LL', idat.read(8))

    fields = {}
    for i in range(field_count):
        field_code, field_offset = struct.unpack('<HQ', idat.read(10))
        field_name = IDAT_FIELD_CODES.get(field_code,'Unknown')

        if field_name in fields:
            raise ValueError('Invalid duplicated field %s in IDAT file' % field_name)

        fields[field_name] = field_offset

    
    if 'snp_count' in fields:
        idat.seek(fields['snp_count'])
        snp_count, = struct.unpack('<L',idat.read(4))

    # Determine array type
    array_type = "27k"
    if  snp_count > 200000:
        array_type = "450k"
    if snp_count > 700000:
        array_type = "EPIC_v1"
    if snp_count > 1100000:
        array_type = "EPIC_v2"

    return array_type