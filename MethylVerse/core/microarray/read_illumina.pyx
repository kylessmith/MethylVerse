# -*- coding: utf-8 -*-

from __future__ import division


def readstr(afile):
  '''
  String data are encoded as a sequence of one or more length bytes followed
  by the specified number of data bytes.
  The lower 7 bits of each length byte encodes the bits that comprise the
  length of the following byte string.  When the most significant bit it
  set, then an additional length byte follows with 7 additional high bits to
  be added to the current length.  The following string lengths are
  accommodated by increasing sequences of length bytes:
  length  maximum
  bytes   length
  ------  --------
    1       127 B
    2        16 KB
    3         2 MB
    4       256 MB
    5        32 GB
  While this seems like a sensible progression, there is some uncertainty
  about this interpretation, since the longest of string observed in the
  wild has been of length 6,264 with two length bytes.
  '''
  cdef char *s
  cdef int n, m
  cdef bytes l

  read = afile.read

  # Read 1 byte, keeping a reference to the Python string
  # long enough to get the value
  l = read(1)
  s = l
  m = <unsigned char>s[0]

  # Fast-path return for zero length strings
  # (these occur frequently)
  if not m:
    return ''

  # Set lower 0-6 bits
  n = m&0x7F

  # Hand-unrolled code to process length bytes 2-5, when present

  # Read second length byte, if present
  if m&0x80:
    # Read 1 byte, keeping a reference to the Python string
    # long enough to get the value
    l = read(1)
    s = l
    m = <unsigned char>s[0]

    # Set bits 7-13
    n += (m&0x7F)<<7

  # Read third length byte, if present
  if m&0x80:
    # Read 1 byte, keeping a reference to the Python string
    # long enough to get the value
    l = read(1)
    s = l
    m = <unsigned char>s[0]

    # Set bits 14-20
    n += (m&0x7F)<<14

  # Read forth length byte, if present
  if m&0x80:
    # Read 1 byte, keeping a reference to the Python string
    # long enough to get the value
    l = read(1)
    s = l
    m = <unsigned char>s[0]

    # Set bits 21-27
    n += (m&0x7F)<<21

  # Read fifth length byte, if present
  if m&0x80:
    # Read 1 byte, keeping a reference to the Python string
    # long enough to get the value
    l  = read(1)
    s  = l
    m  = <unsigned char>s[0]

    # Set bits 27-33
    n += (m&0x7F)<<28

  if m&0x80:
    raise ValueError('Encoded string too long')

  return read(n)