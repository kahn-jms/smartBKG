#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

from smartBKG.preprocessing import CreateJointMemmap  # type:ignore


def GetCmdArgs():
    parser = argparse.ArgumentParser(
        description='''Create joint memory map of many npy files''',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('-i', type=str, required=True, nargs='+',
                        help="Path to input numpy files.", metavar="INPUT",
                        dest='in_files')
    parser.add_argument('-o', type=str, required=True,
                        help="Output memmap file", metavar="OUTPUT",
                        dest='out_file')
    parser.add_argument('--npy-load-memmap', action='store_true',
                        help="Load npy files as memmaps. Cannot be used for dtype=object",
                        dest='mmap_flag')
    return parser.parse_args()


if __name__ == '__main__':

    args = GetCmdArgs()
    os.makedirs(os.path.dirname(args.out_file), exist_ok=True)

    # ADD to CreateJointMemmaps, should just be bool argument, let it set the memmap mode
    mmap_mode = None
    if args.mmap_flag:
        mmap_mode = 'r'

    # Create the memmap object
    cjm = CreateJointMemmap(memmap_mode=mmap_mode)

    # Append the input files to the memmap, setting the location for the
    # final memmap containing all the files
    cjm.grow_memmap(args.in_files, args.out_file)

    # Flush the memory, this ensures the entire memmap is written to disk
    cjm.flush_memmap()
