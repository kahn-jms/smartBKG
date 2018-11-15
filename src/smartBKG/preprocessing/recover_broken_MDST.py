#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import basf2 as b2  # noqa
import modularAnalysis as ma  # noqa

main = b2.create_path()

main.add_module('RootInput', recovery=True)
main.add_module('RootOutput')

b2.process(main)
