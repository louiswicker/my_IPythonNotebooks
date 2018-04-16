#!/usr/bin/env python

import os

# remove all module files in directory - this can trip you up bad!

cmd = 'rm *.mod'
print "\n=====================================================\n"
print("   ---> Removing all module files...safety first!")
print "\n=====================================================\n"
ret = os.system(cmd)

cmd = "f2py --fcompiler='gnu95' --f90flags='-O3' -c -m kessler1d kessler1d.f90"
os.system(cmd)
