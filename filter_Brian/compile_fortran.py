#!/usr/bin/env python

import sys
import os
import glob
import string
from optparse import OptionParser
subfolders = ['./']
# append path to the folder
sys.path.extend(subfolders)

parser = OptionParser()
parser.add_option("-f","--file",dest="file",type="string", help = "fortran file to be compiled - if not supplied, all fortran files are!")
parser.add_option("-a","--all",dest="all",default=False,help = "Boolean flag to compile all files (default=False)", action="store_true")
parser.add_option("-c","--compiler",dest="compiler",type="string",default=None,help = "compiler to use with f2py")

(options, args) = parser.parse_args()

if options.all:
    fortran_files = glob.glob("*.f")
    fortran_files = fortran_files + glob.glob("*.f77")
    fortran_files = fortran_files + glob.glob("*.f90")

if options.file != None:
    prefix = string.split(options.file, ".")[0]
    fortran_files = glob.glob(prefix + ".f90")
    fortran_files = fortran_files + glob.glob(prefix+".f")

if options.compiler == None:
    compiler = "gfortran"
else:
    compiler = options.compiler

# go through all the folders, make the python module and run the program
#   f2py_module_list = []
#   run_module_list = []
#
# Get all the fortran files
#
for item in fortran_files:
    prefix = string.split(item, ".")[0]
    print "\n=====================================================\n"
    print "  Attempting to compile file: %s " % item
    print "\n====================================================="
    ret = os.system('f2py -c --fcompiler=%s -m %s %s -DF2PY_REPORT_ON_ARRAY_COPY=1' % (compiler, prefix, item))
#   ret = os.system('f2py -c -m %s %s -DF2PY_REPORT_ON_ARRAY_COPY=1' % (prefix, item))
    if ret == 0:
        print "\n=====================================================\n"
        print "   Successfully compiled file: %s " % item
        print "   Object file is: %s " % (prefix + ".so")
        print "\n======================================================"
