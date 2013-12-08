#!/usr/bin/env python

import os
import sys
import platform
import fnmatch

program = "run"

system_name = platform.system()

os.system("rm -f %s" % program)

modules_dir = ['src', 'src/util']
modules = list()
for d in modules_dir:
    for f in os.listdir(d):
        if fnmatch.fnmatch(f, "*.c"):
            fp = d + "/" + f
            modules.append(fp)
files_str = ' '.join(modules)

includes = ['src/util']
includes = map(lambda include: "-I" + include, includes)
include_str = ' '.join(includes)

if system_name == "Darwin":
	frameworks = []
        frameworks = map(lambda framework: "-framework " + framework, frameworks)
	framework_str =  ' '.join(frameworks)

if system_name == "Linux":
	compile_cmd = "gcc -Wall -o %s %s %s" % (program, include_str, files_str)
elif system_name == "Darwin":
	compile_cmd = "cc -g -Wall -o %s %s %s" % (program, include_str, files_str)


print compile_cmd
output = os.system(compile_cmd)
sys.exit(0)
