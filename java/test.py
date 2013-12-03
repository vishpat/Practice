#!/usr/bin/env python

import os

def run_cmd(cmd):
    print cmd
    os.system(cmd)

run_cmd("ant dist && export CLASSPATH=$CLASSPATH:./build;$JAVA_HOME/bin/java com.vishpat.backtracing.Queens")

