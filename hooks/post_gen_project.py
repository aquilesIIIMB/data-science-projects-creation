#!/usr/bin/env python

# Any post-generation processing can happen here.  See docs for more info: https://cookiecutter.readthedocs.io/en/2.0.1/advanced/hooks.html

import os

projectDir = os.path.realpath(os.path.curdir)

print("Executing post-generation hook in %s" % (projectDir))
