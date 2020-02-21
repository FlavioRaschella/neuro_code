#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 14:32:39 2020

@author: raschell

This library manages the python modules in the system
"""

import subprocess
import sys
import pkg_resources

def check_if_module_exist(package):
    installed = {pkg.key for pkg in pkg_resources.working_set}
    if package not in installed:
        print('Package {} is missing from the system. Installing...'.format(package))
        try:
            install(package)
        except:
            raise Exception('This package does not exist! Re-check the name {}'.format(package.upper()))    

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
