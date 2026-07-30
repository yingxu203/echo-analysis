# -*- coding: utf-8 -*-
# Derived from pfizer-opensource/mouse-echo-neural-net (https://github.com/pfizer-opensource/mouse-echo-neural-net),
# licensed under the Apache License, Version 2.0. This file has been modified from the original.
"""
util_preprocessing.py
Created on Thu Jan 14 13:36:35 2021

@author: MONTGM11
"""

import numpy as np
import glob
import pydicom
import os

def get_mode(filepath):
    ''' Function to read dicom header and detect m-mode vs b-mode'''
    data = pydicom.dcmread(filepath)
    mode = data.OperatingMode
    return mode

def move_files(parentDir,fileList,subName):
    '''Function to relocate list of files from parentDir to sub directory'''
    subDir = os.path.join(parentDir,subName)
    if not os.path.isdir(subDir):
        os.mkdir(subDir)
    for file in fileList:
        oldPath = os.path.join(parentDir,file)
        newPath = os.path.join(subDir,file)
        os.replace(oldPath,newPath)

def sort_modes(dirName):
    '''Function to sort all files within a directory into mmode and bmode subdirs'''
    
    # Get list of dcm files
    dcmFiles = glob.glob(dirName+'/*.dcm')
    
    # Check each file's mode
    modes = []
    mmode_files = []
    bmode_files = []
    for file in dcmFiles:
        mode = get_mode(file)
        modes.append(mode)
        if mode=='M-Mode':
            #Add just file name to mmode list
            mmode_files.append(os.path.basename(file))
        elif mode=='B-Mode':
            #Add just file name to bmmode list
            bmode_files.append(os.path.basename(file))
        else:
            #Do not analyze this file
            modes.append('Undefined')
    if len(mmode_files) + len(bmode_files) == 0:
        run_mode = 'None'
        return run_mode
                    
    
    # Check if multiple modes
    if len(np.unique(modes)) == 1:
        run_mode = modes[0]
    else:
        # Separate files into subfolders
        if 'M-Mode' in modes:
            move_files(dirName,mmode_files,'M-Mode')
        if 'B-Mode' in modes:
            move_files(dirName,bmode_files,'B-Mode')
        run_mode = 'Multiple'
        
    # Return info on which modes to run
    return run_mode

def write_command(inDir,runMode, testMode, verboseString, python_exe='python'):
    if testMode=='M-Mode':
        mainFunc = 'echoanalysis_mmode_main.py'
    elif testMode=='B-Mode':
        mainFunc = 'echoanalysis_main.py'
    else:
        return ''

    if runMode==testMode:
        subdir = inDir
    elif runMode=='Multiple' and os.path.isdir(os.path.join(inDir,testMode)):
        subdir = os.path.join(inDir,testMode)
    else:
        return ''

    return '"' + python_exe + '" ' + mainFunc + ' ' + verboseString + '--input_dir "' + subdir + '"'