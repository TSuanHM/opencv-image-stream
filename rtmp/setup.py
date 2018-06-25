#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension

setup(name="RtmpPush",
    ext_modules=[
    Extension("rtmp", ["Rtmp.cpp"],
    extra_compile_args=['-std=c++11','-ggdb','-Wno-deprecated-declarations','-Wno-write-strings','-Wno-conversion-null','-Wno-unused-function'],
    library_dirs=['/usr/local/lib'],
    libraries=['avdevice', 'avformat' ,'avfilter','avcodec','swscale','avutil','opencv_core','opencv_highgui','opencv_imgproc' ,'opencv_imgcodecs' ,'opencv_videoio','x264','pthread'])
])