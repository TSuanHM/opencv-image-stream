#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension

setup(name="RtspPush",
    ext_modules=[
    Extension("rtsp", ["rtsp.cpp"],
    extra_compile_args=['-std=c++11','-ggdb','-Wno-deprecated-declarations','-Wno-write-strings','-Wno-conversion-null','-Wno-unused-function'],
    include_dirs=['/usr/include/gstreamer-1.0','/usr/include/glib-2.0/','/usr/lib/x86_64-linux-gnu/glib-2.0/include/','/usr/lib/x86_64-linux-gnu/gstreamer-1.0/include'],
    library_dirs=['/usr/local/lib','/usr/lib/x86_64-linux-gnu/'],
    libraries=['opencv_core','opencv_highgui','opencv_imgproc' ,'opencv_imgcodecs' ,'opencv_videoio','x264','glib-2.0','gstreamer-1.0','gstrtspserver-1.0'])
])