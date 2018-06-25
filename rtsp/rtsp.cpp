#include "Rtsp.h"
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

int64_t Now() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (int64_t)tv.tv_sec*1000 + tv.tv_usec/1000;
}


NumpyAllocator g_numpyAllocator;
RingBuffer<AutoRef> g_rbuff;

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

NumpyAllocator::NumpyAllocator() { stdAllocator = Mat::getStdAllocator(); }
NumpyAllocator::~NumpyAllocator() {}

UMatData* NumpyAllocator::allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const
{
    UMatData* u = new UMatData(this);
    u->data = u->origdata = (uchar*)PyArray_DATA((PyArrayObject*) o);
    npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
    for( int i = 0; i < dims - 1; i++ )
        step[i] = (size_t)_strides[i];
    step[dims-1] = CV_ELEM_SIZE(type);
    u->size = sizes[0]*step[0];
    u->userdata = o;
    return u;
}

UMatData* NumpyAllocator::allocate(int dims0, const int* sizes, int type, void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const 
{
    if( data != 0 )
    {
        // issue #6969: CV_Error(Error::StsAssert, "The data should normally be NULL!");
        // probably this is safe to do in such extreme case
        return stdAllocator->allocate(dims0, sizes, type, data, step, flags, usageFlags);
    }
    PyEnsureGIL gil;

    int depth = CV_MAT_DEPTH(type);
    int cn = CV_MAT_CN(type);
    const int f = (int)(sizeof(size_t)/8);
    int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
    depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
    depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
    depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
    int i, dims = dims0;
    cv::AutoBuffer<npy_intp> _sizes(dims + 1);
    for( i = 0; i < dims; i++ )
        _sizes[i] = sizes[i];
    if( cn > 1 )
        _sizes[dims++] = cn;
    PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
    if(!o)
        CV_Error_(Error::StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
    return allocate(o, dims0, sizes, type, step);
}

bool NumpyAllocator::allocate(UMatData* u, int accessFlags, UMatUsageFlags usageFlags) const 
{
    return stdAllocator->allocate(u, accessFlags, usageFlags);
}

void NumpyAllocator::deallocate(UMatData* u) const 
{
    if(!u)
        return;
    PyEnsureGIL gil;
    CV_Assert(u->urefcount >= 0);
    CV_Assert(u->refcount >= 0);
    if(u->refcount == 0)
    {
        PyObject* o = (PyObject*)u->userdata;
        Py_XDECREF(o);
        //printf("after: %ld\n",o->ob_refcnt);
        delete u;
    }
}

static bool pyopencv_to(PyObject* o, Mat& m, const ArgInfo info)
{
    bool allowND = true;
    if(!o || o == Py_None)
    {
        if( !m.data )
            m.allocator = &g_numpyAllocator;
        return true;
    }

    PyArrayObject* oarr = (PyArrayObject*) o;

    bool needcopy = false, needcast = false;
    int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
    int type = typenum == NPY_UBYTE ? CV_8U :
               typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U :
               typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT ? CV_32S :
               typenum == NPY_INT32 ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;


    if( type < 0 )
    {
        if( typenum == NPY_INT64 || typenum == NPY_UINT64 || typenum == NPY_LONG )
        {
            needcopy = needcast = true;
            new_typenum = NPY_INT;
            type = CV_32S;
        }
        else
        {
            failmsg("%s data type = %d is not supported", info.name, typenum);
            return false;
        }
    }

#ifndef CV_MAX_DIM
    const int CV_MAX_DIM = 32;
#endif

    int ndims = PyArray_NDIM(oarr);
    if(ndims >= CV_MAX_DIM)
    {
        failmsg("%s dimensionality (=%d) is too high", info.name, ndims);
        return false;
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1];
    size_t elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(oarr);
    const npy_intp* _strides = PyArray_STRIDES(oarr);
    bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

    for( int i = ndims-1; i >= 0 && !needcopy; i-- )
    {
        // these checks handle cases of
        //  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
        //  b) transposed arrays, where _strides[] elements go in non-descending order
        //  c) flipped arrays, where some of _strides[] elements are negative
        // the _sizes[i] > 1 is needed to avoid spurious copies when NPY_RELAXED_STRIDES is set
        if( (i == ndims-1 && _sizes[i] > 1 && (size_t)_strides[i] != elemsize) ||
            (i < ndims-1 && _sizes[i] > 1 && _strides[i] < _strides[i+1]) )
            needcopy = true;
    }

    if( ismultichannel && _strides[1] != (npy_intp)elemsize*_sizes[2] )
        needcopy = true;

    if (needcopy)
    {
        if (info.outputarg)
        {
            failmsg("Layout of the output array %s is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)", info.name);
            return false;
        }

        if( needcast ) {
            o = PyArray_Cast(oarr, new_typenum);
            oarr = (PyArrayObject*) o;
        }
        else {
            oarr = PyArray_GETCONTIGUOUS(oarr);
            o = (PyObject*) oarr;
        }

        _strides = PyArray_STRIDES(oarr);
    }

    // Normalize strides in case NPY_RELAXED_STRIDES is set
    size_t default_step = elemsize;
    for ( int i = ndims - 1; i >= 0; --i )
    {
        size[i] = (int)_sizes[i];
        if ( size[i] > 1 )
        {
            step[i] = (size_t)_strides[i];
            default_step = step[i] * size[i];
        }
        else
        {
            step[i] = default_step;
            default_step *= size[i];
        }
    }

    // handle degenerate case
    if( ndims == 0) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if( ismultichannel )
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if( ndims > 2 && !allowND )
    {
        failmsg("%s has more than 2 dimensions", info.name);
        return false;
    }

    m = Mat(ndims, size, type, PyArray_DATA(oarr), step);
    m.u = g_numpyAllocator.allocate(o, ndims, size, type, step);
    m.addref();

    if( !needcopy )
    {
        Py_INCREF(o);
    }
    m.allocator = &g_numpyAllocator;
    return true;
}

struct MyContext {
  GstClockTime        timestamp;
  Rtsp                *server;
};

static void need_data_rtsp (GstElement * appsrc, guint unused, MyContext *ctx)
{
    Mat mat;
    {
      AutoRef ar = g_rbuff.get();
      bool isok = pyopencv_to(ar.get(), mat, ArgInfo("mat", 0));
      if (!isok) {
           g_print("pyopencv_to failed\n");
           return;
      }
    } 
    GstBuffer *buffer = 0 ;
    GstFlowReturn ret = GST_FLOW_OK;
    guint size = mat.rows*mat.cols*mat.channels();
    buffer = gst_buffer_new_allocate (NULL, size, NULL);
    GstMapInfo map;
    gst_buffer_map(buffer,&map,GST_MAP_READ);
    memcpy(map.data,mat.data,size);

    GST_BUFFER_PTS (buffer) = ctx->timestamp;
    GST_BUFFER_DURATION (buffer) = gst_util_uint64_scale_int (1, GST_SECOND, 25);
    ctx->timestamp += GST_BUFFER_DURATION (buffer) ;

    g_signal_emit_by_name (appsrc, "push-buffer", buffer, &ret);
    gst_buffer_unmap(buffer,&map);
    gst_buffer_unref (buffer);
}

static void media_configure (GstRTSPMediaFactory * factory, GstRTSPMedia * media, Rtsp *server)
{
    GstElement *element = 0 , *appsrc = 0;
    MyContext *ctx = 0;
    element = gst_rtsp_media_get_element (media);
    appsrc = gst_bin_get_by_name_recurse_up (GST_BIN (element), "mysrc");

   // gst_util_set_object_arg (G_OBJECT (appsrc), "format", "time");

     g_object_set (G_OBJECT (appsrc),
      "stream-type", 0,
      "format", GST_FORMAT_TIME, NULL);

    g_object_set (G_OBJECT (appsrc), "caps",
        gst_caps_new_simple ("video/x-raw",
            "format", G_TYPE_STRING, "BGR",
            "width", G_TYPE_INT, 1920,
            "height", G_TYPE_INT, 1080,
            "framerate", GST_TYPE_FRACTION, 25, 1, NULL), NULL);

    ctx = g_new0 (MyContext, 1);
    ctx->server = server;
    ctx->timestamp = 0;
    g_object_set_data_full (G_OBJECT (media), "my-extra-data", ctx,
      (GDestroyNotify) g_free);

    g_signal_connect (appsrc, "need-data", (GCallback) need_data_rtsp, ctx);
    gst_object_unref (appsrc);
    gst_object_unref (element);
}

void Rtsp::RtspServer()
{
    g_print("RtspServer loop \n");
    gst_init (NULL, NULL);
    loop = g_main_loop_new (NULL, FALSE);
    server = gst_rtsp_server_new ();
    gst_rtsp_server_set_service (server,"8556");
    //gst_rtsp_server_set_address (server,"192.168.1.164");
    mounts = gst_rtsp_server_get_mount_points (server);
    factory = gst_rtsp_media_factory_new ();
    gst_rtsp_media_factory_set_launch (factory,
      "( appsrc name=mysrc ! videoconvert ! video/x-raw, format=I420 ! x264enc speed-preset=ultrafast tune=zerolatency ! rtph264pay name=pay0 pt=96 )");

    g_signal_connect (factory, "media-configure", (GCallback) media_configure, this);
    gst_rtsp_mount_points_add_factory (mounts, "/test", factory);
    g_object_unref (mounts);
    gst_rtsp_server_attach (server, NULL);
    g_main_loop_run (loop);
}

Rtsp::Rtsp() {
}

Rtsp::~Rtsp() {

}
static int Rtsp_init(Rtsp* Self, PyObject* pArgs)    //构造方法.
{
    // unsigned int width = 1920, height = 1080;
    // unsigned int fps = 25;
    // unsigned int port = 8556;
    // unsigned int latency = 5; //second


    // // if(!PyArg_ParseTuple(pArgs, "IIII", &width,&height,&port,&fps)) {
    // //     failmsg("Parse the argument FAILED! You should pass correct values! init");
    // //     return 0;
    // // }
    g_rbuff.setCap(5*25);
    std::thread t(&Rtsp::RtspServer, Self);
    t.detach();
    return 0;
}

static void Rtsp_Destruct(Rtsp* Self)                   //析构方法.
{
    Py_TYPE(Self)->tp_free((PyObject*)Self);      //释放对象/实例.
}

static PyObject* Rtsp_Str(Rtsp* Self)             //调用str/print时自动调用此函数.
{
    return Py_BuildValue("s", "python Rtsp");
}

static PyObject* Rtsp_Repr(Rtsp* Self)            //调用repr内置函数时自动调用.
{
    return Rtsp_Str(Self);
}

static PyObject* Rtsp_Push(Rtsp* Self, PyObject* Argvs)  
{  
   Py_INCREF(Py_None);  
   PyObject* o = nullptr;
   if(!PyArg_ParseTuple(Argvs, "O", &o)) {  
        failmsg("Parse the argument FAILED! You should pass correct values! push");
        return Py_None;  
   }  else {
        g_rbuff.put(AutoRef(o));
        //g_print("g_rbuff size:%lu\n",g_rbuff.size());
   }
   return Py_None;  
} 

static PyMemberDef Rtsp_DataMembers[] =        
{
    {"loop",   T_LONG,  offsetof(Rtsp, loop),   0, "The loop."},
    {"server", T_LONG,  offsetof(Rtsp, server), 0, "The server."},
    {"mounts", T_LONG,  offsetof(Rtsp, mounts), 0, "The mounts."},
    {"factory",T_LONG,  offsetof(Rtsp, factory),0, "The factory"},
    {NULL, NULL, NULL, 0, NULL}
};

static PyMethodDef Rtsp_MethodMembers[] =      //类的所有成员函数结构列表.
{
   {"Push", (PyCFunction)Rtsp_Push, METH_VARARGS, "Push to Opencv Image to Rtsp Server."},
   {NULL, NULL, NULL, NULL}
};


////////////////////////////////////////////////////////////
// 类/结构的所有成员、内置属性的说明信息.
//
static PyTypeObject Rtsp_ClassInfo =
{
       PyVarObject_HEAD_INIT(NULL, 0)"Rtsp.Rtsp",                 //可以通过__class__获得这个字符串. CPP可以用类.__name__获取.
       sizeof(Rtsp),                 //类/结构的长度.调用PyObject_New时需要知道其大小.
       0,
       (destructor)Rtsp_Destruct,    //类的析构函数.
       0,
       0,
       0,
       0,
       (reprfunc)Rtsp_Repr,          //repr 内置函数调用。
        0,
       0,
       0,
       0,
       0,
       (reprfunc)Rtsp_Str,          //Str/print内置函数调用.
       0,
       0,
       0,
       Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                 //如果没有提供方法的话，为Py_TPFLAGS_DEFAULE
       "Rtsp Objects---Extensioned by C++!",                   //__doc__,类/结构的DocString.
       0,
       0,
       0,
       0,
       0,
       0,
       Rtsp_MethodMembers,        //类的所有方法集合.
       Rtsp_DataMembers,          //类的所有数据成员集合.
       0,
       0,
       0,
       0,
       0,
       0,
       (initproc)Rtsp_init,      //类的构造函数.
       0,
};


static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initrtsp(void)     
{
    PyObject* pReturn = 0;
    Rtsp_ClassInfo.tp_new = PyType_GenericNew;      

    if(PyType_Ready(&Rtsp_ClassInfo) < 0) {
        printf("PyType_Ready Error\n");
        return;
    }

    pReturn = Py_InitModule3("rtsp", module_methods,
                       "Rtsp module that creates an extension type.");

    if (pReturn == nullptr)
        return; 
    Py_INCREF(&Rtsp_ClassInfo);
    PyModule_AddObject(pReturn, "Rtsp", (PyObject*)&Rtsp_ClassInfo);

}

