#include "rtmp.h"

NumpyAllocator g_numpyAllocator;
BlockingQueue<PyObject*> g_blockingqueue;

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


void initialize_avformat_context(AVFormatContext *&fctx, const char *format_name)
{
  int ret = avformat_alloc_output_context2(&fctx, nullptr, format_name, nullptr);
  if (ret < 0) 
  {
    std::cout << "Could not allocate output format context!" << std::endl;
    exit(1);
  }
}

void initialize_io_context(AVFormatContext *&fctx, const char *output)
{
  if (!(fctx->oformat->flags & AVFMT_NOFILE))
  {
    int ret = avio_open2(&fctx->pb, output, AVIO_FLAG_WRITE, nullptr, nullptr);
    if (ret < 0)
    {
      std::cout << "Could not open output IO context!" << std::endl;
      exit(1);
    }
  }
}

void set_codec_params(AVFormatContext *&fctx, AVCodecContext *&codec_ctx, double width, double height, int fps)
{
  const AVRational dst_fps = {fps, 1};
  codec_ctx->codec_tag = 0;
  codec_ctx->codec_id = AV_CODEC_ID_H264;
  codec_ctx->codec_type = AVMEDIA_TYPE_VIDEO;
  codec_ctx->width = width;
  codec_ctx->height = height;
  codec_ctx->gop_size = 12;
  codec_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
  codec_ctx->framerate = dst_fps;
  codec_ctx->time_base = av_inv_q(dst_fps);
  if (fctx->oformat->flags & AVFMT_GLOBALHEADER)
  {
    codec_ctx->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
  }
}

void initialize_codec_stream(AVStream *&stream, AVCodecContext *&codec_ctx, AVCodec *&codec)
{
  int ret = avcodec_parameters_from_context(stream->codecpar, codec_ctx);
  if (ret < 0)
  {
    std::cout << "Could not initialize stream codec parameters!" << std::endl;
    exit(1);
  }

  AVDictionary *codec_options = nullptr;
  av_dict_set(&codec_options, "profile", "high422", 0);
  av_dict_set(&codec_options, "preset", "superfast", 0);
  av_dict_set(&codec_options, "tune", "zerolatency", 0);

  // open video encoder
  ret = avcodec_open2(codec_ctx, codec, &codec_options);
  if (ret < 0)
  {
    std::cout << "Could not open video encoder!" << std::endl;
    exit(1);
  }
}

SwsContext *initialize_sample_scaler(AVCodecContext *codec_ctx, double width, double height)
{
  SwsContext *swsctx = sws_getContext(width, height, AV_PIX_FMT_BGR24, width, height, codec_ctx->pix_fmt, SWS_BICUBIC, nullptr, nullptr, nullptr);
  if (!swsctx)
  {
    std::cout << "Could not initialize sample scaler!" << std::endl;
    exit(1);
  }
  return swsctx;
}

AVFrame *allocate_frame_buffer(AVCodecContext *codec_ctx, uint8_t *out_buf, double width, double height)
{
  AVFrame *frame = av_frame_alloc();
  av_image_fill_arrays(frame->data, frame->linesize, out_buf, codec_ctx->pix_fmt, width, height, 1);
  frame->width = width;
  frame->height = height;
  frame->format = static_cast<int>(codec_ctx->pix_fmt);
  return frame;
}

void write_frame(AVCodecContext *codec_ctx, AVFormatContext *fmt_ctx, AVFrame *frame)
{
  AVPacket pkt = {0};
  av_init_packet(&pkt);

  int ret = avcodec_send_frame(codec_ctx, frame);
  if (ret < 0)
  {
    std::cout << "Error sending frame to codec context!" << std::endl;
    exit(1);
  }

  ret = avcodec_receive_packet(codec_ctx, &pkt);
  if (ret < 0)
  {
    std::cout << "Error receiving packet from codec context!" << std::endl;
    exit(1);
  }

  av_interleaved_write_frame(fmt_ctx, &pkt);
  av_packet_unref(&pkt);
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

Rtmp::Rtmp(const std::string rtmp_server,
    int width, int hight, int fps, int bitrate,
    const std::string& h264_profile ) {
  // av_register_all();
  // avformat_network_init();

  // initialize_avformat_context(ofmt_ctx, "flv");
  // initialize_io_context(ofmt_ctx, rtmp_server.c_str());

  // out_codec = avcodec_find_encoder(AV_CODEC_ID_H264);
  // out_stream = avformat_new_stream(ofmt_ctx, out_codec);
  // out_codec_ctx = avcodec_alloc_context3(out_codec);

  // set_codec_params(ofmt_ctx, out_codec_ctx, width, height, fps, bitrate);
  // initialize_codec_stream(out_stream, out_codec_ctx, out_codec, h264_profile);

  // out_stream->codecpar->extradata = out_codec_ctx->extradata;
  // out_stream->codecpar->extradata_size = out_codec_ctx->extradata_size;

  // av_dump_format(ofmt_ctx, 0, rtmp_server.c_str(), 1);

  // swsctx = initialize_sample_scaler(out_codec_ctx, width, height);
  // frame = allocate_frame_buffer(out_codec_ctx, width, height);

  // if (avformat_write_header(ofmt_ctx, nullptr) < 0)
  // {
  //     failmsg("Could not write header!");
  //     exit(1);
  // }
}

Rtmp::~Rtmp() {
    // if (ofmt_ctx)
    //     av_write_trailer(ofmt_ctx);
    // if (&frame)
    //     av_frame_free(&frame);
    // if (out_codec_ctx)
    //     avcodec_close(out_codec_ctx);
    // if (ofmt_ctx && ofmt_ctx->pb)
    //     avio_close(ofmt_ctx->pb);
    // if (ofmt_ctx)
    //     avformat_free_context(ofmt_ctx);
}


#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>

int64_t Now() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (int64_t)tv.tv_sec*1000 + tv.tv_usec/1000;
}

void Rtmp::push(cv::Mat& image) {
    const int stride[] = {static_cast<int>(image.step[0])};
    sws_scale(swsctx, &image.data, stride, 0, image.rows, frame->data, frame->linesize);
    frame->pts += av_rescale_q(1, out_codec_ctx->time_base, out_stream->time_base);
    write_frame(out_codec_ctx, ofmt_ctx, frame);
}

void Rtmp::run() {
    printf("start push run thread!\n");
    while(1) {
      Mat mat;
      PyObject *o = g_blockingqueue.take();
      printf("queue size:%lu\n",g_blockingqueue.size());
      if (pyopencv_to(o, mat, ArgInfo("mat", 0))) {
          push(mat);
      } else {
           printf("pyopencv_to failed\n");
      }
      Py_XDECREF(o);
    }
}

static int Rtmp_init(Rtmp* Self, PyObject* pArgs)    //构造方法.
{
    const char* rtmp_server = "rtmp://localhost:1935/mytv/1";
    double width = 1920, height = 1080;
    int fps = 25;
    int bitrate = 6000000;
    const char*  h264_profile = "high422";

    if(!PyArg_ParseTuple(pArgs, "sdd", &rtmp_server, &width, &height,&fps,&bitrate,&h264_profile)) {
        failmsg("Parse the argument FAILED! You should pass correct values! init");
        return 0;
    }

    printf("%f %f\n",  width, height);

    av_register_all();
    avformat_network_init();

    initialize_avformat_context(Self->ofmt_ctx, "flv");
    initialize_io_context(Self->ofmt_ctx, rtmp_server);

    Self->out_codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    Self->out_stream = avformat_new_stream(Self->ofmt_ctx, Self->out_codec);
    Self->out_codec_ctx = avcodec_alloc_context3(Self->out_codec);

    set_codec_params(Self->ofmt_ctx, Self->out_codec_ctx, width, height, fps);
    initialize_codec_stream(Self->out_stream, Self->out_codec_ctx, Self->out_codec);

    Self->out_stream->codecpar->extradata = Self->out_codec_ctx->extradata;
    Self->out_stream->codecpar->extradata_size = Self->out_codec_ctx->extradata_size;

    av_dump_format(Self->ofmt_ctx, 0, rtmp_server, 1);

    Self->swsctx = initialize_sample_scaler(Self->out_codec_ctx, width, height);
    Self->out_buf =(uint8_t *)av_malloc(av_image_get_buffer_size(Self->out_codec_ctx->pix_fmt, width,height,1)); 
    Self->frame = allocate_frame_buffer(Self->out_codec_ctx, Self->out_buf, width, height);

    if (avformat_write_header(Self->ofmt_ctx, nullptr)) {
        exit(1);
    }
    std::thread t(&Rtmp::run, Self);
    t.detach();
    return 0;
}

static void Rtmp_Destruct(Rtmp* Self)                   //析构方法.
{
    printf("destructor\n");
    // if (Self->ofmt_ctx)
    //      av_write_trailer(Self->ofmt_ctx);
    // if (&Self->frame)
    //     av_frame_free(&Self->frame);
    // if (Self->out_codec_ctx)
    //     avcodec_close(Self->out_codec_ctx);
    // if (Self->ofmt_ctx && Self->ofmt_ctx->pb)
    //     avio_close(Self->ofmt_ctx->pb);
    // if (Self->ofmt_ctx)
    //     avformat_free_context(Self->ofmt_ctx);
    Py_TYPE(Self)->tp_free((PyObject*)Self);      //释放对象/实例.
}

static PyObject* Rtmp_Str(Rtmp* Self)             //调用str/print时自动调用此函数.
{
    return Py_BuildValue("s", "python rtmp");
}

static PyObject* Rtmp_Repr(Rtmp* Self)            //调用repr内置函数时自动调用.
{
    return Rtmp_Str(Self);
}


static PyObject* Rtmp_Push(Rtmp* Self, PyObject* Argvs)  
{  
   Py_INCREF(Py_None);  
   PyObject* pyobj_mat = nullptr;
   if(!PyArg_ParseTuple(Argvs, "O", &pyobj_mat))  
   {  
        failmsg("Parse the argument FAILED! You should pass correct values! push");
        return Py_None;  
   }  else {
          //printf("push ref : %p %ld \n",pyobj_mat, pyobj_mat->ob_refcnt);
          // Mat mat;
          // if (pyopencv_to(pyobj_mat, mat, ArgInfo("mat", 0))) {
          //    // Self->push(mat);
          //    g_blockingqueue.put(mat);
          // } else {
          //      printf("pyopencv_to failed\n");
          // }
          // printf("after ref : %p %ld \n",pyobj_mat, pyobj_mat->ob_refcnt);
          if (g_blockingqueue.size() > 20) {
            printf(" queue full");
            return Py_None;
          }
          Py_INCREF(pyobj_mat);  
          g_blockingqueue.put(pyobj_mat);
   }
   return Py_None;  
} 

static PyMemberDef Rtmp_DataMembers[] =        
{
    {"ofmt_ctx",     T_LONG,  offsetof(Rtmp, ofmt_ctx),   0, "The ofmt_ctx."},
    {"out_codec",    T_LONG,  offsetof(Rtmp, out_codec),    0, "The out_codec."},
    {"out_stream",   T_LONG,  offsetof(Rtmp, out_stream), 0, "The out_stream."},
    {"out_codec_ctx",T_LONG,  offsetof(Rtmp, out_codec_ctx),   0, "The out_codec_ctx"},
    {"frame",        T_LONG,  offsetof(Rtmp, frame),   0, "The frame"},
    {"swsctx",       T_LONG,  offsetof(Rtmp, swsctx),   0, "The swsctx"},
    {"out_buf",      T_LONG,  offsetof(Rtmp, out_buf),   0, "The ofmt_ctx."},
    {NULL, NULL, NULL, 0, NULL}
};


static PyMethodDef Rtmp_MethodMembers[] =      //类的所有成员函数结构列表.
{
   {"Push", (PyCFunction)Rtmp_Push, METH_VARARGS, "Push to Opencv Image to Rtmp Server."},
   {NULL, NULL, NULL, NULL}
};


////////////////////////////////////////////////////////////
// 类/结构的所有成员、内置属性的说明信息.
//
static PyTypeObject Rtmp_ClassInfo =
{
       PyVarObject_HEAD_INIT(NULL, 0)"rtmp.Rtmp",                 //可以通过__class__获得这个字符串. CPP可以用类.__name__获取.
       sizeof(Rtmp),                 //类/结构的长度.调用PyObject_New时需要知道其大小.
       0,
       (destructor)Rtmp_Destruct,    //类的析构函数.
       0,
       0,
       0,
       0,
       (reprfunc)Rtmp_Repr,          //repr 内置函数调用。
        0,
       0,
       0,
       0,
       0,
       (reprfunc)Rtmp_Str,          //Str/print内置函数调用.
       0,
       0,
       0,
       Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,                 //如果没有提供方法的话，为Py_TPFLAGS_DEFAULE
       "Rtmp Objects---Extensioned by C++!",                   //__doc__,类/结构的DocString.
       0,
       0,
       0,
       0,
       0,
       0,
       Rtmp_MethodMembers,        //类的所有方法集合.
       Rtmp_DataMembers,          //类的所有数据成员集合.
       0,
       0,
       0,
       0,
       0,
       0,
       (initproc)Rtmp_init,      //类的构造函数.
       0,
};


static PyMethodDef module_methods[] = {
    {NULL}  /* Sentinel */
};

#ifndef PyMODINIT_FUNC  /* declarations for DLL import/export */
#define PyMODINIT_FUNC void
#endif
PyMODINIT_FUNC initrtmp(void)     
{
    PyObject* pReturn = 0;
    Rtmp_ClassInfo.tp_new = PyType_GenericNew;      

    if(PyType_Ready(&Rtmp_ClassInfo) < 0) {
        printf("PyType_Ready Error\n");
        return;
    }

    pReturn = Py_InitModule3("rtmp", module_methods,
                       "rtmp module that creates an extension type.");

    if (pReturn == nullptr)
        return; 
    Py_INCREF(&Rtmp_ClassInfo);
    PyModule_AddObject(pReturn, "Rtmp", (PyObject*)&Rtmp_ClassInfo);

}

