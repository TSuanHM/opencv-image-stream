#if defined(_MSC_VER) && (_MSC_VER >= 1800)
// eliminating duplicated round() declaration
#define HAVE_ROUND 1
#pragma warning(push)
#pragma warning(disable:5033)  // 'register' is no longer a supported storage class
#endif
#include <math.h>
#include <Python.h>
#include <structmember.h>
#if defined(_MSC_VER) && (_MSC_VER >= 1800)
#pragma warning(pop)
#endif

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/ndarrayobject.h>

#if PY_MAJOR_VERSION >= 3
#  define CV_PYTHON_TYPE_HEAD_INIT() PyVarObject_HEAD_INIT(&PyType_Type, 0)
#else
#  define CV_PYTHON_TYPE_HEAD_INIT() PyObject_HEAD_INIT(&PyType_Type) 0,
#endif

#include <numpy/ndarrayobject.h>

#include <map>

#if PY_MAJOR_VERSION >= 3
// Python3 treats all ints as longs, PyInt_X functions have been removed.
#define PyInt_Check PyLong_Check
#define PyInt_CheckExact PyLong_CheckExact
#define PyInt_AsLong PyLong_AsLong
#define PyInt_AS_LONG PyLong_AS_LONG
#define PyInt_FromLong PyLong_FromLong
#define PyNumber_Int PyNumber_Long

// Python3 strings are unicode, these defines mimic the Python2 functionality.
#define PyString_Check PyUnicode_Check
#define PyString_FromString PyUnicode_FromString
#define PyString_FromStringAndSize PyUnicode_FromStringAndSize
#define PyString_Size PyUnicode_GET_SIZE

// PyUnicode_AsUTF8 isn't available until Python 3.3
#if (PY_VERSION_HEX < 0x03030000)
#define PyString_AsString _PyUnicode_AsString
#else
#define PyString_AsString PyUnicode_AsUTF8
#endif
#endif

#include <iostream>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/video.hpp>


#include <mutex>
#include <condition_variable>
#include <deque>
#include <thread>


#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>


using namespace cv;


struct ArgInfo
{
    const char * name;
    bool outputarg;
    // more fields may be added if necessary

    ArgInfo(const char * name_, bool outputarg_)
        : name(name_)
        , outputarg(outputarg_) {}

    // to match with older pyopencv_to function signature
    operator const char *() const { return name; }
};

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

class NumpyAllocator : public MatAllocator
{
public:
    NumpyAllocator() ;
    ~NumpyAllocator();
    UMatData* allocate(PyObject* o, int dims, const int* sizes, int type, size_t* step) const;
    UMatData* allocate(int dims0, const int* sizes, int type, void* data, size_t* step, int flags, UMatUsageFlags usageFlags) const ;
    bool allocate(UMatData* u, int accessFlags, UMatUsageFlags usageFlags) const ;
    void deallocate(UMatData* u) const CV_OVERRIDE;
    const MatAllocator* stdAllocator;
};


class AutoRef {
public:
  AutoRef(PyObject *o = nullptr) {
    this->o = o;
    if (o) {
      Py_INCREF(o);
    }
  }
  ~AutoRef() {
    if (o) {
      Py_XDECREF(o);
    }
  }
  AutoRef(AutoRef && r) {
      std::swap(o,r.o);
  }

  AutoRef& operator = (AutoRef&& r) {
      std::swap(o,r.o);
      return *this;
  }

  AutoRef& operator = (const AutoRef& r) =delete;
  AutoRef(const AutoRef & r) = delete;

  PyObject* get() {
    return o;
  }
protected:
  PyObject*  o{nullptr};
};

template <typename T>
class RingBuffer {
public:
    typedef std::lock_guard<std::mutex> MutexLockGuard ;

    RingBuffer(size_t cap = 100)
        : _mutex(),
          _notEmpty(),
          _queue(),
          _cap(cap)
    {
    }

    RingBuffer(const RingBuffer &) = delete;
    RingBuffer& operator=(const RingBuffer &) = delete;

    void put(T &&x)
    {
        {
            MutexLockGuard lock(_mutex);
            _queue.push_back(std::move(x));
            if (_queue.size() > _cap) {
                 T front(std::move(_queue.front()));
                  _queue.pop_front();
            }
        }
        _notEmpty.notify_one();
    }

    T get()
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _notEmpty.wait(lock, [this]{  return !this->_queue.empty(); });  
        assert(!_queue.empty());
        T front(std::move(_queue.front()));
        _queue.pop_front();
        return  front;
    }

    void get(T &x)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _notEmpty.wait(lock, [this]{  return !this->_queue.empty(); });  
        assert(!_queue.empty());
        x = std::move(_queue.front());
        _queue.pop_front();
    }

    size_t size() const 
    {
        MutexLockGuard lock(_mutex);
        return _queue.size();
    }

    void setCap(size_t cap) {
      _cap = cap;
    }
private:
    mutable std::mutex        _mutex;
    std::condition_variable   _notEmpty;
    std::deque<T>             _queue;
    size_t                    _cap;
};

class Rtsp {
public:
  Rtsp();
  ~Rtsp() ;
  void RtspServer();
  PyObject_HEAD
  GMainLoop           *loop{nullptr};
  GstRTSPServer       *server{nullptr};
  GstRTSPMountPoints  *mounts{nullptr};
  GstRTSPMediaFactory *factory{nullptr};
};

