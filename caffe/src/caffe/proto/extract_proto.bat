set INCLUDE_PROTO_DIR=..\..\..\include\caffe\proto
set PYTHON_PROTO_DIR=..\..\..\Build\x64\Release\pycaffe\caffe\proto
protoc caffe.proto --cpp_out=./
protoc caffe.proto --python_out=./
echo ProtoCompile.cmd : Move newly generated caffe.pb.h to "%INCLUDE_PROTO_DIR%\caffe.pb.h"
move /y "caffe.pb.h" "%INCLUDE_PROTO_DIR%\caffe.pb.h"
move /y "caffe_pb2.py" "%PYTHON_PROTO_DIR%\caffe_pb2.py"
pause