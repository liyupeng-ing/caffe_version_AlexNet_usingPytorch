#autor: gukedream
#date : 2018-02-18
#This simple script is used to dump caffemodel weights to pth formats ,you should change the relative path according to your situation.
import torch
import sys
import os
import subprocess
import tempfile

caffe_pb2 = None

#to generate the 'caffe_pb2' profile which is necessary to interpret 'caffemodel'
def initialize(caffe_proto, codegen_dir = tempfile.mkdtemp(), shadow_caffe = True):
    global caffe_pb2
    if caffe_pb2 is None:
        local_caffe_proto = os.path.join(codegen_dir, os.path.basename(caffe_proto))
        with open(local_caffe_proto, 'w') as f:
            f.write(open(caffe_proto).read())
        subprocess.check_call(
        [protoc_path, 
        '--proto_path', os.path.dirname(local_caffe_proto), 
        '--python_out', codegen_dir, local_caffe_proto]
        )
        sys.path.insert(0, codegen_dir)      
        import caffe_pb2 as caffe_pb2
        sys.modules[__name__ + '.proto'] = sys.modules[__name__]
        if shadow_caffe:
            sys.modules['caffe'] = sys.modules[__name__]
            sys.modules['caffe.proto'] = sys.modules[__name__]
        return caffe_pb2

if __name__ == '__main__':
    #change the path here to your tool 'protoc'
    protoc_path = 'C:/ProgramData/Anaconda3/Lib/site-packages/protoc-3.0.0-win32/bin/protoc'
    #you should also change here according to your situation
    model_caffemodel = 'C:/Users/lypeng/AnacondaProjects/caffemodel/bvlc_reference_caffenet.caffemodel'
    output_path      = 'C:/Users/lypeng/AnacondaProjects/caffemodel/'
    caffe_proto      = 'C:/Users/lypeng/AnacondaProjects/caffemodel/caffe.proto'
    output_path      = model_caffemodel + '.pth'

    net_param = initialize(caffe_proto).NetParameter()
    net_param.ParseFromString(open(model_caffemodel,'rb').read()) #read the file using 'b' mode
    blobs = {layer.name + '.' + name : dict(data = blob.data, shape = list(blob.shape.dim)
    if len(blob.shape.dim) > 0 else [blob.num, blob.channels, blob.height, blob.width]) 
                                    for layer in list(net_param.layer) + list(net_param.layers) 
                                    for name, blob in zip(['weight', 'bias'], layer.blobs)}

    torch.save({k : torch.FloatTensor(blob['data']).view(*blob['shape'])
                                for k, blob in blobs.items()}, output_path)
