# Copyright 2017-2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
import ast
import argparse
import logging
import json
import os
import glob
from io import BytesIO

import torch
import torch.nn.functional as F

import fastai
from fastai import *
from fastai.vision import *


torch.backends.cudnn.benchmark=True
import torch.distributed.deprecated as dist
from torch.utils.data.distributed import DistributedSampler

from models import *
from loss import TransferLoss
from data import ContentStyleLoader, InputDataset, SimpleDataBunch

# setup the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# set the constants for the content types
JSON_CONTENT_TYPE = 'application/json'
JPEG_CONTENT_TYPE = 'image/jpeg'

# The train method
def _train(args):
    print(f'Called _train method with model arch: {args.model_arch}, batch size: {args.batch_size}, image size: {args.image_size}, epochs: {args.epochs}')
    print(f'Getting training data from dir: {args.data_dir}')
    
    # Create models
    data = _load_data(args)
    print('Loaded data')

    # Create models
    mt = StyleTransformer()
    ms = StylePredict.create_resnet() if args.resnet else StylePredict.create_inception()
    m_com = CombinedModel(mt, ms).cuda()
    m_vgg = VGGActivations().cuda()

    print('Created models')

    # Training
    opt_func = partial(Adam, betas=(0.9,0.999), weight_decay=1e-3)
    st_wgt = 2.5e9
    ct_wgt = 5e2    
    tva_wgt = 1e-6
    st_block_wgts = [1,80,200,5] # 2,3,4,5
    c_block = 1 # 1=3

    style_phases = [(1,1e2,st_wgt*2),(1,st_wgt,st_wgt/2)]*2 + [(epochs,st_wgt,st_wgt)]
    cont_phases = [(1,ct_wgt,ct_wgt/2),(1,ct_wgt,ct_wgt*2)]*2 + [(epochs,ct_wgt,ct_wgt)]

    loss_func = TransferLoss(m_vgg, ct_wgt, st_wgt, st_block_wgts, tva_wgt, c_block)

    learner = Learner(data, m_com, opt_func=opt_func, loss_func=loss_func)
    w_sched = partial(WeightScheduler, loss_func=loss_func, cont_phases=cont_phases, style_phases=style_phases)
    recorder = partial(DistributedRecorder, save_path=args.save, print_freq=args.print_freq)
    learner.callback_fns = [recorder, w_sched]

    print('Begin training')
    learner.fit_one_cycle(args.epochs, 5e-5)

    print('Saving model weights to dir: {args.model_dir}')
    # learn.save(Path(args.model_dir))
    trace_data = torch.ones((1,3,64,64))
    traced_model = torch.jit.trace(m_com, (trace_data, trace_data))
    torch.jit.save(traced_model, Path(args.model_dir)/'test.pth')


def _load_data(args):
    # Content Data
    content_files = get_files(args.content_dir, recurse=True)
    style_files = get_files(args.style_dir, recurse=True)

    train_dl = get_data(content_files, style_files, size=args.image_size, cont_bs=args.batch_size)
    return SimpleDataBunch(train_dl, MODEL_PATH)

# Return the Convolutional Neural Network model
def model_fn(model_dir):
    logger.debug('model_fn')
    print('Model_fn called')
    model_path = Path(args.model_dir)/'test.pth'
    if not model_path.exists(): 
        download_url('https://s3-us-west-2.amazonaws.com/ashaw-fastai-imagenet/fastai_style_lowerlr_8_jit.pth', model_path)
    jit_model = torch.jit.load()
    return jit_model

# Deserialize the Invoke request body into an object we can perform prediction on
def input_fn(request_body, content_type=JPEG_CONTENT_TYPE):
    logger.info('Deserializing the input data.')
    print('Input fn')
    if content_type == JPEG_CONTENT_TYPE:
        con_fn = _tfm_img(BytesIO(request_body['content']))
        style_fn = _tfm_img(BytesIO(request_body['style']))
    elif content_type == JSON_CONTENT_TYPE:
        con_fn = download_image(con_url, 'content.jpg')
        style_fn = download_image(style_url, 'style.jpg')
    else: raise Exception('Requested unsupported ContentType in content_type: {}'.format(content_type))
    x_style = _tfm_img(style_fn).unsqueeze(0)
    x_cont = _tfm_img(con_fn).unsqueeze(0)
    return x_cont, x_style

def _tfm_img(img_fn):
    img = open_image(img_fn)
    return data_norm((apply_tfms(style_tds.tfms, img.resize(size)).px,None))[0]

# Perform prediction on the deserialized object, with the loaded model
def predict_fn(input_object, model):
    logger.info("Calling model")
    x_con, x_style = input_object
    print('Predicting')
    with torch.no_grad(): 
        out = model(x_con, x_style)
    print('Predicted')
    return Image((out[idx].detach().cpu())).px

# Serialize the prediction result into the desired response content type
def output_fn(prediction, accept=JPEG_CONTENT_TYPE):        
    logger.info('Serializing the generated output.')
    # if accept == JSON_CONTENT_TYPE:
    #     return json.dumps(prediction), accept
    if accept == JPEG_CONTENT_TYPE:
        return json.dumps(prediction), accept
    raise Exception('Requested unsupported ContentType in Accept: {}'.format(accept))    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--workers', type=int, default=2, metavar='W',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--epochs', type=int, default=2, metavar='E',
                        help='number of total epochs to run (default: 2)')
    parser.add_argument('--batch-size', type=int, default=64, metavar='BS',
                        help='batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='initial learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--dist-backend', type=str, default='gloo', help='distributed backend (default: gloo)')

    # fast.ai specific parameters
    parser.add_argument('--image-size', type=int, default=224, metavar='IS',
                        help='image size (default: 224)')
    parser.add_argument('--model-arch', type=str, default='resnet34', metavar='MA',
                        help='model arch (default: resnet34)')
    
    # The parameters below retrieve their default values from SageMaker environment variables, which are
    # instantiated by the SageMaker containers framework.
    # https://github.com/aws/sagemaker-containers#how-a-script-is-executed-inside-the-container
    parser.add_argument('--hosts', type=str, default=ast.literal_eval(os.environ['SM_HOSTS']))
    parser.add_argument('--current-host', type=str, default=os.environ['SM_CURRENT_HOST'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAINING'])
    parser.add_argument('--content-dir', type=str, default=os.environ['SM_CHANNEL_CONTENT'])
    parser.add_argument('--style-dir', type=str, default=os.environ['SM_CHANNEL_STYLE'])
    parser.add_argument('--num-gpus', type=int, default=os.environ['SM_NUM_GPUS'])

    _train(parser.parse_args())
