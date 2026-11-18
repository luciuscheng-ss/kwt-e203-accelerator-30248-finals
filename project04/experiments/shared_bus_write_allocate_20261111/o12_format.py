"""Native signed12 FFN output and training-calibrated requantization."""
from hashlib import sha256
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SHAPE = (99, 256)
WORDS = 9504

def numeric_freeze():
    parts = [sha256((HERE/name).read_bytes()).hexdigest()
             for name in ('o12_format.py','o12_training_bounds.json')]
    return sha256(''.join(parts).encode()).hexdigest()

def o12_scales(config, input_scale, layer):
    knobs = json.loads((HERE/'o12_training_bounds.json').read_text())
    bound = np.array(knobs['per_layer_bounds'][layer])*knobs['selected_factor']/knobs['bound_stored_factor']
    gain = 2047*np.asarray(input_scale)[...,None]*np.asarray(config['weight_scale'])/bound
    shift = np.clip(np.floor(np.log2(15/gain)),0,38).astype(np.int64)
    mult = np.clip(np.floor(np.ldexp(gain,shift)+.5),1,15).astype(np.int64)
    return shift,mult

def encode_matrix(codes):
    q=np.asarray(codes,dtype=np.int64)
    if q.shape!=SHAPE or np.any(np.abs(q)>2047):
        raise ValueError('Expected 99x256 O12 codes in [-2047,2047]')
    v=(q.ravel()&4095).astype(np.uint32).reshape(-1,8)
    out=np.empty((v.shape[0],3),dtype=np.uint32)
    out[:,0]=v[:,0]|(v[:,1]<<12)|(v[:,2]<<24)
    out[:,1]=(v[:,2]>>8)|(v[:,3]<<4)|(v[:,4]<<16)|(v[:,5]<<28)
    out[:,2]=(v[:,5]>>4)|(v[:,6]<<8)|(v[:,7]<<20)
    return out.ravel()

def decode_matrix(words):
    a=np.asarray(words,dtype=np.uint32)
    if a.shape!=(WORDS,):
        raise ValueError('Expected 9504 words for a complete matrix')
    x,y,z=a.reshape(-1,3).T
    bits=np.stack([x,x>>12,(x>>24)|(y<<8),y>>4,y>>16,(y>>28)|(z<<4),z>>8,z>>20],axis=1)
    raw=(bits&4095).astype(np.int64).reshape(SHAPE)
    signed=np.where(raw>=2048,raw-4096,raw)
    if np.any(signed==-2048):
        raise ValueError('Reserved output code -2048')
    return signed

def decode_float(words,metadata):
    q=decode_matrix(words)
    return q*float(metadata['input_scale'])*np.asarray(metadata['weight_scale'])*np.exp2(np.asarray(metadata['R']))/np.asarray(metadata['M'])
