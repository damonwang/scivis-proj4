
import numpy as np
import math
import random

# linear interpolation
def lerp(a,lo,hi):
  return (1-a)*lo + a*hi

# as v varies between i and I, the output varies between o and O
def affine(i,v,I,o,O):
  return (1.0*O-o)*(1.0*v-i)/(1.0*I-i) + 1.0*o

# bilinear interpolation in vector field
def blinterpvec(uv, xymin, xymax, xy, norm):
  sy = uv.shape[0]
  sx = uv.shape[1]
  xidx = affine(xymin[0], xy[0], xymax[0], 0, sx-1)
  yidx = affine(xymin[1], xy[1], xymax[1], 0, sy-1)
  xi = int(xidx)
  yi = int(yidx)
  xi -= (xi == sx-1)
  yi -= (yi == sy-1)
  xf = xidx - xi
  yf = yidx - yi
  y0 = lerp(xf, uv[yi  ,xi,:], uv[yi  ,xi+1,:])
  y1 = lerp(xf, uv[yi+1,xi,:], uv[yi+1,xi+1,:])
  ret = lerp(yf, y0, y1)
  if norm:
    len = math.sqrt(ret[0]*ret[0] + ret[1]*ret[1])
    if len:
      ret /= len
    else:
      ret[:] = (random.gauss(0, 1),random.gauss(0, 1))
      ret /= math.sqrt(ret[0]*ret[0] + ret[1]*ret[1])
  return ret

# read vectors from file
def read2vecs(fname, optimise=True):
  ff = open(fname)
  line = ff.readline()
  while '#' == line[0]:
    line = ff.readline()
  flds = line.strip().split()
  if not '2vecs' == flds[0]:
    raise ValueError, '''Didn't see "2vecs" at beginning of line after comments'''
  sizex = int(flds[1])
  sizey = int(flds[2])
  xymin = (float(flds[3]),float(flds[4]))
  xymax = (float(flds[5]),float(flds[6]))
  if optimise:
      uv = np.array([ i.split() for i in ff ], dtype='float64')
  else:
      uv = np.array([float(j) for i in ff.readlines() for j in i.split()])
  uv.shape = sizey,sizex,2
  return xymin,xymax,uv

# compute streamlines in vector field uv (bounded by xymin
# and xymax), starting from seed, with step size h, taking
# S steps forward and backward.  vectors are normalized 
# prior to integration if norm is true.
def sline(uv, xymin, xymax, seed, h, S, norm, direction=2):
  bl = lambda pos: blinterpvec(uv, xymin, xymax, pos, norm)
  xrng = [min(xymin[0], xymax[0]), max(xymin[0], xymax[0])]
  yrng = [min(xymin[1], xymax[1]), max(xymin[1], xymax[1])]
  path = np.zeros([2*S+1,2])
  path[S,:] = seed.copy()
  sign = [1, -1]
  steps = [0,0]  # diridx=0: forward, 1: backward
  apos = seed.copy()
  bpos = seed.copy()
  for diridx in range(direction):
    ss = sign[diridx]
    apos[:] = seed[:]
    for stepidx in range(S):
      dir = ss*bl(apos)
      # RK2 integration
      bpos = apos + (h/2)*dir
      if (bpos[0] <= xrng[0] or xrng[1] <= bpos[0] \
            or bpos[1] <= yrng[0] or yrng[1] <= bpos[1]):
        stepidx -= 1
        break
      dir = ss*bl(bpos)
      apos += h*dir
      if (apos[0] <= xrng[0] or xrng[1] <= apos[0] \
            or apos[1] <= yrng[0] or yrng[1] <= apos[1]):
        stepidx -= 1
        break
      path[S+ss*(stepidx+1),:] = apos
    steps[diridx] = stepidx+1
  return path[S-steps[1]:S+steps[0]+1,:]
