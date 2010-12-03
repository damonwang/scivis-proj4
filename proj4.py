from __future__ import with_statement
import numpy as np
import cairo
from cairo import Context, ImageSurface, Pattern
import sline
from sline import sline, read2vecs, blinterpvec
import cnv

import os
from os import path

if __debug__: log = open('errlog', 'w')

class Options(dict):
    '''a thin dict wrapper that aliases getitem to getattribute'''

    def __getattr__(self, name):
        return self.__getitem__(name)

#-----------------------------------------------------------------------
# Functions for part 1
# from python itertools documentation b/c Techstaff doesn't have it installed
def product(*args, **kwds):
    # product('ABCD', 'xy') --> Ax Ay Bx By Cx Cy Dx Dy
    # product(range(2), repeat=3) --> 000 001 010 011 100 101 110 111
    pools = map(tuple, args) * kwds.get('repeat', 1)
    result = [[]]
    for pool in pools:
        result = [x+[y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)

def draw_arrow(ctx, path, arrowhead_size=1):
    ctx.move_to(*path[0])
    for point in path[1:]: ctx.line_to(*point)
    if path.shape[0] > 1:
        last = path[-1] - path[-2]
        last_norm = np.array([last[1], -last[0]])
        diag = last - last_norm
        diag /= np.sqrt((diag**2).sum())
        arrowhead = path[-1] - arrowhead_size * diag
        ctx.line_to(*arrowhead)
        assert np.dot(last_norm, last) == 0
    ctx.stroke()
    

def glyphs(*args, **kwargs):

    options = Options(infname='data/vorttest.txt', saveas='output/part1/',
            outfname='vorttest.png', scale=100., seed_num=20, stepsize=.01,
            steps=5, directions=1, norm=False, line_width=.5)
    options.update(kwargs)

    print >> log, options.infname

    if not path.exists(options.saveas): os.makedirs(options.saveas)

    (xmin, ymin), (xmax, ymax), uv = read2vecs(options.infname)

    width, height = xmax - xmin, ymax - ymin

    def index2world(points, xmin=xmin, ymin=ymin, scale=options.scale,
            debug=False):
        if debug: 
            print "index2world:",
            print xscale, yscale, points.dtype
        if debug: print points,
        points[:,0] -= xmin
        if debug: print points,
        points[:,1] -= ymin
        if debug: print points,
        points *= scale
        if debug: print points

    if 'seed' in options:
        seed = options.seed
    else: 
        seed = product(np.linspace(xmin, xmax, num=options.seed_num), 
                np.linspace(ymin, ymax, num=options.seed_num))


    ctx = Context(ImageSurface(cairo.FORMAT_ARGB32, int(options.scale * width),
        int(options.scale * height)))

    ctx.set_source_rgba(0,0,0)
    ctx.set_line_width(options.line_width)
    for s in seed:
        points = sline(uv, (xmin, ymin), (xmax, ymax), np.array(s),
            options.stepsize, options.steps, options.norm, options.directions)
        print >> log, points
        index2world(points)
        print >> log, points
        draw_arrow(ctx, points, arrowhead_size=2)

    with open(path.join(options.saveas, options.outfname), 'w') as outf:
        ctx.get_target().write_to_png(outf)

#-----------------------------------------------------------------------
# Functions for part 2

def magnitude(data):
    '''
    magnitude(ndarray data) -> ndarray magnitudes

    data has shape (...,2) and represents 2-vectors.
    
    uses Euclidean norm to calculate the vectors' magnitudes
    '''

    return np.sqrt(data[...,0]**2 + data[...,1]**2)

def vorticity(data, xmin, xmax, ymin, ymax, upsample_x=1, upsample_y=1):
    '''
    vorticity(ndarray data) -> ndarray vorticity

    data has shape (r, c, 2) and represents 2-vectors of velocity

    calculates vorticity using formula 
        vorticity = dvy/dx - dvx/dy
    '''

    assert data.ndim == 3

    bl = lambda xy: blinterpvec(data, (xmin, ymin), (xmax, ymax), xy, False)
    r, c, l = data.shape

    if upsample > 1:
        samples = np.array( [ bl((x, y)) 
            for x in np.linspace(xmin, xmax, num=upsample_x*c) 
            for y in np.linspace(ymin, ymax, num=upsample_y*r) ] )
    else: samples = data

    dx = cnv.x_partial(samples[...,1])
    dy = cnv.y_partial(samples[...,0])

    pixel_width = (xmax - xmin) / c
    pixel_height = (ymax - ymin) / r

    gradient = cnv.index_to_world(np.dstack((dx, dy)), cnv.M(pixel_height, pixel_width))

    return gradient[...,0] - gradient[...,1]

def color_mag(data):

    pass

def color_vorticity(*args, **kwargs):

    options = Options(infname='data/vorttest.txt', saveas='output/part2/',
            outfname='vorttest.png', scale=100.)
    options.update(kwargs)

    if not path.exists(options.saveas): os.makedirs(options.saveas)

    (xmin, ymin), (xmax, ymax), uv = read2vecs(options.infname)
    width, height = xmax - xmin, ymax - ymin
    r, c = uv.shape[:2]

#-----------------------------------------------------------------------
# Functions to generate output

def part1(*args, **kwargs):
    glyphs(infname='data/vorttest.txt', scale=100, seed_num=20, outfname='vorttest.png',
            **kwargs)
    glyphs(infname='data/synth.txt', scale=100, seed_num=20, outfname='synth.png', **kwargs)
    glyphs(infname='data/turbl.txt', scale=100, seed_num=20, outfname='turbl.png', **kwargs)


if __name__ == '__main__':
    part1()
