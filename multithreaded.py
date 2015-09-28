import sys
import multiprocessing as mp
import scipy.ndimage as nd
import numpy as np
import scipy.misc as misc
import itertools as it

# python multithreaded.py manzanas.jpg samples.png multi-out.png
# python multithreaded.py path-imagen path-sample path-salida

CORES = 4
SAMPLES = 10


def buildimage(img, results):
    dim = np.shape(img)
    r = map(lambda x: x.get(), results)
    r = np.concatenate(r, axis=0)
    return r.reshape(dim[0], dim[1], dim[2])


def buildslices(img):
    dim = np.shape(img)
    l = reduce(lambda p, x: x*p,  dim[:2])
    img = img.reshape(l, 1, dim[2])
    s = l/CORES
    return [img[i*s:(i+1)*s] for i in range(CORES)]


def runslices(slices, samples):
    pool = mp.Pool(processes=CORES)
    results = [pool.apply_async(knn, [s, samples]) for s in slices]
    return results


def rgbdistance(pixel, region):
    return map(lambda x: sum(np.power(x - pixel[0], 2)), region)


def getpixel(l):
    if l == 'r' or l == 0:
        return np.array([255, 0, 0])
    elif l == 'g' or l == 1:
        return np.array([0, 255, 0])
    else:
        return np.array([255, 255, 255])


def knn(img, samples):
    dim = np.shape(img)
    r = np.zeros((dim[0], dim[2]))
    for i in xrange(dim[0]):
        r[i] = step(img, samples, i)
    return r


# TODO usar np.append en cambio de itertools, con itertools toca copiar la lista.
# Usar axis=0 para que no aplane los arrays.
def step(img, rs, i):
    tmp = map(lambda x:
              map(lambda z: [x, z],
                  rgbdistance(img[i, :], rs[x])), rs.keys())
    l = list(it.chain(*tmp))
    l.sort(key=lambda t: t[1])
    counter = np.zeros(3)
    for k in xrange(SAMPLES):
        counter = counter + getpixel(l[k][0])
    counter = counter - np.array([counter[2], counter[2], 0])
    return getpixel(counter.argmax())


def readsamples(samples, img):
    dim = np.shape(img)
    results = {'r': [], 'g': [], 'b': []}
    for i in xrange(dim[0]):
        for j in xrange(dim[1]):
            if (samples[i, j] == [255, 0, 0]).all():
                results['r'].append(img[i, j])
            elif (samples[i, j] == [0, 255, 0]).all():
                results['g'].append(img[i, j])
            elif (samples[i, j] == [0, 0, 0]).all():
                results['b'].append(img[i, j])
    return results


def test():
    img = nd.imread(sys.argv[1]).astype("int")
    samples = nd.imread(sys.argv[2]).astype("int")
    samples = readsamples(samples, img)
    img = buildimage(img, runslices(buildslices(img), samples))
    misc.imsave(sys.argv[3], img)

test()
