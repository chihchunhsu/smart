import numpy as np
from numpy.linalg import inv, det

################################################################

def bilinear_interpolation(x, y, points):
    '''Interpolate (x,y) from values associated with four points.

    The four points are a list of four triplets:  (x, y, value).
    The four points can be in any order.  They should form a rectangle.

        >>> bilinear_interpolation(12, 5.5,
        ...                        [(10, 4, 100),
        ...                         (20, 4, 200),
        ...                         (10, 6, 150),
        ...                         (20, 6, 300)])
        165.0

    '''
    # See formula at:  http://en.wikipedia.org/wiki/Bilinear_interpolation

    (x0, y0, q00), (x0, y1, q01), (x1, y0, q10), (x1, y1, q11) = points

    # Check if we just need linear interpolation!
    if x0 == x1:
        interpFlux = 10**( ( q00 * ( y1 - y ) + q11 * ( y - y0 ) ) / ( ( y1 - y0 ) ) )
        return interpFlux

    if y0 == y1:
        interpFlux = 10**( ( q00 * ( x1 - x ) + q11 * ( x - x0 ) ) / ( ( x1 - x0 ) ) )
        return interpFlux


    #print(x1.data[0],y1.data[0],_x1.data[0],x2.data[0],_x2.data[0],y1.data[0],y2.data[0],_y1.data[0],_y2.data[0])
    
    c = np.array([ [1., x0, y0, x0*y0], #00
                   [1., x1, y0, x1*y0], #10
                   [1., x0, y1, x0*y1], #01
                   [1., x1, y1, x1*y1], #11
                  ], dtype='float')

    invc      = inv(c)
    transinvc = np.transpose(invc)

    final = np.dot(transinvc, [1, x, y, x*y])
    #print('Final Sum (bilinear):', np.sum(final)) # Should be very close to 1

    interpFlux = 10**( (q00*final[0] + q10*final[1] + q01*final[2] + q11*final[3] ) )

    return interpFlux


################################################################

def trilinear_interpolation(x, y, z, points):
    '''Interpolate (x,y,z) from values associated with 9 points.

    Custom routine

    '''

    (x0, y0, z0, q000), (x1, y0, z0, q100), (x0, y1, z0, q010), (x1, y1, z0, q110), \
    (x0, y0, z1, q001), (x1, y0, z1, q101), (x0, y1, z1, q011), (x1, y1, z1, q111),  = points
    #x0 = x0.data[0]
    #x1 = x1.data[0]
    #y0 = y0.data[0]
    #y1 = y1.data[0]
    #z0 = z0.data[0]
    #z1 = z1.data[0]
    #print(x0.data[0],x1.data[0],y0.data[0],y1.data[0],z0.data[0],z1.data[0])

    # Check if we just need bilinear interpolation!
    if x0 == x1: 
        points2 = (y0, z0, q000), (y0, z1, q001), (y1, z0, q010), (y1, z1, q011)
        interpFlux = bilinear_interpolation(y, z, points2)
        return interpFlux

    if y0 == y1:
        points2 = (x0, z0, q000), (x0, z1, q001), (x1, z0, q100), (x1, z1, q101)
        interpFlux = bilinear_interpolation(x, z, points2)
        return interpFlux

    if z0 == z1:
        points2 = (x0, y0, q000), (x0, y1, q010), (x1, y0, q100), (x1, y1, q110)
        interpFlux = bilinear_interpolation(x, y, points2)
        return interpFlux

    c = np.array([ [1., x0, y0, z0, x0*y0, x0*z0, y0*z0, x0*y0*z0], #000
                   [1., x1, y0, z0, x1*y0, x1*z0, y0*z0, x1*y0*z0], #100
                   [1., x0, y1, z0, x0*y1, x0*z0, y1*z0, x0*y1*z0], #010
                   [1., x1, y1, z0, x1*y1, x1*z0, y1*z0, x1*y1*z0], #110
                   [1., x0, y0, z1, x0*y0, x0*z1, y0*z1, x0*y0*z1], #001
                   [1., x1, y0, z1, x1*y0, x1*z1, y0*z1, x1*y0*z1], #101
                   [1., x0, y1, z1, x0*y1, x0*z1, y1*z1, x0*y1*z1], #011
                   [1., x1, y1, z1, x1*y1, x1*z1, y1*z1, x1*y1*z1], #111
                  ], dtype='float')

    invc      = inv(c)
    transinvc = np.transpose(invc)

    final = np.dot(transinvc, [1, x, y, z, x*y, x*z, y*z, x*y*z])
    #print('Final Sum (trilinear):', np.sum(final)) # Should be very close to 1

    interpFlux = 10**( (q000*final[0] + q100*final[1] + q010*final[2] + q110*final[3] + 
                        q001*final[4] + q101*final[5] + q011*final[6] + q111*final[7] ) )

    return interpFlux

################################################################

def quadlinear_interpolation(x, y, z, t, points):
    '''Interpolate (x,y,z,t) from values associated with 16 points.

    Custom routine

    '''

    (x0, y0, z0, t0, q0000), (x1, y0, z0, t0, q1000), (x0, y1, z0, t0, q0100), (x0, y0, z1, t0, q0010), (x0, y0, z0, t1, q0001), \
    (x1, y0, z0, t1, q1001), (x0, y1, z0, t1, q0101), (x0, y0, z1, t1, q0011), (x1, y0, z1, t1, q1011), (x0, y1, z1, t1, q0111), \
    (x1, y1, z1, t1, q1111), (x0, y1, z1, t0, q0110), (x1, y0, z1, t0, q1010), (x1, y1, z0, t0, q1100), (x1, y1, z0, t1, q1101), \
    (x1, y1, z1, t0, q1110) = points
    #x0 = x0.data[0]
    #x1 = x1.data[0]
    #y0 = y0.data[0]
    #y1 = y1.data[0]
    #z0 = z0.data[0]
    #z1 = z1.data[0]
    #t0 = t0.data[0]
    #t1 = t1.data[0]

    # Check if we just need trilinear interpolation!
    if x0 == x1: 
        points2 = (y0, z0, t0, q0000), (y1, z0, t0, q0100), (y0, z1, t0, q0010), (y1, z1, t0, q0110), \
                  (y0, z0, t1, q0001), (y1, z0, t1, q0101), (y0, z1, t1, q0011), (y1, z1, t1, q0111),
        interpFlux = trilinear_interpolation(y, z, t, points2)
        return interpFlux

    if y0 == y1: 
        points2 = (x0, z0, t0, q0000), (x1, z0, t0, q1000), (x0, z1, t0, q0010), (x1, z1, t0, q1010), \
                  (x0, z0, t1, q0001), (x1, z0, t1, q1001), (x0, z1, t1, q0011), (x1, z1, t1, q1011),
        interpFlux = trilinear_interpolation(x, z, t, points2)
        return interpFlux

    if z0 == z1: 
        points2 = (x0, y0, t0, q0000), (x1, y0, t0, q1000), (x0, y1, t0, q0100), (x1, y1, t0, q1100), \
                  (x0, y0, t1, q0001), (x1, y0, t1, q1001), (x0, y1, t1, q0101), (x1, y1, t1, q1101),
        interpFlux = trilinear_interpolation(x, y, t, points2)
        return interpFlux

    if t0 == t1: 
        points2 = (x0, y0, z0, q0000), (x1, y0, z0, q1000), (x0, y1, z0, q0100), (x1, y1, z0, q1100), \
                  (x0, y0, z1, q0010), (x1, y0, z1, q1010), (x0, y1, z1, q0110), (x1, y1, z1, q1110),
        interpFlux = trilinear_interpolation(x, y, z, points2)
        return interpFlux


    c = np.array([ [1., x0, y0, z0, t0, x0*y0, x0*z0, x0*t0, y0*z0, y0*t0, z0*t0, x0*y0*z0, x0*y0*t0, x0*z0*t0, y0*z0*t0, x0*y0*z0*t0], #0000
                   [1., x1, y0, z0, t0, x1*y0, x1*z0, x1*t0, y0*z0, y0*t0, z0*t0, x1*y0*z0, x1*y0*t0, x1*z0*t0, y0*z0*t0, x1*y0*z0*t0], #1000
                   [1., x0, y1, z0, t0, x0*y1, x0*z0, x0*t0, y1*z0, y1*t0, z0*t0, x0*y1*z0, x0*y1*t0, x0*z0*t0, y1*z0*t0, x0*y1*z0*t0], #0100
                   [1., x0, y0, z1, t0, x0*y0, x0*z1, x0*t0, y0*z1, y0*t0, z1*t0, x0*y0*z1, x0*y0*t0, x0*z1*t0, y0*z1*t0, x0*y0*z1*t0], #0010
                   [1., x0, y0, z0, t1, x0*y0, x0*z0, x0*t1, y0*z0, y0*t1, z0*t1, x0*y0*z0, x0*y0*t1, x0*z0*t1, y0*z0*t1, x0*y0*z0*t1], #0001
                   [1., x1, y0, z0, t1, x1*y0, x1*z0, x1*t1, y0*z0, y0*t1, z0*t1, x1*y0*z0, x1*y0*t1, x1*z0*t1, y0*z0*t1, x1*y0*z0*t1], #1001
                   [1., x0, y1, z0, t1, x0*y1, x0*z0, x0*t1, y1*z0, y1*t1, z0*t1, x0*y1*z0, x0*y1*t1, x0*z0*t1, y1*z0*t1, x0*y1*z0*t1], #0101
                   [1., x0, y0, z1, t1, x0*y0, x0*z1, x0*t1, y0*z1, y0*t1, z1*t1, x0*y0*z1, x0*y0*t1, x0*z1*t1, y0*z1*t1, x0*y0*z1*t1], #0011
                   [1., x1, y0, z1, t1, x1*y0, x1*z1, x1*t1, y0*z1, y0*t1, z1*t1, x1*y0*z1, x1*y0*t1, x1*z1*t1, y0*z1*t1, x1*y0*z1*t1], #1011
                   [1., x0, y1, z1, t1, x0*y1, x0*z1, x0*t1, y1*z1, y1*t1, z1*t1, x0*y1*z1, x0*y1*t1, x0*z1*t1, y1*z1*t1, x0*y1*z1*t1], #0111
                   [1., x1, y1, z1, t1, x1*y1, x1*z1, x1*t1, y1*z1, y1*t1, z1*t1, x1*y1*z1, x1*y1*t1, x1*z1*t1, y1*z1*t1, x1*y1*z1*t1], #1111
                   [1., x0, y1, z1, t0, x0*y1, x0*z1, x0*t0, y1*z1, y1*t0, z1*t0, x0*y1*z1, x0*y1*t0, x0*z1*t0, y1*z1*t0, x0*y1*z1*t0], #0110
                   [1., x1, y0, z1, t0, x1*y0, x1*z1, x1*t0, y0*z1, y0*t0, z1*t0, x1*y0*z1, x1*y0*t0, x1*z1*t0, y0*z1*t0, x1*y0*z1*t0], #1010
                   [1., x1, y1, z0, t0, x1*y1, x1*z0, x1*t0, y1*z0, y1*t0, z0*t0, x1*y1*z0, x1*y1*t0, x1*z0*t0, y1*z0*t0, x1*y1*z0*t0], #1100
                   [1., x1, y1, z0, t1, x1*y1, x1*z0, x1*t1, y1*z0, y1*t1, z0*t1, x1*y1*z0, x1*y1*t1, x1*z0*t1, y1*z0*t1, x1*y1*z0*t1], #1101
                   [1., x1, y1, z1, t0, x1*y1, x1*z1, x1*t0, y1*z1, y1*t0, z1*t0, x1*y1*z1, x1*y1*t0, x1*z1*t0, y1*z1*t0, x1*y1*z1*t0], #1110
                  ], dtype='float')

    invc      = inv(c)
    transinvc = np.transpose(invc)

    final = np.dot(transinvc, [1, x, y, z, t, x*y, x*z, x*t, y*z, y*t, z*t, x*y*z, x*y*t, x*z*t, y*z*t, x*y*z*t])
    #print('Final Sum (quadlinear):', np.sum(final)) # Should be very close to 1

    interpFlux = 10**( (q0000*final[0] + q1000*final[1] + q0100*final[2] + q0010*final[3] + q0001*final[4] +
                        q1001*final[5] + q0101*final[6] + q0011*final[7] + q1011*final[8] + q0111*final[9] +
                        q1111*final[10]+ q0110*final[11]+ q1010*final[12]+ q1100*final[13]+ q1101*final[14]+
                        q1110*final[15])
                       )

    return interpFlux