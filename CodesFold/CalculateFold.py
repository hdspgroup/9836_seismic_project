# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 18:30:33 2020

@author: Usuario
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 14:46:07 2019

@author: RECORDER article,  Canadian Society of Exploration Geophysicists (CSEG), NOVEMBER FOCUS: Programming a Seismic Program.

Adding codes: Dorian Caraballo L. PhD IN GEOPHYSICS

    
"""

import matplotlib.patches as mpatches
import math
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import fiona
# from shapely.geometry import Point
from shapely.geometry import Point, LineString
from geopandas import GeoDataFrame

# from rtree import index
# %matplotlib inline
# initial conditions
SL = 60  # Source line interval (m)
RL = 60  # Receiver line interval (m)
si = 10  # Source point interval (m)
ri = 10  # Receiver point interval (m)
x = 300  # x extent of survey (m)
y = 180  # y extent of survey (m)
#
x = 300  # x extent of survey (m)
y = 180  # y extent of survey (m)
#
xmi = 0.0  # leftmost corner of grid (m)
ymi = 0.0  # bottommost corner of grid (m)
# SL = 250       # shot line interval (m)
# RL = 125       # receiver line interval (m)
# si = 25       # source point interval (m)
# ri = 12.5       # receiver interval (m)
# x = 1450      # width of survey (m)
# y = 1250     # height of survey (m)


# epsg = 26911


# Number of receiver lines and source lines
rlines = int(y / RL) + 1
slines = int(x / SL) + 1

# Put recevier lines East-West, and shot lines North South
rperline = int(x / ri) + 2
sperline = int(y / si) + 2

# offset the receivers relative to the sources
shiftx = -si / 2.
shifty = -ri / 2.

# [x**2 for x in range(10)]
# [x**y for x in range(4) for y in range(4)]

# Find x,y coordinates of recs and shots
rcvrx = [xmi + rcvr * ri + shiftx for line in range(rlines) for rcvr in range(rperline)]
rcvry = [ymi + line * RL - shifty for line in range(rlines) for rcvr in range(rperline)]

srcx = [xmi + line * SL for line in range(slines) for src in range(sperline)]
srcy = [ymi + src * si for line in range(slines) for src in range(sperline)]

xrec = rcvrx
yrec = rcvry
xc = np.array(xmi)
yc = np.array(ymi)
angle = 0.0
# angle=317
# Function to rotate a point about another point, returning a list [X,Y]
# def RotateXY(xrec,yrec,xc=0,yc=0,angle=0):
xrec = xrec - xc
yrec = yrec - yc
xrr = (xrec * math.cos(angle)) - (yrec * math.sin(angle)) + xc
yrr = (xrec * math.sin(angle)) + (yrec * math.cos(angle)) + yc
#        return [xrr,yrr]

xs = srcx
ys = srcy
xc = np.array(xmi)
yc = np.array(ymi)
# angle=317
angle = 0.0
# def RotateXY(xs,ys,xc=0,yc=0,angle=0):

xs = xs - xc
ys = ys - yc
xrs = (xs * math.cos(angle)) - (ys * math.sin(angle)) + xc
yrs = (xs * math.sin(angle)) + (ys * math.cos(angle)) + yc


#        return [xrr,yrr]


def plot_geoms(xcoords, ycoords, color='none', size=50, alpha=0.5):
    """
    A helper function to make it a bit easier to plot multiple things.
    """
    plot = plt.scatter(xcoords,
                       ycoords,
                       c=color,
                       s=size,
                       marker='o',
                       alpha=alpha,
                       edgecolor='none'
                       )
    return plot


fig = plt.figure(figsize=(15, 10))
r = plot_geoms(xrr, yrr, 'b')
s = plot_geoms(xrs, yrs, 'r')
plt.xlabel('X(m)')
plt.ylabel('Y(m)')
plt.axis('equal')
plt.show()

rcvrxy = zip(xrr, yrr)
srcxy = zip(xrs, yrs)

# Create lists of shapely Point objects.
rcvrs = [Point(x, y) for x, y in rcvrxy]
srcs = [Point(x, y) for x, y in srcxy]

# Add lists to GeoPandas GeoDataFrame objects.
receivers = GeoDataFrame({'geometry': rcvrs})
sources = GeoDataFrame({'geometry': srcs})

# =======Midpoint calculations=================================================

midpoint_list = [LineString([r, s]).interpolate(0.5, normalized=True)
                 for r in rcvrs
                 for s in srcs]

offsets = [r.distance(s)
           for r in rcvrs
           for s in srcs]

azimuths = [np.arctan((r.x - s.x) / (r.y - s.y))
            for r in rcvrs
            for s in srcs]

midpoints = gpd.GeoDataFrame({'geometry': midpoint_list,
                              'offset': offsets,
                              'azimuth': np.degrees(azimuths),
                              })
# midpoints = gpd.GeoDataFrame({'geometry': midpoint_list})

# midpoints[:5]
#
#
##except:
#    # This will work regardless.
# ax=midpoints.plot()
midpoints.plot(column='offset', colormap='jet')
plt.grid()
plt.show()

# p = {'distanceˈ: 0.5, ˈnormalized': True}
# midpoint_list = [LineString([r,s]).interpolate(**p)
#   for r in rcvrs
#   for s in srcs]

# =================Bins========================================================
# Factor to shift the bins relative to source and receiver points
jig = si / 4.
bin_centres = gpd.GeoSeries([Point(xmi + 0.5 * r * ri + jig, ymi + 0.5 * s * si + jig)
                             for r in range(2 * rperline - 3)
                             for s in range(2 * sperline - 2)
                             ])

# Buffers are diamond shaped so we have to scale and rotate them.
scale_factor = np.sin(np.pi / 4.) / 2.
bin_polys = bin_centres.buffer(scale_factor * ri, 1).rotate(-45)
bins = gpd.GeoDataFrame(geometry=bin_polys)
ax = bins.plot()
plt.show()


# =========================FOLD CALCULATION===================================
#
def bin_the_midpoints(bins, midpoints):
    b = bins.copy()
    m = midpoints.copy()
    reindexed = b.reset_index().rename(columns={'index': 'bins_index'})
    joined = gpd.tools.sjoin(reindexed, m)
    bin_stats = joined.groupby('bins_index')['offset'] \
        .agg({'fold': len, 'offset': np.min})
    return gpd.GeoDataFrame(b.join(bin_stats))

# bin_stats = bin_the_midpoints(bins, midpoints)
#
# ax = bin_stats.plot(column="fold")
