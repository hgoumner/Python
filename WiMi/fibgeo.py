# -*- coding: utf-8 -*-
"""
Created on Wed May  9 17:47:36 2018

******************
*** INCOMPLETE ***
******************

This file creates a set of cylinders randomly oriented within a box, as found 
in a matrix-fiber RVE.

@author: Hristo Goumnerov
"""
#%% Import modules

# import numpy for advanced mathematical operations
import numpy as np

# import scipy for scientific operations
import scipy as sp

# import matplotlib for plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#%% Create geometry

c = 1e-6
el = 64*c
cf = 0.05
angd = 90
df = np.pi/180
ang = angd*df
n = 10

xmin = 0
xmax = el
width = xmax - xmin
xl = xmin + cf*width
xr = xmax - cf*width
#xminang = 0
xmaxang = ang

ymin = 0
ymax = el
height = ymax - ymin
yb = ymin + cf*height
yt = ymax - cf*height
#yminang = 0
ymaxang = ang

zmin = 0
zmax = el
depth = zmax - zmin
zb = zmin + cf*depth
zf = zmax - cf*depth
#zminang = 0
zmaxang = ang

xyzr = np.zeros((n,17))

rc = 4

#%% First fiber

xyzr[0,0] = xl + (xr-xl)*np.random.rand()                                       # xc
xyzr[0,1] = yb + (yt-yb)*np.random.rand()                                       # yc
xyzr[0,2] = zb + (zf-zb)*np.random.rand()                                       # zc
xyzr[0,3] = 5*c + (2*c)*np.random.rand()                                        # r - mean
xyzr[0,4] = 6*c*np.random.normal(1,0.01)                                        # r - gauss

xyz = np.random.randint(1,3)

if xyz == 1:

	check = 1
	while (abs(check) > 1e-3):

		a = xmaxang*np.random.rand()                                    # alpha

		bmin = np.arccos(np.sqrt(1 - (np.cos(a))**2))
		b = bmin + (np.pi/2-bmin)*np.random.rand()                      # beta

		g = np.arccos(np.sqrt(1 - (np.cos(a))**2) - (np.cos(b))**2)     # gamma

		check = (np.cos(a))**2+(np.cos(b))**2+(np.cos(g))**2 - 1

	xyzr[0,5] = a/df
	xyzr[0,6] = b/df
	xyzr[0,7] = g/df

elif xyz == 2:

	check = 1
	while (abs(check) > 1e-3):

		b = xmaxang*np.random.rand()                                    # beta

		amin = np.arccos(np.sqrt(1 - (np.cos(b))**2))
		a = amin + (np.pi/2-amin)*np.random.rand()                      # alpha

		g = np.arccos(np.sqrt(1 - (np.cos(b))**2) - (np.cos(a))**2)     # gamma

		check = (np.cos(a))**2+(np.cos(b))**2+(np.cos(g))**2 - 1

	xyzr[0,5] = a/df
	xyzr[0,6] = b/df
	xyzr[0,7] = g/df

elif xyz == 3:

	check = 1
	while (abs(check) > 1e-3):

		g = xmaxang*np.random.rand()                                    # gamma

		amin = np.arccos(np.sqrt(1 - (np.cos(g))**2))
		a = amin + (np.pi/2-amin)*np.random.rand()                      # alpha

		b = np.arccos(np.sqrt(1 - (np.cos(g))**2) - (np.cos(a))**2)     # beta

		check = (np.cos(a))**2+(np.cos(b))**2+(np.cos(g))**2 - 1

	xyzr[0,5] = a/df
	xyzr[0,6] = b/df
	xyzr[0,7] = g/df

xyzr[0,8] = np.cos(xyzr[0,5]*df)*100*c                                          # ux
xyzr[0,9] = np.cos(xyzr[0,6]*df)*100*c                                          # uy
xyzr[0,10] = np.cos(xyzr[0,7]*df)*100*c                                         # uz

xyzr[0,11] = xyzr[0,0] - xyzr[0,8]                                              # x-
xyzr[0,12] = xyzr[0,1] - xyzr[0,9]                                              # y-
xyzr[0,13] = xyzr[0,2] - xyzr[0,10]                                             # z-

xyzr[0,14] = xyzr[0,0] + xyzr[0,8]                                              # x+
xyzr[0,15] = xyzr[0,1] + xyzr[0,9]                                              # y+
xyzr[0,16] = xyzr[0,2] + xyzr[0,10]                                             # z+

#%% Remaining fibers

mit = 10000
it = 0
count = 1

dc = []
cc = []
dccd = []

while (count<n):

    xcur = xl + (xr-xl)*np.random.rand()   # xc
    ycur = yb + (yt-yb)*np.random.rand()   # yc
    zcur = zb + (zf-zb)*np.random.rand()   # zc
    r1cur = 5*c + (2*c)*np.random.rand()    # r - mean
    r2cur = 6*c*np.random.normal(1,0.01)    # r - gauss

    xyz = np.random.randint(1,3)

    if xyz == 1:

    	check = 1
    	while (abs(check) > 1e-3):

    		a = xmaxang*np.random.rand()                                    # alpha

    		bmin = np.arccos(np.sqrt(1 - (np.cos(a))**2))
    		b = bmin + (np.pi/2-bmin)*np.random.rand()                      # beta

    		g = np.arccos(np.sqrt(1 - (np.cos(a))**2) - (np.cos(b))**2)     # gamma

    		check = (np.cos(a))**2+(np.cos(b))**2+(np.cos(g))**2 - 1

    	alpcur = a/df
    	betcur = b/df
    	gamcur = g/df

    elif xyz == 2:

    	check = 1
    	while (abs(check) > 1e-3):

    		b = xmaxang*np.random.rand()                                    # beta

    		amin = np.arccos(np.sqrt(1 - (np.cos(b))**2))
    		a = amin + (np.pi/2-amin)*np.random.rand()                      # alpha

    		g = np.arccos(np.sqrt(1 - (np.cos(b))**2) - (np.cos(a))**2)     # gamma

    		check = (np.cos(a))**2+(np.cos(b))**2+(np.cos(g))**2 - 1

    	alpcur = a/df
    	betcur = b/df
    	gamcur = g/df

    elif xyz == 3:

    	check = 1
    	while (abs(check) > 1e-3):

    		g = xmaxang*np.random.rand()                                    # gamma

    		amin = np.arccos(np.sqrt(1 - (np.cos(g))**2))
    		a = amin + (np.pi/2-amin)*np.random.rand()                      # alpha

    		b = np.arccos(np.sqrt(1 - (np.cos(g))**2) - (np.cos(a))**2)     # beta

    		check = (np.cos(a))**2+(np.cos(b))**2+(np.cos(g))**2 - 1

    	alpcur = a/df
    	betcur = b/df
    	gamcur = g/df

    ucur = np.cos(alpcur*df)*100*c                              # ux
    vcur = np.cos(betcur*df)*100*c                              # uy
    wcur = np.cos(gamcur*df)*100*c                             # uz

    xmincur = xcur - ucur                          # x-
    ymincur = ycur - vcur                          # y-
    zmincur = zcur - wcur                          # z-

    xmaxcur = xcur + ucur                          # x-
    ymaxcur = ycur + vcur                          # y-
    zmaxcur = zcur + wcur                          # z-

    all_l = 0

    for j in range(count):

        p1 = np.array([xyzr[j,0],xyzr[j,1],xyzr[j,2]])
        u1 = np.array([xyzr[j,8],xyzr[j,9],xyzr[j,10]])

        p2 = np.array([xcur,ycur,zcur])
        u2 = np.array([ucur,vcur,wcur])

        n2 = np.cross(u2,np.cross(u1,u2))
        n1 = np.cross(u1,np.cross(u2,u1))

#        c1 = np.zeros(3)
#        c2 = np.zeros(3)

        c1 = p1 + (np.dot((p2-p1),n2)/(np.dot(u1,n2)))*u1
        c2 = p2 + (np.dot((p1-p2),n1)/(np.dot(u2,n1)))*u2

        distv = c1 - c2
        dist = np.linalg.norm(distv)

        if (xmin < distv[0] < xmax) and (ymin < distv[1] < ymax) and (zmin < distv[2] < zmax):
            pass
        else:
            dist = 1.1 * dist

#        p12 = p1 - p2
#        pq_cross = np.cross(u1,u2)
#        pq_mag = np.linalg.norm(pq_cross)

#        dist = (np.dot(p12,pq_cross)/pq_mag)

        crit = 1.5*(xyzr[j,rc] + r2cur)

        dc.append(abs(dist))
        cc.append(crit)
        dccd.append(abs(dist)-crit)

        if abs(dist)-crit > 0:

            all_l += 1

    if all_l == count:

        xyzr[count,0] = xcur                              # xc
        xyzr[count,1] = ycur                              # yc
        xyzr[count,2] = zcur                              # zc
        xyzr[count,3] = r1cur                             # r - mean
        xyzr[count,4] = r2cur                             # r - gauss
        xyzr[count,5] = alpcur                            # alpha
        xyzr[count,6] = betcur                            # beta
        xyzr[count,7] = gamcur                            # gamma
        xyzr[count,8] = ucur                              # ux
        xyzr[count,9] = vcur                              # uy
        xyzr[count,10] = wcur                             # uz

        xyzr[count,11] = xmincur                          # x-
        xyzr[count,12] = ymincur                          # y-
        xyzr[count,13] = zmincur                          # z-

        xyzr[count,14] = xmaxcur                          # x+
        xyzr[count,15] = ymaxcur                          # y+
        xyzr[count,16] = zmaxcur                          # z+

        count += 1

    it += 1

    print('%3d' % (100*it/mit) + ' %')

    if it == mit:
        break


dcc = np.c_[dc,cc,dccd]

#%% Plot lines

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for k in range(n):
    x = []
    x.append(xyzr[k,11])
    x.append(xyzr[k,14])
    y = []
    y.append(xyzr[k,12])
    y.append(xyzr[k,15])
    z = []
    z.append(xyzr[k,13])
    z.append(xyzr[k,16])
    ax.plot(x,y,z, lw=5)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")

fig = plt.figure()
plt.subplot(311)
plt.scatter(xyzr[:count,5],xyzr[:count,6])
plt.xlabel("X")
plt.ylabel("Y")

#fig = plt.figure()
plt.subplot(312)
plt.scatter(xyzr[:count,5],xyzr[:count,7])
plt.xlabel("X")
plt.ylabel("Z")

#fig = plt.figure()
plt.subplot(313)
plt.scatter(xyzr[:count,6],xyzr[:count,7])
plt.xlabel("Y")
plt.ylabel("Z")

print(count)