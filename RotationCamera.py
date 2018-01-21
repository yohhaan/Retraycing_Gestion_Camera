#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 11 08:21:44 2018

@author: yohanscene
"""

from raytracer import *
from raytracer_colors import *
from raytracer_materials import *

scene = Scene('camera_init', black, 1)
scene.append(LightSource(vec3(-5., 4, -5), pink*0.8))
#
scene.append(LightSource(vec3(5., 6, -5), white*0.8))
#
scene.append(Plane(vec3(0,-0.1,0), vec3(0,1,0), diffuse=0.75, mirror=0, specular=0))

# fonctions pour objets bicolores
def checker(self,M):
    P = M - self.center
    theta = 10 * np.arccos(P.x/np.sqrt(self.radius*self.radius - P.y*P.y)) * np.where(P.z < 0, -1, 1) / np.pi
    phi = 12 * np.arccos((M.y-self.center.y)/self.radius) / np.pi
    return np.floor(theta) % 2 == phi.astype(int) % 2

def squares(self,M):
    P = M - self.center
    theta = 10 * np.arccos(P.x/np.sqrt(self.radius*self.radius - P.y*P.y)) * np.where(P.z < 0, -1, 1) / np.pi
    phi = 12 * np.arccos((M.y-self.center.y)/self.radius) / np.pi
    t = theta % 2 - 1
    p = phi % 2 - 1
    return ((t < 0.75)*(p < 0.75))

def dots(self,M):
    P = M - self.center
    theta = 10 * np.arccos(P.x/np.sqrt(self.radius*self.radius - P.y*P.y)) * np.where(P.z < 0, -1, 1) / np.pi
    phi = 12 * np.arccos((M.y-self.center.y)/self.radius) / np.pi
    t = theta %2 - 1
    p = phi %2 - 1
    return np.where( phi > np.pi * 0.53, t*t + p*p < 0.5, 1 )

r = 0.33
args = {'diffuse': (lightsteelblue,steelblue), 'specular': grey, 'phong':200, 'mirror':0}
s1 = BicolorSphere(vec3(-r-0.1,0.2,-1+r), r, **args).set_selector(checker)

args = {'diffuse': (plum,purple), 'specular': grey, 'phong':200, 'mirror':0}
s2 = BicolorSphere(vec3(+r+0.1,0.2,-1+r), r, **args).set_selector(squares)

args = {'diffuse': (yellow,orange), 'specular': grey, 'phong':200, 'mirror':0}
s3 = BicolorSphere(vec3(0,0.2,-1-r), r, **args).set_selector(dots)

scene.append(s1)
scene.append(s2)
scene.append(s3)

# centre de la scène (i.e. point visé par la caméra)
P = vec3(0,0.2,-0.8)

# position de la caméra
E = vec3(2,4,3)

# direction de la caméra
U = P - E

# orientation de la caméra
V = U.crossprod(vec3(0,-1,0))

# angle de la caméra
ang = 25 * np.pi / 180.0

# et hop !
scene.initialize(400,300,Camera(E,U,V,ang))

scene.trace().save_image()



scene.camera.move_around(P, np.pi/4)
scene.initialize().trace().save_image("camera_move_1.png")

scene.camera.move_around(P, np.pi/4)
scene.initialize().trace().save_image("camera_move_2.png")
pass