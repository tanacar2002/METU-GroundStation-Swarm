"""
Welcome to the METU Modeluydu Ground Station Control.

Choose a video file to send to the satellite.
Then select a COM port with the correct baudrate to achieve comminucation.

Press ESC to quit.
"""
from __future__ import print_function
import argparse
import os
import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg
import time
from datetime import datetime
import traceback
import cv2
import pygame
import serial
import serial.tools.list_ports
import serial.threaded
import threading
import multiprocessing as mp
import csv
import urllib.error
import urllib.request
from classes.tangui import *
from classes.urlsigner import *
from classes.ObjLoader import ObjLoader
from classes.TextureLoader import load_texture,load_texture_pygame
import struct
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GL.shaders import compileProgram, compileShader
import pyrr
import pdb
import gc
import functools
from tkinter import Tk
from tkinter.filedialog import askopenfilename
Tk().withdraw()

import hashlib
import hmac
import base64
import urllib.parse as urlparse

from OpenGL.GL import glBindTexture, glTexParameteri, GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, \
    GL_TEXTURE_WRAP_T, GL_REPEAT, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR,\
    glTexImage2D, GL_RGBA, GL_UNSIGNED_BYTE
from PIL import Image

try:
    # Python 3.4+
    if sys.platform.startswith('win'):
        import multiprocessing.popen_spawn_win32 as forking
    else:
        import multiprocessing.popen_fork as forking
except ImportError:
    import multiprocessing.forking as forking

if sys.platform.startswith('win'):
    # First define a modified version of Popen.
    class _Popen(forking.Popen):
        def __init__(self, *args, **kw):
            if hasattr(sys, 'frozen'):
                # We have to set original _MEIPASS2 value from sys._MEIPASS
                # to get --onefile mode working.
                os.putenv('_MEIPASS2', sys._MEIPASS)
            try:
                super(_Popen, self).__init__(*args, **kw)
            finally:
                if hasattr(sys, 'frozen'):
                    # On some platforms (e.g. AIX) 'os.unsetenv()' is not
                    # available. In those cases we cannot delete the variable
                    # but only set it to the empty string. The bootloader
                    # can handle this case.
                    if hasattr(os, 'unsetenv'):
                        os.unsetenv('_MEIPASS2')
                    else:
                        os.putenv('_MEIPASS2', '')

    # Second override 'Popen' class with our modified version.
    forking.Popen = _Popen

class ProgressBar:
    inner_color = (200,220,210)
    progress_color = (0,255,0)
    text_color = (255,255,255)
    border_color = (128,128,128)
    border_width = 3
    border_radius = 5
    deriv,eta = 0,0
    eta_visible = False

    def __init__(self,x=0,y=0,width=300,height=25,min=0,max=100,value=0):
        self.font = pygame.font.SysFont(pygame.font.get_fonts()[0],int(height))
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.min = min
        self.max = max
        self.valuerange = abs(self.max - self.min)
        self.value = value
        self.resetTime()
        self.text = self.font.render("%{:4.1f}".format(self.value*100/self.valuerange), True, self.text_color)
        self.textrect = self.text.get_size()
        self.etatext = self.font.render("ETA: {:d}m {:2.0f}s".format(int(self.eta//60),self.eta%60), True, self.text_color)
        self.etatextrect = self.etatext.get_size()

    def resetTime(self):
        self.reftime = time.time()
        self.refvalue = self.value

    def toggleEta(self,*args):
        if len(args) == 1:
            self.eta_visible = args[0]
        else:
            self.eta_visible = not self.eta_visible
        self.resetTime()

    def setColor(self,color):
        self.inner_color = color

    def posScale(self,scalex,scaley):
        self.x = scalex*self.x
        self.y = scaley*self.y
        self.width = scalex*self.width
        self.height = scaley*self.height
        self.font = pygame.font.SysFont(pygame.font.get_fonts()[0],int(self.height))
        self.text = self.font.render("%{:4.1f}".format(self.value*100/self.valuerange), True, self.text_color)
        self.textrect = self.text.get_size()
        self.etatext = self.font.render("ETA: {:d}m {:2.0f}s".format(int(self.eta//60),self.eta%60), True, self.text_color)
        self.etatextrect = self.etatext.get_size()

    def setValue(self,value):
        self.value = value
        try:
            self.deriv = (self.value-self.refvalue)/(time.time()-self.reftime)
            self.eta = (self.max-self.value)/self.deriv
        except:
            self.eta = 0
        self.text = self.font.render("%{:4.1f}".format(self.value*100/self.valuerange), True, self.text_color)
        self.textrect = self.text.get_size()
        if self.eta < 0:
            self.etatext = self.font.render("Completed!", True, self.text_color)
        else:
            self.etatext = self.font.render("ETA: {:d}m {:2.0f}s".format(int(self.eta//60),self.eta%60), True, self.text_color)
        self.etatextrect = self.etatext.get_size()

    def draw(self,surface):
        pygame.draw.rect(surface,self.inner_color,(self.x-self.width//2,self.y-self.height//2,self.width,self.height),border_radius=self.border_radius)
        pygame.draw.rect(surface,self.progress_color,(self.x-self.width//2+self.border_width,self.y-self.height//2,(self.width-self.border_width*2)*self.value/self.valuerange,self.height),border_radius=self.border_radius)
        pygame.draw.rect(surface,self.border_color,(self.x-self.width//2,self.y-self.height//2,self.width,self.height),width=self.border_width,border_radius=self.border_radius)
        surface.blit(self.text,(self.x + self.width//2 + self.border_width, self.y - self.textrect[1]//2))
        if self.eta_visible:
            surface.blit(self.etatext,(self.x - self.etatextrect[0]//2, self.y - self.etatextrect[1]//2 + self.height))

class Selector:
    inner_color = (255,255,255)
    text_color = (0,0,0)
    border_color = (128,128,128)
    border_width = 2
    border_radius = 0
    index = 0

    def __init__(self,x=0,y=0,width=50,height=20,choices=["Choice"]):
        self.font = pygame.font.SysFont(
            pygame.font.get_fonts()[0], int(height))  # int(height))
        self.x = x
        self.y = y
        self.height = height*(30/20)
        self.values = choices
        self.choices = []
        for i,choice in enumerate(self.values):
            self.choices.append(Button(i,x = self.x,height=self.height,text=str(choice)))
        self.width = max(max([choice.width for choice in self.choices]),width)
        for i,choice in enumerate(self.choices):
            choice.width = self.width
            choice.height = self.height
            choice.y = self.y + self.height*(i+1)
            choice.visibility = False
            choice.border_radius = 0

    def getValue(self):
        return self.values[self.index]

    def getChoice(self):
        return self.current_choice

    def updateChoices(self,choices):
        self.__init__(self.x, self.y, self.width,
                      self.height*(20/30), choices=choices)
        if self.index >= len(choices):
            self.index = 0

    def checkHover(self,pos):
        for choice in self.choices:
            if choice.checkHover(pos) and choice.visibility:
                self.index = int(choice.clickFunction())
                self.clickFunction()

        return (self.x-self.width//2 < pos[0] < self.x + self.width//2) and (self.y - self.height//2 < pos[1] < self.y + self.height//2)

    def setColor(self,color):
        self.inner_color = color

    def posScale(self,scalex,scaley):
        self.__init__(self.x*scalex,self.y*scaley,scalex*self.width,scaley*self.height,self.values)
        # self.x = scale*self.x
        # self.y = scale*self.y
        # self.width = scale*self.width
        # self.height = scale*self.height
        # for choice in self.choices:
        #     choice.x = scale*choice.x
        #     choice.y = scale*choice.y
        #     choice.width = scale*choice.width
        #     choice.height = scale*choice.height

    def clickFunction(self):
        for choice in self.choices:
            choice.toggle()

    def draw(self,surface):
        for choice in self.choices:
            choice.draw(surface)
        self.text = self.font.render(str(self.values[self.index]), True, self.text_color)
        self.textrect = self.text.get_size()
        pygame.draw.rect(surface,self.inner_color,(self.x-self.width//2,self.y-self.height//2,self.width,self.height),border_radius=self.border_radius)
        pygame.draw.rect(surface,self.border_color,(self.x-self.width//2,self.y-self.height//2,self.width,self.height),width=self.border_width,border_radius=self.border_radius)
        surface.blit(self.text,(self.x - self.textrect[0]//2, self.y - self.textrect[1]//2))

    def __del__(self):
        for choice in self.choices:
            del choice

class Text:
    def __init__(self,x,y,text,font,color=(255,255,255),allign='center'):
        self.font = font
        self.textstr = text
        self.color = color
        self.text = self.font.render(self.textstr, True, self.color)
        self.x = x
        self.y = y
        self.allignment = allign

    def setText(self,text):
        self.textstr = text
        self.text = self.font.render(self.textstr, True, self.color)

    def posScale(self,scalex,scaley):
        self.x = scalex*self.x
        self.y = scaley*self.y

    def setAllignment(self,allign):
        self.allignment = allign

    def draw(self,surface):
        surface.blit(self.text,(self.x-(self.text.get_width()//2 if self.allignment.lower() == 'center' else 0),self.y-(self.text.get_height()//2 if self.allignment.lower() == 'center' else 0)))

    def __str__(self):
        return self.textstr

class Button:
    inner_color = (255,255,255)
    text_color = (0,0,0)
    border_color = (128,128,128)
    border_width = 2
    border_radius = 5
    visibility = True

    def function(self): pass

    def __init__(self,id,x=0,y=0,width=50,height=30,text="Button"):
        self.id = str(id)
        self.font = pygame.font.SysFont(pygame.font.get_fonts()[0],int(height*(20/30)))
        self.x = x
        self.y = y
        self.textstr = str(text)
        self.text = self.font.render(self.textstr, True, self.text_color)
        self.textrect = self.text.get_size()
        self.width = max(width,self.textrect[0] + self.border_width*3)
        self.height = max(height,self.textrect[1] + self.border_width*3)

    def getText(self):
        return self.textstr

    def checkHover(self,pos):
        return (self.x-self.width//2 < pos[0] < self.x + self.width//2) and (self.y - self.height//2 < pos[1] < self.y + self.height//2)

    def setFunction(self,func):
        self.function = func

    def toggle(self):
        self.visibility = not self.visibility

    def posScale(self,scalex,scaley):
        self.x = scalex*self.x
        self.y = scaley*self.y
        self.font = pygame.font.SysFont(pygame.font.get_fonts()[0], int(scaley*self.height*(20/30)))

    def setColor(self,color):
        self.inner_color = color

    def clickFunction(self):
        if self.visibility:
            self.function()
            return self.id

    def draw(self,surface):
        if self.visibility:
            pygame.draw.rect(surface,self.inner_color,(self.x-self.width//2,self.y-self.height//2,self.width,self.height),border_radius=self.border_radius)
            pygame.draw.rect(surface,self.border_color,(self.x-self.width//2,self.y-self.height//2,self.width,self.height),width=self.border_width,border_radius=self.border_radius)
            surface.blit(self.text,(self.x - self.textrect[0]//2, self.y - self.textrect[1]//2))

class ObjLoader:
    buffer = []

    @staticmethod
    def search_data(data_values, coordinates, skip, data_type):
        for d in data_values:
            if d == skip:
                continue
            if data_type == 'float':
                coordinates.append(float(d))
            elif data_type == 'int':
                coordinates.append(int(d)-1)


    @staticmethod # sorted vertex buffer for use with glDrawArrays function
    def create_sorted_vertex_buffer(indices_data, vertices, textures, normals):
        for i, ind in enumerate(indices_data):
            if i % 3 == 0: # sort the vertex coordinates
                start = ind * 3
                end = start + 3
                ObjLoader.buffer.extend(vertices[start:end])
            elif i % 3 == 1: # sort the texture coordinates
                start = ind * 2
                end = start + 2
                ObjLoader.buffer.extend(textures[start:end])
            elif i % 3 == 2: # sort the normal vectors
                start = ind * 3
                end = start + 3
                ObjLoader.buffer.extend(normals[start:end])


    @staticmethod # TODO unsorted vertex buffer for use with glDrawElements function
    def create_unsorted_vertex_buffer(indices_data, vertices, textures, normals):
        num_verts = len(vertices) // 3

        for i1 in range(num_verts):
            start = i1 * 3
            end = start + 3
            ObjLoader.buffer.extend(vertices[start:end])

            for i2, data in enumerate(indices_data):
                if i2 % 3 == 0 and data == i1:
                    start = indices_data[i2 + 1] * 2
                    end = start + 2
                    ObjLoader.buffer.extend(textures[start:end])

                    start = indices_data[i2 + 2] * 3
                    end = start + 3
                    ObjLoader.buffer.extend(normals[start:end])

                    break


    @staticmethod
    def show_buffer_data(buffer):
        for i in range(len(buffer)//8):
            start = i * 8
            end = start + 8
            print(buffer[start:end])


    @staticmethod
    def load_model(file, sorted=True):
        vert_coords = [] # will contain all the vertex coordinates
        tex_coords = [] # will contain all the texture coordinates
        norm_coords = [] # will contain all the vertex normals

        all_indices = [] # will contain all the vertex, texture and normal indices
        indices = [] # will contain the indices for indexed drawing


        with open(file, 'r') as f:
            line = f.readline()
            while line:
                values = line.split()
                if values[0] == 'v':
                    ObjLoader.search_data(values, vert_coords, 'v', 'float')
                elif values[0] == 'vt':
                    ObjLoader.search_data(values, tex_coords, 'vt', 'float')
                elif values[0] == 'vn':
                    ObjLoader.search_data(values, norm_coords, 'vn', 'float')
                elif values[0] == 'f':
                    for value in values[1:]:
                        val = value.split('/')
                        ObjLoader.search_data(val, all_indices, 'f', 'int')
                        indices.append(int(val[0])-1)

                line = f.readline()

        if sorted:
            # use with glDrawArrays
            ObjLoader.create_sorted_vertex_buffer(all_indices, vert_coords, tex_coords, norm_coords)
        else:
            # use with glDrawElements
            ObjLoader.create_unsorted_vertex_buffer(all_indices, vert_coords, tex_coords, norm_coords)

        # ObjLoader.show_buffer_data(ObjLoader.buffer)

        buffer = ObjLoader.buffer.copy() # create a local copy of the buffer list, otherwise it will overwrite the static field buffer
        ObjLoader.buffer = [] # after copy, make sure to set it back to an empty list

        return np.array(indices, dtype='uint32'), np.array(buffer, dtype='float32')

# for use with GLFW
def load_texture(path, texture):
    glBindTexture(GL_TEXTURE_2D, texture)
    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # load image
    image = Image.open(path)
    image = image.transpose(Image.FLIP_TOP_BOTTOM)
    img_data = image.convert("RGBA").tobytes()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image.width, image.height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    return texture

# for use with pygame
def load_texture_pygame(path, texture):
    import pygame
    glBindTexture(GL_TEXTURE_2D, texture)
    # Set the texture wrapping parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # Set texture filtering parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    # load image
    image = pygame.image.load(path)
    image = pygame.transform.flip(image, False, True)
    image_width, image_height = image.get_rect().size
    img_data = pygame.image.tostring(image, "RGBA")
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, image_width, image_height, 0, GL_RGBA, GL_UNSIGNED_BYTE, img_data)
    return texture

def sign_url(input_url=None, secret=None):
    """ Sign a request URL with a URL signing secret.
      Usage:
      from urlsigner import sign_url
      signed_url = sign_url(input_url=my_url, secret=SECRET)
      Args:
      input_url - The URL to sign
      secret    - Your URL signing secret
      Returns:
      The signed request URL
  """

    if not input_url or not secret:
        raise Exception("Both input_url and secret are required")

    url = urlparse.urlparse(input_url)

    # We only need to sign the path+query part of the string
    url_to_sign = url.path + "?" + url.query

    # Decode the private key into its binary format
    # We need to decode the URL-encoded private key
    decoded_key = base64.urlsafe_b64decode(secret)

    # Create a signature using the private key and the URL-encoded
    # string using HMAC SHA1. This signature will be binary.
    signature = hmac.new(decoded_key, str.encode(url_to_sign), hashlib.sha1)

    # Encode the binary signature into base64 for use within a URL
    encoded_signature = base64.urlsafe_b64encode(signature.digest())

    original_url = url.scheme + "://" + url.netloc + url.path + "?" + url.query

    # Return signed URL
    return original_url + "&signature=" + encoded_signature.decode()

def resource_path(relative_path):
     if hasattr(sys, '_MEIPASS'):
         return os.path.join(sys._MEIPASS, relative_path)
     return os.path.join(os.path.abspath("."), relative_path)

#Specify the model for the simulation
model_path = resource_path("assets/cube.obj")
texture_path = resource_path("assets/cube.jpg")
sorted_model = False
pos_vec = [8.7, 3.5, -9]
#Url and keys for the Google Maps Static API
SECRET = ""
GOOGLE_API_KEY = "AIzaSyACej-uNjJkY2q9Z7eIsoeth698h_ivoz4"
GMAP_BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"
#Constants
plot_length = 20
VIDEO_FPS = 10
#Common constants for communication
VIDEO_BUFFER_SIZE = 250                  # Limited by the AltSerialLib RX_BUFFER_SIZE
TELEMETRY_DATA_BYTE = bytes([0b10010011])
ACTIVATE_VIDEO_TRANSMISSION_CMD = bytes([0b11000000])
DEACTIVATE_VIDEO_TRANSMISSON_CMD = bytes([0b10000010])
ACTIVATE_EJECTION_CMD = bytes([0b10000100])
ACTIVATE_RESET_CMD = bytes([0b10001000])
VIDEO_TRANSMISSION_CMPLTD = bytes([0b10100000])      # To ACK Ground Station (State Change @ GS)
MOTOR_TEST_CMD = bytes([0b10010001])
CAMERA_CMD = bytes([0b10100101])

def getvideobytes(path):
    vbytes = []
    with open(path,'rb') as f:
        b = f.read(1)
        while b:
            vbytes.append(b)
            b = f.read(1)
        print("Size of the video file is %d bytes" % len(vbytes))
    return vbytes

def resource_path(relative_path):
     if hasattr(sys, '_MEIPASS'):
         return os.path.join(sys._MEIPASS, relative_path)
     return os.path.join(os.path.abspath("."), relative_path)

def convert8bitu(data):
    return (data & 0b01111111)-(data>>7)*128

def convert16bitu(data):
    return (data & 0x7fff)-(data>>15)*(2**15)

def convert32bitu(data):
    return (data & 0x7fffffff)-(data>>31)*(2**31)

def convert32bitf(data):
    exp = ((data >> 23)&0x0ff)-127 #k-excess
    mantissa = (data & 0x007fffff)/(2**23) + (0 if exp == -127 else 1)
    sign = -1 if (data >> 31) else 1
    exp = -126 if exp == -127 else exp
    if exp == 128 :
        if mantissa == 1:
            return sign*float('inf')
        else:
            return float('nan')
    return sign*mantissa*(2**exp)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR,borderValue=(0,0,0,0))
  return result

#Create the graphs
def create_graphs():
    global fig,ax,canvas
    fig, ax = plt.subplots(3,2,figsize=(resolution[0]*(5.5/1920),resolution[1]*(5.5/1080)))
    ax[0,0].set_title("Pressure (mBar)")
    ax[0,1].set_title("Altitude (m)")
    ax[1,0].set_title("Downward Speed ($m/s^2$)")
    ax[1,1].set_title("Temperature (C degrees)")
    ax[2,0].set_title("Battery Voltage (V)")
    ax[2,1].set_title("Revolutions Around Z Axis")
    for a in ax.flatten():
        a.set_xlabel("Mission Time(s)",color='w')
        a.patch.set_facecolor('gray')
        a.patch.set_alpha(0.2)
        a.grid(True,c="gray")
    plt.subplots_adjust(hspace=1)
    fig.suptitle("Graphs", fontsize=int(resolution[1]*(14/1080)))
    fig.patch.set_alpha(0)
    canvas = agg.FigureCanvasAgg(fig)

def plotsurf():
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = bytes(canvas.buffer_rgba()[:])
    size = canvas.get_width_height()
    return raw_data, size, "RGBA"

def plot(conn):
    #Configure matplotlib
    mpl.use("Agg")
    mpl.rcParams["text.color"] = 'white'
    mpl.rcParams['axes.labelcolor'] = 'white'
    mpl.rcParams['xtick.color'] = 'white'
    mpl.rcParams['ytick.color'] = 'white'
    global resolution
    resolution = conn.recv()
    create_graphs()
    while True:
        if conn.poll():
            plot_data = conn.recv()
            fig.clf()
            plt.close(fig)
            create_graphs()
            for a in ax.flatten():
                a.set_xlim(min(plot_data["packageCounter"][-plot_length:]),max(plot_data["packageCounter"][-1],plot_data["packageCounter"][0]+1))
            ax[0,0].plot(plot_data["packageCounter"][-plot_length:],plot_data["pressure"][-plot_length:],'b')
            ax[0,1].plot(plot_data["packageCounter"][-plot_length:],plot_data["altitude"][-plot_length:],'g')
            ax[1,0].plot(plot_data["packageCounter"][-plot_length:],plot_data["speed"][-plot_length:],'b')
            ax[1,1].plot(plot_data["packageCounter"][-plot_length:],plot_data["temperature"][-plot_length:],'g')
            ax[2,0].plot(plot_data["packageCounter"][-plot_length:],plot_data["batteryVoltage"][-plot_length:],'b')
            ax[2,1].plot(plot_data["packageCounter"][-plot_length:],plot_data["revolution"][-plot_length:],'g')
            conn.send(plotsurf())

def openglinit():
    global VAO,VBO,EBO,textures,model_loc,obj_indices,obj_pos,shader,shaderBackground,backgroundVAO,texture,pos_vec
    #Load 3D object for simulation
    obj_indices, obj_buffer = ObjLoader.load_model(model_path,sorted=sorted_model)
    #Set GL shaders
    shader = compileProgram(compileShader(vertex_src, GL_VERTEX_SHADER),compileShader(fragment_src, GL_FRAGMENT_SHADER))
    shaderBackground = compileProgram(compileShader(VERTEX_SHADER, GL_VERTEX_SHADER), compileShader(FRAGMENT_SHADER, GL_FRAGMENT_SHADER))

    # VAO and VBO
    VAO = glGenVertexArrays(1)
    VBO = glGenBuffers(2)
    EBO = glGenBuffers(1)

    # obj VAO
    glBindVertexArray(VAO)
    # obj Vertex Buffer Object
    glBindBuffer(GL_ARRAY_BUFFER, VBO[0])
    glBufferData(GL_ARRAY_BUFFER, obj_buffer.nbytes, obj_buffer, GL_STATIC_DRAW)

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, obj_indices.nbytes, obj_indices, GL_STATIC_DRAW)

    # obj vertices
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, obj_buffer.itemsize * 8, ctypes.c_void_p(0))
    # obj textures
    glEnableVertexAttribArray(1)
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, obj_buffer.itemsize * 8, ctypes.c_void_p(12))
    # obj normals
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, obj_buffer.itemsize * 8, ctypes.c_void_p(20))
    glEnableVertexAttribArray(2)

    ############### Background Texture #################################
    backgroundVAO = glGenVertexArrays(1)
    glBindVertexArray(backgroundVAO)

    # Bind the buffer
    glBindBuffer(GL_ARRAY_BUFFER, VBO[1])
    glBufferData(GL_ARRAY_BUFFER, 128, rectangle, GL_STATIC_DRAW)

    # Create EBO
    EBO = glGenBuffers(1)
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO)
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, indicesBackground, GL_STATIC_DRAW)

    # get the position from  shader
    position = glGetAttribLocation(shaderBackground, 'position')
    glVertexAttribPointer(position, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(0))
    glEnableVertexAttribArray(position)

    # get the color from  shader
    color = glGetAttribLocation(shaderBackground, 'color')
    #glVertexAttribPointer(color, 3, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(12))
    #glEnableVertexAttribArray(color)

    texCoords = glGetAttribLocation(shaderBackground, "InTexCoords")
    glVertexAttribPointer(texCoords, 2, GL_FLOAT, GL_FALSE, 32, ctypes.c_void_p(24))
    glEnableVertexAttribArray(texCoords)

    # Creating Texture
    texture = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture)
    # texture wrapping params
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    # texture filtering params
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    ######

    textures = glGenTextures(2)
    load_texture_pygame(texture_path, textures[0])
    glUseProgram(shader)

    glEnable(GL_TEXTURE_2D)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
    glTexEnvf(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL)

    projection = pyrr.matrix44.create_perspective_projection_matrix(45, resolution[0]/resolution[1], 0.1, 100)
    obj_pos = pyrr.matrix44.create_from_translation(pyrr.Vector3(pos_vec))
    # eye, target, up
    view = pyrr.matrix44.create_look_at(pyrr.Vector3([0, 0, 8]), pyrr.Vector3([0, 0, 0]), pyrr.Vector3([0, 1, 0]))

    model_loc = glGetUniformLocation(shader, "model")
    proj_loc = glGetUniformLocation(shader, "projection")
    view_loc = glGetUniformLocation(shader, "view")

    glUniformMatrix4fv(proj_loc, 1, GL_FALSE, projection)
    glUniformMatrix4fv(view_loc, 1, GL_FALSE, view)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    gluPerspective(45, (resolution[0]/resolution[1]), 0.1, 100.0)
    glTranslatef(0.0, 0.0, -5)

def eventcheck():
    for event in pygame.event.get():
        global videobytes,video_send_counter,mouse_pos_scale,f,videosendstate,serialPort,serThread
        if event.type==pygame.QUIT:
            return True
        elif event.type == pygame.VIDEORESIZE:
            mouse_pos_scale = np.array(event.size)/np.array(resolution)
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
        elif event.type == pygame.MOUSEBUTTONDOWN:
            #print(event.button)
            #print(np.array(event.pos)/mouse_pos_scale)
            #Check for left mouse click
            if event.button == 1:
                for button in buttons:
                    if button.checkHover(np.array(event.pos)/mouse_pos_scale):
                        #Get which button is pressed
                        result = button.clickFunction()
                        if result == "connectPort":
                            if selectors[1].getValue() == "NaN":
                                print("Please connect a Serial Device!")
                            else:
                                if serialPort.is_open:
                                    serialPort.close()
                                serialPort.baudrate = selectors[0].getValue()
                                serialPort.port = selectors[1].getValue()
                                if args.debug:
                                    print("Connecting to port {} at {} baudrate.".format(
                                        serialPort.port, serialPort.baudrate))
                                texts.append(Text(resolution[0]*(800/1920),resolution[1]*(46/64),'Establishing...',midfont))
                                serialPort.open()
                                while not serialPort.is_open:
                                    pass
                                serialPort.flushInput()  # Clear Rx Buffer
                                texts.pop()
                        elif result == "selectVideo":
                            pathinput = askopenfilename()
                            if pathinput != "":
                                videobytes = getvideobytes(pathinput)
                                video_send_counter = 0
                        elif result == "eject":
                            buttons[0].setColor((200,0,0))
                            cmd_send.append(ACTIVATE_EJECTION_CMD)
                        elif result == "startVideoSend":
                            videosendstate = True
                            videoProgressBar.toggleEta(True)
                            #cmd_send.append(ACTIVATE_VIDEO_TRANSMISSION_CMD)
                            if serialPort.is_open:
                                serialPort.write(ACTIVATE_VIDEO_TRANSMISSION_CMD)
                        elif result == "stopVideoSend":
                            #videosendstate = False
                            videoProgressBar.toggleEta(False)
                            cmd_send.append(DEACTIVATE_VIDEO_TRANSMISSON_CMD)
                        elif result == "resetVideoSend":
                            videoProgressBar.toggleEta(False)
                            #Reset the video_send counter to start sending from beginning of the video file
                            video_send_counter = 0
                        elif result == "motorTest":
                            cmd_send.append(MOTOR_TEST_CMD)
                        elif result == "toggleCamera":
                            cmd_send.append(CAMERA_CMD)
                        elif result == "resetSatellite":
                            cmd_send.append(ACTIVATE_RESET_CMD)
                            global plot_data,start_location,writer,fig
                            resetSat = True
                            #Reset start_location to update starting point in map
                            start_location = None
                            #Reset the telemetry data history for graphs
                            for key in plot_data:
                                plot_data[key] = []
                            #Close the csv file and open a new one
                            f.close()
                            f = open(datapath,'w+',newline='')
                            writer = csv.DictWriter(f,fieldnames=[key for key in telemetry])
                            writer.writeheader()
                for selector in selectors:
                    if selector.checkHover(np.array(event.pos)/mouse_pos_scale):
                        selector.clickFunction()
    return False

def render(display):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    if main:
        surface = pygame.surfarray.make_surface(background2.swapaxes(0, 1))
        display.blit(surface, (0,0))
        #Draw the Graphs
        global graphsurf
        if graphPipe.poll():
            graphsurf = pygame.image.fromstring(*graphPipe.recv())
        display.blit(graphsurf, (resolution[0]*(10/1920),resolution[1]*(530/1080)))
        #Draw the Video Frame
        surface = pygame.surfarray.make_surface(video_frame.swapaxes(0, 1))
        display.blit(surface, (resolution[0]//2-video_frame.shape[1]*(50/100),resolution[1]*(220/1080)))
        #Draw the Map
        surface = pygame.surfarray.make_surface(map_image.swapaxes(0, 1))
        display.blit(surface, (resolution[0]-mapsize*(120/100),resolution[1]-mapsize*(110/100)))
        #Rotate Object
        rot_y = pyrr.matrix44.create_from_y_rotation(telemetry["yaw"]*(np.pi/180))
        rot_z = pyrr.matrix44.create_from_z_rotation(telemetry["pitch"]*(np.pi/180))
        rot_x = pyrr.matrix44.create_from_x_rotation(telemetry["roll"]*(np.pi/180))
        model = pyrr.matrix44.multiply(rot_y, obj_pos)
        model = pyrr.matrix44.multiply(rot_z, model)
        model = pyrr.matrix44.multiply(rot_x, model)
        #Draw Simulation
        glUseProgram(shader)
        glBindVertexArray(VAO)
        glBindTexture(GL_TEXTURE_2D, textures[0])
        glUniformMatrix4fv(model_loc, 1, GL_FALSE, model)
        glDrawArrays(GL_TRIANGLES, 0, len(obj_indices))
        glDrawElements(GL_TRIANGLES, len(obj_indices), GL_UNSIGNED_INT, None)
    else:
        surface = pygame.surfarray.make_surface(background1.swapaxes(0, 1))
        display.blit(surface, (0,0))
        #Draw the Video Frame
        surface = pygame.surfarray.make_surface(video_frame.swapaxes(0, 1))
        display.blit(
            surface, (resolution[0]*(1400/1920)-video_frame.shape[1]*(50/100), resolution[1]*(590/1080)))
    #Draw GUI elements to surface
    surface = pygame.display.get_surface()
    for obj in texts + buttons + selectors + progressbars:
        obj.draw(surface)
    #Draw surface to screen OpenGL style
    surdata = pygame.image.tostring(surface, "RGBA", True)
    glUseProgram(shaderBackground)
    glDepthFunc(GL_LEQUAL)
    glBindVertexArray(backgroundVAO)
    glBindTexture(GL_TEXTURE_2D, texture)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, surface.get_width(), surface.get_height(), 0, GL_RGBA, GL_UNSIGNED_BYTE, surdata)
    glDrawElements(GL_TRIANGLES, len(obj_indices), GL_UNSIGNED_INT, None)

    pygame.display.flip()
#Create a thread for handling incoming data and sending file at the background
def serialThread(id):
    global serialPort,telemetry,writer,texts,cmd_send_state,cmd_send,video_send_counter,video_send_state,plot_data,plot_length,telemetry_update,data_buffer
    while True:
        if serialPort.is_open:
            if serialPort.in_waiting > 0:
                data = serialPort.read()
                if data == TELEMETRY_DATA_BYTE:
                    data_buffer = struct.unpack(telemetrystruct,serialPort.read(36))
                    telemetry_update = True
                elif (data == ACTIVATE_VIDEO_TRANSMISSION_CMD) and (len(videobytes)>video_send_counter):
                    #Send one packet of videobytes
                    cmd_send_state = False
                    #Calculate the Checksum Byte to send
                    chk = (
                        ~sum(ord(b) for b in videobytes[video_send_counter : (video_send_counter + VIDEO_BUFFER_SIZE)])+ 1 & 0xFF)
                    #print("Checksum is {}".format(chk))
                    packageSize = 0 if ((video_send_counter+VIDEO_BUFFER_SIZE) < len(videobytes)) else (len(videobytes)-video_send_counter)
                    if packageSize == 0:
                        if cmd_send:
                            cmd = cmd_send.pop(0)
                            if cmd == DEACTIVATE_VIDEO_TRANSMISSON_CMD:
                                videosendstate = False
                            if args.debug:
                                print("Sent the {} command".format(cmd))
                        else:
                            cmd = bytes([0])
                    else:
                        videosendstate = False
                        cmd = DEACTIVATE_VIDEO_TRANSMISSON_CMD
                    serialPort.write(b''.join([cmd]+[bytes([packageSize])] + videobytes[video_send_counter:(video_send_counter+VIDEO_BUFFER_SIZE)]+[bytes([chk])]))
                    serialPort.flush()
                    if (video_send_counter+VIDEO_BUFFER_SIZE) < len(videobytes):
                        videosendstate = False
                    ## TODO: Test and optimize the video sending process
                    if args.debug:
                        print("Sent {} bytes of video data.".format(VIDEO_BUFFER_SIZE))
                        print(b''.join([cmd]+[bytes([0 if ((video_send_counter+VIDEO_BUFFER_SIZE) < len(videobytes)) else (len(videobytes)-video_send_counter)])] + videobytes[video_send_counter:(video_send_counter+VIDEO_BUFFER_SIZE)]+[bytes([chk])]))
                elif data == VIDEO_TRANSMISSION_CMPLTD:
                    #We successfully transmitted the data, increase the counter and enable sending commands
                    video_send_counter += VIDEO_BUFFER_SIZE
                    cmd_send_state = True
                    if args.debug:
                        print("Video send counter incremented.")
                elif data == DEACTIVATE_VIDEO_TRANSMISSON_CMD:
                    #We couldn't successfully transmitt the data, don't increase the counter but enable sending commands
                    if args.debug:
                        print("Error recieved.")
                    cmd_send_state = True
        else:
            time.sleep(0.1) #To release GIL momentarily
#Create a thread for requesting a map from GOOGLE MAPS API and wait for the response
def mapThread(id):
    global map_image,start_location,telemetry
    #Use GMAP API to get the map
    url = "{}?size={}x{}&markers=color:green%7Clabel:S%7C{},{}&markers=color:red%7Clabel:F%7C{},{}&key={}".format(GMAP_BASE_URL,mapsize,mapsize,start_location[0],start_location[1],telemetry["GPS_Latitude"],telemetry["GPS_Longitude"],GOOGLE_API_KEY)
    if not ((SECRET == None or SECRET == "") or SECRET == " "):
        url = sign_url(url, SECRET)
    try:
        response = urllib.request.urlopen(url,timeout=10)####
    except urllib.error.URLError as error:
        print('Data of map is not retrieved because {}\nURL: {}'.format(error, url))
    except:
        print("Unknown map error! (Possibly a socket timeout)")
    else:
        #If we didn't get an IOError then parse the result
        image = np.asarray(bytearray(response.read()), dtype="uint8")
        map_image = cv2.imdecode(image, cv2.IMREAD_COLOR)

def camThread(conn):
    global VIDEO_FPS
    videosizescale = conn.recv()
    camId = conn.recv()
    cap = cv2.VideoCapture(camId, cv2.CAP_DSHOW)
    #Recorder for the live video feed
    recorder = cv2.VideoWriter("Data/LiveVideo.mp4", cv2.VideoWriter_fourcc(*"mp4v"),
                               VIDEO_FPS, (int(cap.get(3)), int(cap.get(4))))
    if cap.isOpened():
        camClock = pygame.time.Clock()
        while True:
            camClock.tick(VIDEO_FPS)
            if conn.poll():
                camId = conn.recv()
                if camId == -1:
                    #Save the video properly
                    cap.release()
                    recorder.release()
                    return
                elif not(camId == "NaN"):
                    cap.release()
                    cap = cv2.VideoCapture(camId, cv2.CAP_DSHOW)
            #Get the next video frame from OTG Reciever
            suc,frame = cap.read()
            if suc:
                recorder.write(frame)
                video_frame = cv2.resize(frame[:,:,::-1],(int(frame.shape[1]*videosizescale),int(frame.shape[0]*videosizescale)))
                conn.send(video_frame)


if __name__ == '__main__':
    # On Windows calling this function is necessary.
    mp.freeze_support()
    print(__doc__)
    ############## Arg Parser ##############################
    argparser = argparse.ArgumentParser(description='METU Modeluydu Ground Station Control')

    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='Print debug information')
    argparser.add_argument(
        '-r', '--res',
        metavar='WIDTHxHEIGHT',
        default='1920x1080',
        help='Window resolution (default: 1920x1080)')
    argparser.add_argument(
        '-f','--fullscreen',
        action='store_true',
        dest='fullscreen',
        help='Open in fullscreen mode')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    ############## Background Texture ######################
    # positions        colors               texture coords
    rectangle = [-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0,
                 1.0, -1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
                 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0,
                 -1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0]

    # convert to 32bit float

    rectangle = np.array(rectangle, dtype=np.float32)

    indicesBackground = [0, 1, 2,
                         2, 3, 0]

    indicesBackground = np.array(indicesBackground, dtype=np.uint32)

    VERTEX_SHADER = """

        #version 330

        in vec3 position;
        in vec3 color;
        in vec2 InTexCoords;

        out vec3 newColor;
        out vec2 OutTexCoords;

        void main() {

            gl_Position = vec4(position, 1.0);
            gl_Position.z = gl_Position.w;
            newColor = color;
            OutTexCoords = InTexCoords;

        }


       """

    FRAGMENT_SHADER = """
        #version 330

        in vec3 newColor;
        in vec2 OutTexCoords;

        out vec4 outColor;
        uniform sampler2D samplerTex;

        void main() {

            outColor = texture(samplerTex, OutTexCoords);

        }

        """

    vertex_src = """
        # version 330

        layout(location = 0) in vec3 a_position;
        layout(location = 1) in vec2 a_texture;
        layout(location = 2) in vec3 a_normal;

        uniform mat4 model;
        uniform mat4 projection;
        uniform mat4 view;

        out vec2 v_texture;

        void main()
        {
            gl_Position = projection * view * model * vec4(a_position, 1.0);
            v_texture = a_texture;
        }
        """

    fragment_src = """
        # version 330

        in vec2 v_texture;

        out vec4 out_color;

        uniform sampler2D s_texture;

        void main()
        {
            out_color = texture(s_texture, v_texture);
        }
        """

    #Initialize pygame
    pygame.init()
    pygame.font.init()
    #Get the full display resolution for fullscreen
    if args.fullscreen:
        info = pygame.display.Info()
        resolution = (info.current_w, info.current_h)
    else:
        resolution = (args.width,args.height)
    #Load images for GUI
    background1 = cv2.imread(resource_path("assets/1.png"))
    background1 = cv2.cvtColor(background1,cv2.COLOR_BGR2RGB)
    background1 = cv2.resize(background1,resolution)
    background2 = cv2.imread(resource_path("assets/2.png"))
    background2 = cv2.cvtColor(background2,cv2.COLOR_BGR2RGB)
    #Make the background of telemetry black for better visibility
    background2[60:530,60:520,:] = 0
    background2 = cv2.resize(background2,resolution)
    #Get the list of available camera devices
    index = 0
    cam_array = []
    while True:
        cap = cv2.VideoCapture(index,cv2.CAP_DSHOW)
        if not cap.read()[0]:
            break
        else:
            cam_array.append(index)
        cap.release()
        index += 1
    if args.debug:
        print("Cam device indexes available: ".format(cam_array))
    #GUI Declairations
    texts = []
    buttons = []
    selectors = []
    progressbars = []
    resetSat = False
    cmd_send_state = True
    videosendstate = False
    cmd_send = []
    mouse_pos_scale = np.array([1.0,1.0])

    create_graphs()
    plot_data = {"packageCounter":[],"pressure":[],"altitude":[],"speed":[],"temperature":[],"batteryVoltage":[],"revolution":[]}
    main = False
    mapsize = int(resolution[1]*(400/1080))
    videosizescale = resolution[1]*(0.8/1080)

    units = ["","","","","","","","","mBar","m","cm/s","C degrees","V","degrees","degrees","m","","degrees","degrees","degrees","",""]
    telemetry = {"teamNumber":0,"packageCounter":0,"day":0,"month":0,"year":0,"hour":0,"minute":0,"second":0,"pressure":0,"altitude":0,"speed":0,"temperature":0,
                "batteryVoltage":0,"GPS_Latitude":0,"GPS_Longitude":0,"GPS_Altitude":0,
                "status":0,"pitch":0,"roll":0,"yaw":0,"revolution":0,"videoTransmission":0}
    telemetrystruct = "<HHBBHBBBHHhHBffhBbbbBB"

    if not os.path.exists('Data'):
        os.makedirs("Data")
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    datapath = 'Data/Tele_' + now + '.csv'
    videopath = resource_path('Data/SampleVideo_1280x720_1mb.mp4')
    #Open a .csv file to record telemetry data
    f = open(datapath,'w+',newline='')
    writer = csv.DictWriter(f,fieldnames=[key for key in telemetry])
    writer.writeheader()

    try:
        #Setting display
        display = pygame.display.set_mode(
            resolution,
            pygame.HWSURFACE | pygame.DOUBLEBUF | pygame.OPENGL | pygame.RESIZABLE | (pygame.FULLSCREEN if args.fullscreen else 0))

        openglinit()
        pygame.display.set_caption('METU Ground Station Control')
        iconsurf = pygame.image.load(resource_path("assets/metuappicon.png"))
        pygame.display.set_icon(iconsurf)
        #Determening fonts
        bigfont = pygame.font.SysFont(pygame.font.get_fonts()[0], int(resolution[1]*(30/1080)))
        midfont = pygame.font.SysFont(pygame.font.get_fonts()[0], int(resolution[1]*(26/1080)))
        smallfont = pygame.font.SysFont(pygame.font.get_fonts()[0], int(resolution[1]*(20/1080)))
        telemetrytitlefont = pygame.font.SysFont('consolas', int(resolution[1]*(24/1080)))
        telemetryfont = pygame.font.SysFont('consolas', int(resolution[1]*(18/1080)))
        #Load the default video file
        videobytes = getvideobytes(videopath)
        #Create Serial Port
        serialPort = serial.Serial(bytesize=8, timeout=2,write_timeout=0, stopbits=serial.STOPBITS_ONE)
        #Creating GUI elements
        for i in range(len(telemetry)):
            texts.append(Text(resolution[0]*(5/32),resolution[1]*((30+7*i)/400),"",telemetryfont,allign = 'center'))
        texts.append(Text(resolution[0]*(280/1920),resolution[1]*(35/1080),"Telemetry Data",telemetrytitlefont))
        texts.append(Text(resolution[0]*(960/1920),resolution[1]*(190/1080),"Live Feed",midfont))
        texts.append(Text(resolution[0]*(960/1920),resolution[1]*(665/1080),"Command Console",midfont))
        texts.append(Text(resolution[0]*(1640/1920),resolution[1]*(550/1080),"Location",midfont))
        texts.append(Text(resolution[0]*(1640/1920),resolution[1]*(30/1080),"Orientation",midfont))
        buttonwh = (resolution[0]*(120/1920),resolution[1]*(35/1080))
        buttons.append(Button("eject",resolution[0]*(850/1920),resolution[1]*(900/1080),width=buttonwh[0],height=buttonwh[1],text="Eject Satellite"))
        buttons.append(Button("resetSatellite",resolution[0]*(850/1920),resolution[1]*(950/1080),width=buttonwh[0],height=buttonwh[1],text="Reset Satellite"))
        buttons.append(Button("motorTest",resolution[0]*(850/1920),resolution[1]*(800/1080),width=buttonwh[0],height=buttonwh[1],text="Motor Test"))
        buttons.append(Button("startVideoSend",resolution[0]*(1100/1920),resolution[1]*(750/1080),width=buttonwh[0],height=buttonwh[1],text="Start Video Transmission"))
        buttons.append(Button("stopVideoSend",resolution[0]*(1100/1920),resolution[1]*(890/1080),width=buttonwh[0],height=buttonwh[1],text="Stop Video Transmission"))
        buttons.append(Button("resetVideoSend",resolution[0]*(1100/1920),resolution[1]*(950/1080),width=buttonwh[0],height=buttonwh[1],text="Reset Video Transmission"))
        buttons.append(Button("connectPort", resolution[0]*(850/1920), resolution[1]*(750/1080), 
                              width=buttonwh[0], height=buttonwh[1], text="Connect"))
        buttons.append(Button("toggleCamera", resolution[0]*(850/1920), resolution[1]*(850/1080), 
                              width=buttonwh[0], height=buttonwh[1], text="Toggle Recording"))
        #For Selecting Serial Port
        texts.append(Text(resolution[0]*(730/1920),
                          resolution[1]*(710/1080), 'Baudrate:', smallfont))
        selectors.append(Selector(resolution[0]*(730/1920), resolution[1]*(750/1080), width=resolution[0]*(
            80/1920), height=resolution[1]*(20/1080), choices=[9600, 19200, 38400, 57600, 74880, 115200, 230400]))
        texts.append(Text(resolution[0]*(630/1920),
                          resolution[1]*(710/1080), 'COM Port:', smallfont))
        ports = serial.tools.list_ports.comports()
        selectors.append(Selector(resolution[0]*(630/1920), resolution[1]*(750/1080), width=resolution[0]*(
            80/1920), height=resolution[1]*(20/1080), choices=([_port for _port, desc, hwid in sorted(ports)] if ports else ["NaN"])))
        #For Selecting Cam Device
        selectors.append(
            Selector(
                resolution[0] * (1280 / 1920),
                resolution[1] * (190 / 1080),
                width=resolution[0] * (80 / 1920),
                height=resolution[1] * (20 / 1080),
                choices=cam_array or ["NaN"],
            )
        )

        texts.append(Text(resolution[0]*(1190/1920),
                          resolution[1]*(190/1080), 'Camera:', midfont))
        preSelect = selectors[2].getValue()
        buttons.append(Button("selectVideo", resolution[0]*(1100/1920), resolution[1] * (700/1080),
                              height=resolution[1]*(30/1080), text="Select Video File"))
        videoProgressBar = ProgressBar(resolution[0]*(1100/1920), resolution[1]*(820/1080), width=resolution[0]*(300/1920), height=resolution[1]*(25/1080), max=len(videobytes))
        progressbars.append(videoProgressBar)
        video_send_counter = 0
        main = True
        telemetry_update = False
        graphsurf = pygame.Surface((0,0))
        map_image = np.zeros((0,0,3), np.uint8)
        video_frame = np.zeros((0,0,3), np.uint8)
        start_location = None
        t1 = time.time()
        #Preparing the Loop
        done = False
        #Create and start the threads
        x1 = threading.Thread(target=serialThread,args=(1,), daemon=True)
        x1.start()

        camPipe, camProcessPipe = mp.Pipe()
        x2 = mp.Process(target=camThread, args=(camProcessPipe,), daemon=True)
        x2.start()
        camPipe.send(videosizescale)
        camPipe.send(preSelect)

        graphPipe, graphProcessPipe = mp.Pipe()
        x3 = mp.Process(target=plot, args=(graphProcessPipe,), daemon=True)
        x3.start()
        graphPipe.send(resolution)

        while not done:
            #Manipulate user input end exit as requested
            done = eventcheck()
            #Update the value of ProgressBar for video sending process
            videoProgressBar.setValue(video_send_counter)

            #If the serial devices changed, update the serial selector's choices
            if ports != serial.tools.list_ports.comports():
                ports = serial.tools.list_ports.comports()
                selectors[1].updateChoices([_port for _port, desc, hwid in sorted(ports)] if ports else ["NaN"])

            #If the user changed the camera selector, switch the camera
            if preSelect != selectors[2].getValue():
                preSelect = selectors[2].getValue()
                camPipe.send(preSelect)

            if camPipe.poll():
                video_frame = camPipe.recv()
            #If we recieved new telemetry data from the thread, update values accordingly
            if telemetry_update and ((not resetSat) or (telemetry["packageCounter"] == 0)):
                telemetry_update = False

                #Cast the data buffered into the dict
                for i,key in enumerate(telemetry):
                    telemetry[key] = data_buffer[i]
                #Re-scale some data
                telemetry["temperature"] /= 10.0
                telemetry["batteryVoltage"] = 11.8#/= 10.0

                telemetry["GPS_Latitude"] = 39.890052 + np.random.randint(-10,10)/1000000
                telemetry["GPS_Longitude"] = 32.779753 + np.random.randint(-10,10)/1000000
                telemetry["GPS_Altitude"] = 910 + np.random.randint(-10,10)/5

                dt = datetime.now()
                telemetry["year"] = dt.year
                telemetry["month"] = dt.month
                telemetry["day"] = dt.day
                telemetry["hour"] = dt.hour
                telemetry["minute"] = dt.minute
                telemetry["second"] = dt.second

                #Keep last n data as a history for graphs
                for key in plot_data:
                    plot_data[key].append(telemetry[key])
                    if len(plot_data[key]) > plot_length:
                        plot_data[key].pop(0)

                if plot_data["packageCounter"]:
                    graphPipe.send(plot_data)

                #Record first geo data for the map
                if start_location is None:
                    start_location = telemetry["GPS_Latitude"],telemetry["GPS_Longitude"]

                if telemetry["packageCounter"]%5 == 0:
                    x4 = threading.Thread(target=mapThread, args=(1,))
                    x4.start()
                #Save new data into the .csv file
                writer.writerow(telemetry)
                #Create Text objects to display the telemetry data in pygame gui
                for i,key in enumerate(telemetry):
                    if type(telemetry[key]) == float:
                        texts[i].setText("{:20}:{:10.5f} {:9}".format(key, telemetry[key],units[i]))
                    elif type(telemetry[key]) == bool:
                        texts[i].setText("{:20}:{:>10} {:9}".format(key, telemetry[key].__str__(),units[i]))
                    elif type(telemetry[key]) == int:
                        texts[i].setText("{:20}:{:10d} {:9}".format(key, telemetry[key],units[i]))
                    else:
                        texts[i].setText("{:20}: {:19}".format(key, telemetry[key]))

            #If not transmitting video file, send the buffered commands
            if (not videosendstate) and serialPort.is_open:
                for cmd in cmd_send:
                    serialPort.write(cmd)
                    if args.debug:
                        print("Sent the {} command".format(cmd))
                cmd_send = []

            #Do drawings for the pygame gui
            render(display)
            gc.collect()
    finally:
        #Release resources at exit
        camPipe.send(-1)
        if camPipe.poll():
            a = camPipe.recv()
        x2.join(timeout=2)
        f.close()
        pygame.quit()
