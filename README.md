# METU-GroundStation
### The METU Modeluydu Ground Station Control

![Supported Python versions](https://img.shields.io/badge/python-3.7-green.svg)
![Supported Python versions](https://img.shields.io/badge/python-3.9-green.svg)
  Tested on Windows 10.

[Python Dependencies](requirements.txt):
- numpy
- opencv-python
- matplotlib
- pygame
- pillow
- pyserial
- pyopengl
- pyopengl-accelerate
- pyrr

You can install the required packages by writing `pip install -r requirements.txt` in the command line.

Additional Required Libraries (Supplied in /classes folder):
[urlsigner](https://github.com/googlemaps/url-signing), tangui (Custom)

Required files are supplied under `/assets` and `/Data` folders.

Main code: `groundstation_v2.py`

Cython code: `groundstation_v2.pyx`

Executable: `dist\METUgroundstation.exe`

### Usage

__Usage from the command line:__
```
python groundstation_v2.py [-h] [-v] [-r WIDTHxHEIGHT] [-f]

Optional Arguments:
  -h, --help            show this help message and exit
  -v, --verbose         Print debug information
  -r WIDTHxHEIGHT, --res WIDTHxHEIGHT
                        Window resolution (default: 1920x1080)
  -f, --fullscreen      Open in fullscreen mode
```

__Usage with Cython:__

Before doing anything, you have to follow the steps in the [Cythonizing](https://github.com/tanacar2002/METU-GroundStation#cythonizing) section.

Type `python` to open the interpreter.

Then in the interpreter type `import groundstation_v2`.

__Usage with .exe:__

Just double click on it :)

### Cythonizing and Converting to an EXE

#### Cythonizing

Using [Cython](https://cython.org/) to partially compile our code (meaning you still need an interpreter) improves performance, while converting to an .exe makes the code dependency free (including Python).

Before we start, to cythonize and create an executable, you still need the aforementioned dependencies, a `setup.py` script which is provided to you and 2 packages called [Cython](https://cython.org/) and [PyInstaller](https://www.pyinstaller.org/). You might also need a C/C++ compiler such as [MSVC](https://www.microsoft.com/en-us/download/details.aspx?id=58317&WT.mc_id=DX_MVP4025064).

Use `pip install cython` to install the Cython package. PyInstaller package is only needed to convert the code to an executable. So if you also want to convert your file into an executable, also type `pip install pyinstaller`.

The code we use with Cython is in the `groundstation_v2.pyx` file. The code is the same with `groundstation.py` but all the classes that are not pip packages are not imported but directly included. Also, to take more advantage of Cython, a special cdef keyword is used to declare the variables. You can see some of them even have a type. If you want to see the changes in the executable, you must apply them to the `groundstation_v2.pyx` file.

In order to cythonize our code, we run `python .\setup.py build_ext --inplace`. This will create a build folder which we don't worry about and 2 files that we will use, named `groundstation_v2.c` and `groundstation_v2.cp37-win_amd64.pyd`. File extentions migth differ depending on your system configuration.

Now at this point we can open python interpreter from our terminal and type `import groundstation_v2` and run the program with relatively better performance than pure Python. But we still need all the dependent python packages installed to do so. If the program failed, **don't** move on to the [Converting to an EXE](https://github.com/tanacar2002/METU-GroundStation#converting-to-an-exe) section and try to solve the problems first. Alternatively if you want to create an executable without cythonizing, you should use `prog_without_cython.py` instead of `prog.py` at the [Converting to an EXE](https://github.com/tanacar2002/METU-GroundStation#converting-to-an-exe) section.

#### Converting to an EXE

We convert our cython code to an exe with [PyInstaller](https://www.pyinstaller.org/) package. So if you haven't got it already, type `pip install pyinstaller` to install the package.

We need a .py script that will import the "groundstation_v2" package like we did in the interpreter. Such file is provided to you named `prog.py`. If you look inside the file, you'll see all the packages used in the "groundstation_v2" package are imported. This is for Pyinstaller to correctly detect our dependencies and include them in our executable. Alternatively you can use `prog_without_cython.py` instead, if you have problems with cython.

To add some additional data such as pictures and object models to our executable, we need to specify them. We do this when we type the command to create our exe. You should also use the `resource_path("yourfolder\\yourfile.png")` function in the `groundstation.pyx` code whenever you want to use a file. Only that way the executable will be able to use the additional data at runtime. Same goes for `prog_without_cython.py` if your using that.

To create the executable, type 
```
pyinstaller -F -w -i assets/ModelUydu.ico --add-data "assets/metuappicon.png;./assets" --add-data "assets/1.png;./assets" --add-data "assets/2.png;./assets" --add-data "assets/cube.obj;./assets" --add-data "assets/cube.jpg;./assets" --add-data "Data/SampleVideo_1280x720_1mb.mp4;./Data" -n METUgroundstation prog.py
```
in your terminal. Note that you will need to change `prog.py` with `prog_without_cython.py` if you took the route without cython. This command will take some time. After it has finished, your executable will be located in the `\dist` folder.

##### Caveats

One might argue that exe file runs a bit slower then importing cython code directly from the python interpreter. Also .exe takes some time to load the program. You should try both and compare them and use the one that suits your target usage the most. My advice would be to only use the exe if you need a dependency free and easily distributable program.

###### Important:
In order for map function to work, you need to get an [Google API Key](https://developers.google.com/maps/gmp-get-started?hl=en) for free and copy it in GOOGLE_API_KEY = "".
If you want to sign your url request, which you don't have to, paste your signature in SECRET = "".
