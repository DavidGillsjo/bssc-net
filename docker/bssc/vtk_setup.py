from setuptools import setup



setup(
    name='vtk',
    version='8.2.0',
    author='VTK Community',
    author_email='vtk-developers@vtk.org',
    # packages=['vtk'],
    # package_dir={'vtk': 'vtk'},
    package_dir={'':'lib/python3.6/dist-packages'},
    py_modules=['vtk'],
    package_data={'': ['vtkmodules']},
    description='VTK is an open-source toolkit for 3D computer graphics, image processing, and visualization',
    long_description='VTK is an open-source, cross-platform library that provides developers with '
                     'an extensive suite of software tools for 3D computer graphics, image processing,'
                     'and visualization. It consists of a C++ class library and several interpreted interface '
                     'layers including Tcl/Tk, Java, and Python. VTK supports a wide variety of visualization '
                     'algorithms including scalar, vector, tensor, texture, and volumetric methods, as well as '
                     'advanced modeling techniques such as implicit modeling, polygon reduction, mesh '
                     'smoothing, cutting, contouring, and Delaunay triangulation. VTK has an extensive '
                     'information visualization framework and a suite of 3D interaction widgets. The toolkit '
                     'supports parallel processing and integrates with various databases on GUI toolkits such '
                     'as Qt and Tk.',
    classifiers=[
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: C++",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries",
        "Operating System :: Android",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: Unix",
        "Operating System :: MacOS"
        ],
    license='BSD',
    keywords='VTK visualization imaging',
    url=r'https://vtk.org/',
    install_requires=[
    ]
    )
