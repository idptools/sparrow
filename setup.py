"""
sparrow
Next generation package for sequence parameter calculation
"""
import sys
from setuptools import setup, find_packages
import versioneer


# ................................m
# added for cython compilation
from setuptools.extension import Extension

try:
    from Cython.Build import cythonize
except ModuleNotFoundError:
    print('########################################\n')
    print('Error: Please install cython first:\n\npip install cython\n')
    print('########################################\n')
    exit(1)

try:
    import numpy
except ModuleNotFoundError:

    print('########################################\n')
    print('Error: Please install numpy first:\n\npip install numpy\n')
    print('########################################\n')
    exit(1)
    

extensions = [
    Extension(
        "sparrow.patterning.patterning",
        ["sparrow/patterning/patterning.pyx"],
        include_dirs=[numpy.get_include()], 
    ),
    Extension(
        "sparrow.patterning.kappa",
        ["sparrow/patterning/kappa.pyx"],
        include_dirs=[numpy.get_include()], 
    ),
    Extension(
        "sparrow.patterning.iwd",
        ["sparrow/patterning/iwd.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    ),
    Extension(
        "sparrow.patterning.scd",
        ["sparrow/patterning/scd.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=["-O3"],
    )]


# ................................


short_description = __doc__.split("\n")

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])


setup(
    # Self-descriptive entries which should always be present
    name='sparrow',
    author='Alex Holehouse',
    author_email='alex.holehouse@wustl.edu',
    description=short_description[0],
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='CC-NC-ND',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # external modules
    # ext_modules = cythonize(extensions, compiler_directives={'language_level' : "3"}),
    ext_modules=cythonize(extensions, compiler_directives={
            'boundscheck': False,
            'wraparound': False,
            'cdivision': False,
            'language_level': "3"
        }),
    zip_safe=False,
    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,
    
    # dependencies camparitraj requires
    install_requires=[
        "numpy>=1.14.0",
        "cython",
        "protfasta",
        "metapredict>2",
        "ipython",
        "idptools-parrot",
        "afrc",
        "tqdm",
        "pyfamsa",
    ],
    
    python_requires=">=3.7,<3.12.0",          # Python version restrictions; update to < 3.12 in Nov 2023


    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,

    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='http://www.my_package.com',  # Website
    # install_requires=[],              # Required packages, pulls from pip if needed; do not use for Conda deployment
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor
    # python_requires=">=3.5",          # Python version restrictions

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)
