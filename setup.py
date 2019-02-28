from setuptools import setup, find_packages
import binary_data_factorization

setup(
    name='codes',
    packages=find_packages(),

    author="Eddy",
    
    description="P",
    long_description=open('README.md').read(),
 
 
    include_package_data=True,
 

    url='http://github.com/borenze/binary_data_factorization',
 
    classifiers=[
        "Programming Language :: Python",
        "Development Status :: 1 - Planning",
        "License :: OSI Approved",
        "Natural Language :: French",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 2.7",
        "Topic :: Communications",
    ],
 
 

    license="WTFPL",

 
)
