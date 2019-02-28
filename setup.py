from setuptools import setup, find_packages
import codes

setup(
    name='codes',
    packages=find_packages(),

    author="Eddy",
 
    author_email="lesametlemax@gmail.com",

    description="Proclame la bonne parole de sieurs Sam et Max",
    long_description=open('README.md').read(),
 
 
    include_package_data=True,
 

    url='http://github.com/borenze/codes',
 
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
