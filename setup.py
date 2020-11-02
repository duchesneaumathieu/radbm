import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
    
with open("version.txt", "r") as fh:
    version = fh.read()

setuptools.setup(
    name="radbm", # Replace with your own username
    version=version,
    author="Mathieu Duchesneau",
    author_email="duchesneau.mathieu@gmail.com",
    description="Rapid Access Database Benchmark",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/duchesneaumathieu/radbm",
    packages=setuptools.find_packages(),
    data_files=[('radbm/utils/gdrive', ['radbm/utils/gdrive/fileid.json'])],
    classifiers=[
        "Programming Language :: Python :: 3",
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
