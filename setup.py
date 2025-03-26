from setuptools import setup, find_packages

setup(
    name="city2graph",
    version="0.1.0",
    packages=find_packages(),
    description="A package for constructing graphs from geospatial dataset of urban forms and functions",
    author="Yuta Sato",
    author_email="y.sato@liverpool.ac.uk",
    install_requires=[
        "geopandas",
        "shapely",
        "numpy",
        "pandas",
        "networkx",
        "momepy",
        "libpysal",
        "folium",
        "torch_geometric"
    ],
    python_requires=">=3.7",
)