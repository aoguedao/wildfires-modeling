from setuptools import setup, find_packages


with open("README.md") as f:
    readme = f.read()

with open("LICENSE") as f:
    license = f.read()

setup(
    name="wfm",
    version="0.1.0",
    description="Wildfires statistical modeling",
    long_description=readme,
    author="Alonso Ogueda",
    author_email="alonso.ogueda@gmail.com",
    url="https://github.com/aoguedao/wfm",
    license=license,
    packages=find_packages(exclude=("tests", "docs"))
)