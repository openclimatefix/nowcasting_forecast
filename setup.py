""" Usual setup file for package """
from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
install_requires = (this_directory / "requirements.txt").read_text().splitlines()
long_description = (this_directory / "README.md").read_text()
#
# version = open("nowcasting_utils/version.py").readlines()[-1].split()[-1].strip("\"'")

setup(
    name="nowcasting_forecast",
    packages=find_packages(),
    version="0.0.12",
    license="MIT",
    description="Live forecast for the OCF nowcasting project",
    author="Peter Dudfield",
    author_email="peter@openclimatefix.org",
    company="Open Climate Fix Ltd",
    url="https://github.com/openclimatefix/nowcasting_forecast",
    keywords=[
        "artificial intelligence",
        "forecast",
    ],
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
)
