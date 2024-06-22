# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

with open("README.md", mode="r", encoding="utf-8") as file:
    readme_text = file.read()

with open("LICENSE", mode="r", encoding="utf-8") as file:
    license_text = file.read()

setup(
    name="fratgpt",
    version="0.0.1",
    description="Tools for fine-tuning a GPT-2 Model on GroupMe messages.",
    long_description=readme_text,
    long_description_content_type="text/markdown",
    author="Jimmy Hoerschgen",
    author_email="jthoerschgen@gmail.com",
    url="https://github.com/jthoerschgen/fratgpt",
    license=license_text,
    packages=find_packages(exclude="tests"),
)
