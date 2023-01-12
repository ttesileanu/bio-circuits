from distutils.core import setup

setup(
    name="biocircuits",
    version="0.0.1",
    author="Tiberiu Tesileanu",
    author_email="ttesileanu@flatironinstitute.org",
    url="https://github.com/ttesileanu/bio-circuits",
    packages=["biocircuits"],
    package_dir={"": "src"},
    install_requires=[
        "setuptools",
    ],
)
