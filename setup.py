from distutils.core import setup

setup(
    name="bio-circuits",
    version="0.0.1",
    author="Tiberiu Tesileanu",
    author_email="ttesileanu@flatironinstitute.org",
    url="https://github.com/ttesileanu/bio-circuits",
    packages=["bio-circuits"],
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "setuptools",
        "torch",
        "torchvision",
        "matplotlib",
        "seaborn",
        "tqdm",
        "pydove",
        "ipykernel",
    ],
)
