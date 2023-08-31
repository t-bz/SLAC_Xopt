from os import path

import versioneer

from setuptools import find_packages, setup

cur_dir = path.abspath(path.dirname(__file__))

with open(path.join(cur_dir, "requirements.txt"), "r") as f:
    requirements = f.read().split()


setup(
    name="slac_xopt",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    package_dir={"slac_xopt": "slac_xopt"},
    url="https://github.com/roussel-ryan/SLAC_Xopt",
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.7",
)
