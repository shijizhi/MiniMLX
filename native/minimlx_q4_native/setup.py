from pathlib import Path

from setuptools import Extension, setup


class CMakeExtension(Extension):
    def __init__(self, name: str):
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(__file__).parent.resolve())


setup(
    name="minimlx-q4-native",
    version="0.0.1",
    packages=["minimlx_q4_native"],
    package_dir={"minimlx_q4_native": "."},
    ext_modules=[CMakeExtension("minimlx_q4_native._ext")],
)
