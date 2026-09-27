from pathlib import Path

from mlx import extension
from setuptools import setup


if __name__ == "__main__":
    setup(
        name="minimlx-q4-probe",
        version="0.0.0",
        ext_modules=[extension.CMakeExtension("minimlx_q4_probe._ext", sourcedir=Path(__file__).parent)],
        cmdclass={"build_ext": extension.CMakeBuild},
        packages=["minimlx_q4_probe"],
        package_dir={"": str(Path(__file__).parent.parent)},
        package_data={"minimlx_q4_probe": ["*.so", "*.dylib", "*.metallib"]},
        zip_safe=False,
    )
