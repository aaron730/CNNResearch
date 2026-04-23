from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="captcha-solver",
    version="0.1.0",
    description="CNN-based image classification CAPTCHA solver for research",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "captcha-solver=captcha_solver.cli:main",
        ],
    },
)
