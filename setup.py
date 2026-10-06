from setuptools import setup

setup(
    name="bci-realtime-project",
    version="0.1.0",
    description="ECoG finger decoding: offline training, Beam/Dataflow streaming decode.",
    packages=["src", "src.beam"],
    python_requires=">=3.10",
)
