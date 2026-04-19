import re
from pathlib import Path

import setuptools


def get_version_from_readme(readme_path="README.md"):
    readme = Path(readme_path)
    content = readme.read_text(encoding="utf-8")
    version_pattern = re.compile(r"^- (\d+\.\d+\.\d+(?:[A-Za-z0-9.+-]*)?):")

    in_release_notes = False
    for line in content.splitlines():
        stripped = line.strip()
        if stripped == "## Release Notes":
            in_release_notes = True
            continue
        if in_release_notes and stripped.startswith("## "):
            break
        if not in_release_notes:
            continue
        match = version_pattern.match(stripped)
        if match:
            return match.group(1)

    raise RuntimeError(
        "Failed to parse package version from README.md. "
        "Please add a bullet like '- 2.1.6: description' under '## Release Notes'."
    )


README = Path("README.md").read_text(encoding="utf-8")


setuptools.setup(
    name="mpdb",
    version=get_version_from_readme(),
    author="zzqq2199",
    author_email="zhouquanjs@qq.com",
    description="A lightweight distributed debugger built on top of ipdb.",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/zzqq2199/mpdb",
    packages=setuptools.find_packages(include=["mpdb*", "dpdb*"]),
    include_package_data=True,
    package_data={
        "mpdb": ["templates/*.html"],
        "dpdb": [],
    },
    install_requires=[
        "ipython",
    ],
    python_requires=">=3.9",
    entry_points={
        "console_scripts": [
            "mpdb=mpdb.__main__:main",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
)
