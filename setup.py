from setuptools import setup, find_packages


def main():
    with open("requirements.txt") as f:
        requirements = f.read()

    setup(
        name="dialogueBot",
        version="0.1",
        author="F. Philipp",
        package_dir={"": "src"},
        packages=find_packages("src"),
        description="...",
        install_requires=requirements,
    )


if __name__ == "__main__":
    main()
