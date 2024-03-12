from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='human-vs-twitter',
      version="0.0.1",
      description="Sentiment Analysis for tweets",
      author="LP-Isaac, Brvsf, rghanem91",
      #url="https://github.com/LP-Isaac/human_vs_twitter",
      install_requires=requirements,
      packages=find_packages(),
      test_suite="tests",
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
