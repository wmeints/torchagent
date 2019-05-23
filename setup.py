import os
import io
import sys
from setuptools import setup, find_packages, Command
from shutil import rmtree

VERSION = '0.2.3'
REQUIRES_PYTHON = '>=3.5.0'

here = os.path.abspath(os.path.dirname(__file__))


class UploadCommand(Command):
    """Support setup.py upload."""

    description = 'Build and publish the package.'
    user_options = []

    @staticmethod
    def status(s):
        """Prints things in bold."""
        print('\033[1m{0}\033[0m'.format(s))

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            self.status('Removing previous builds…')
            rmtree(os.path.join(here, 'dist'))
        except OSError:
            pass

        self.status('Building Source and Wheel (universal) distribution…')
        os.system(
            '{0} setup.py sdist bdist_wheel --universal'.format(sys.executable))

        self.status('Uploading the package to PyPI via Twine…')
        os.system('twine upload dist/*')

        self.status('Pushing git tags…')
        os.system('git tag v{0}'.format(VERSION))
        os.system('git push --tags')

        sys.exit()


with open('README.rst') as readme_file:
    long_description = readme_file.read()

setup(
    name='torchagent',
    version=VERSION,
    description='Reinforcement learning extensions for PyTorch',
    author='Willem Meints',
    author_email='willem.meints@gmail.com',
    url='https://github.com/wmeints/torchagent',
    license='Apache-2.0',
    long_description=long_description,
    long_description_content_type='text/x-rst',
    python_requires=REQUIRES_PYTHON,
    packages=find_packages(),
    cmdclass={
        'upload': UploadCommand,
    },
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        'License :: OSI Approved :: Apache Software License',
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy'
    ],
)
