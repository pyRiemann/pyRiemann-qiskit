import os.path as op

from setuptools import setup, find_packages


# get the version (don't import mne here, so dependencies are not needed)
version = None
with open(op.join('pyriemann_qiskit', '_version.py'), 'r') as fid:
    for line in (line.strip() for line in fid):
        if line.startswith('__version__'):
            version = line.split('=')[1].strip().strip('\'')
            break
if version is None:
    raise RuntimeError('Could not determine version')

with open('README.md', 'r', encoding="utf8") as fid:
    long_description = fid.read()

setup(name='pyriemann-qiskit',
      version=version,
      description='Qiskit wrapper for pyRiemann',
      url='https://pyriemann-qiskit.readthedocs.io',
      author='Gregoire Cattan',
      author_email='gcattan@hotmail.com',
      license='BSD (3-clause)',
      packages=find_packages(),
      long_description=long_description,
      long_description_content_type='text/markdown',
      project_urls={
          'Documentation': 'https://pyriemann.readthedocs.io',
          'Source': 'https://github.com/pyRiemann/pyRiemann-qiskit',
          'Tracker': 'https://github.com/pyRiemann/pyRiemann-qiskit/issues/',
      },
      platforms='any',
      python_requires=">=3.6",
      install_requires=['cython', 'pyriemann @ git+https://github.com/pyRiemann/pyRiemann#egg=pyriemann',
                        'qiskit==0.20.0', 'cvxpy==1.1.12',
                        'scipy==1.5.4; python_version < "3.7.0"',
                        'scipy==1.7.3; python_version >= "3.7.0"',
                        'docplex',
                        'braininvaders2012',
                        'mne==0.20.1',
                        'tqdm'
                        ],
      extras_require={'docs': ['sphinx-gallery', 'sphinx-bootstrap_theme', 'numpydoc',, 'seaborn'],
                      'tests': ['pytest', 'seaborn', 'flake8', 'pooch', 'tqdm']},
      zip_safe=False,
)
