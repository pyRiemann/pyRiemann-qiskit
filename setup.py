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
      python_requires=">=3.7",
      install_requires=['cython', 'pyriemann==0.3',
                        'qiskit_machine_learning==0.5.0',
                        'qiskit-ibmq-provider==0.19.2',
                        'qiskit-optimization==0.4.0',
                        'cvxpy==1.2.3',
                        'scipy==1.7.3',
                        'docplex>=2.21.207',
                        'firebase_admin==6.0.1',
                        'tqdm'
                        ],
      extras_require={'docs': ['sphinx-gallery', 'sphinx-bootstrap_theme', 'numpydoc', 'mne', 'seaborn', 'moabb>=0.4.6'],
                      'tests': ['pytest', 'seaborn', 'flake8', 'mne', 'pooch', 'tqdm']},
      zip_safe=False,
)
