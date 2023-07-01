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
      python_requires=">=3.8",
      install_requires=[
                        'numpy<1.24',
                        'cython',
                        'pyriemann==0.4',
                        'qiskit_machine_learning==0.6.1',
                        'qiskit-ibm-provider==0.6.1',
                        'qiskit-optimization==0.5.0',
                        'qiskit-aer==0.12.1',
                        'cvxpy==1.3.2',
                        'scipy==1.10.1',
                        'docplex>=2.21.207',
                        'firebase_admin==6.1.0',
                        'tqdm'
                        ],
      extras_require={'docs': ['sphinx-gallery', 'sphinx-bootstrap_theme', 'numpydoc', 'mne', 'seaborn', 'moabb>=0.4.6'],
                      'tests': ['pytest', 'seaborn', 'flake8', 'mne', 'pooch'],
                      # GPU optimization not available on all platform.
                      # See https://github.com/Qiskit/qiskit-aer/issues/929#issuecomment-691716936
                      'optim': ['qiskit-aer-gpu==0.11.2']},
      zip_safe=False,
)
