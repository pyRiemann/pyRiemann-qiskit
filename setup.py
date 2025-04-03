import os.path as op

from setuptools import find_packages, setup

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
          'Documentation': 'https://pyriemann-qiskit.readthedocs.io/',
          'Source': 'https://github.com/pyRiemann/pyRiemann-qiskit',
          'Tracker': 'https://github.com/pyRiemann/pyRiemann-qiskit/issues/',
      },
      platforms='any',
      python_requires=">=3.10",
      install_requires=[
                        'numpy<2.2',
                        'cython',
                        'pyriemann==0.6',
                        'qiskit==1.*',
                        'qiskit_algorithms==0.3.1',
                        'qiskit_machine_learning==0.7.2',
                        'qiskit_ibm_runtime==0.34.0',
                        'qiskit-optimization==0.6.1',
                        'qiskit-aer==0.15.1',
                        'cvxpy==1.6.0',
                        'scipy==1.13.1',
                        'docplex==2.28.240',
                        'firebase_admin==6.6.0',
                        'scikit-learn==1.5.2',
                        'tqdm',
                        'pandas',
                        ],
      extras_require={'docs': [
                                'sphinx-gallery',
                                'sphinx-bootstrap_theme',
                                'numpydoc',
                                'mne==1.9.0',
                                'mne-bids==0.16.0',
                                'seaborn>=0.12.1',
                                'moabb==1.1.1',
                                'imbalanced-learn==0.12.4'
                            ],
                      'tests': ['pytest', 'seaborn', 'flake8', 'mne', 'pooch'],
                      # GPU optimization not available on all platform.
                      # See https://github.com/Qiskit/qiskit-aer/issues/929#issuecomment-691716936
                      'optim': ['qiskit-symb==0.2.0', 'symengine==0.11.0'],
                      'optim_linux': ['qiskit-aer-gpu==0.15.1']},
      zip_safe=False,
)
