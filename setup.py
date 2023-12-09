# Copyright 2023 The python_auditory_toolbox Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2023 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Malcolm Notes:
# To create distributions:  python3 -m build
# To upload distribution: python3 -m twine upload dist/*
#   Use __token__ as the User name and an API key for the password
# To test in a new environment: conda create --name test
# Then: conda activate test
# Then: pip install python_auditory_toolbox
# Then: python3
#   from python_auditory_toolbox import auditory_toolbox_jax as pat
#   pat.ErbSpace()

"""Create the python_auditory_toolbox package files.
"""

import setuptools

with open('README.md', 'r', encoding='utf-8') as fh:
  long_description = fh.read()

setuptools.setup(
  name='python_auditory_toolbox',
  version='1.0.3',
  author='Malcolm Slaney',
  author_email='malcolm@ieee.org',
  description='Several simple auditory models in JAX, Numpy and Torch',
  long_description=long_description,
  long_description_content_type='text/markdown',
  url='https://github.com/MalcolmSlaney/python_auditory_toolbox',
  packages=['python_auditory_toolbox'],
  classifiers=[
    'Programming Language :: Python :: 3',
    'License :: OSI Approved :: Apache Software License',
    'Operating System :: OS Independent',
    'Topic :: Multimedia :: Sound/Audio :: Analysis',
  ],
  python_requires='>=3.6',
  install_requires=[
    'absl-py',
    'numpy',
    'jax', 
    'jaxlib',
    'matplotlib',
    'scipy',
    'torch',
    'torchaudio',
  ],
  include_package_data=True,  # Using the files specified in MANIFEST.in
)
