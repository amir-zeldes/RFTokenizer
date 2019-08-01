from setuptools import setup, find_packages

setup(
  name = 'rftokenizer',
  packages = find_packages(),
  version = '1.0.1',
  description = 'A character-wise tokenizer for morphologically rich languages',
  author = 'Amir Zeldes',
  author_email = 'amir.zeldes@georgetown.edu',
  package_data = {'':['README.md','LICENSE.md','requirements.txt'],'rftokenizer':['data/*','pred/*','models/*']},
  url = 'https://github.com/amir-zeldes/RFTokenizer',
  install_requires=["scikit-learn","numpy","pandas","xgboost","hyperopt"],
  license='Apache License, Version 2.0',
  download_url = 'https://github.com/amir-zeldes/RFTokenizer/releases/tag/v1.0.1',
  keywords = ['NLP', 'tokenization', 'segmentation', 'morphology', 'morphological', 'Hebrew', 'Arabic', 'Coptic', 'word', 'splitting'],
  classifiers = ['Programming Language :: Python',
'Programming Language :: Python :: 2',
'Programming Language :: Python :: 3',
'License :: OSI Approved :: Apache Software License',
'Operating System :: OS Independent'],
)
