"""
TODO: In main.py - check version of deepdrive against version in Pipfile
  - if minor version in Pipfile is greater, upgrade sim and extension
    - Delete sim, run python install.py which will install bumped version of deepdrive extension via the new Pipfile
    - Missing sim will be detected and correct major_minor version downloaded
  - else just run install.py which will update any packages via Pipfile / pipenv install
"""