from setuptools import setup, find_packages
import versioneer

if __name__ == '__main__':
    setup(name='detection-utils',
          description='Common functionality for object detection',
          packages=find_packages(where='src', exclude=['tests*']),
          package_dir={'': 'src'},
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          python_requires='>=3.6'
          )
