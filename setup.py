from setuptools import setup, find_packages
import versioneer

if __name__ == '__main__':
    setup(name='detection-utils',
          packages=find_packages(where='src'),
          package_dir={'': 'src'},
          version=versioneer.get_version(),
          cmdclass=versioneer.get_cmdclass(),
          )
