from distutils.core import setup

from catkin_pkg.python_setup import generate_distutils_setup

# Fetching the package's directory
package_dir = {"": "src"}

# Fetching all Python files including those in subdirectories
packages = ["leap_utils"]  # Update 'leap_utils' with your package name
package_data = {"leap_utils": ["mingrui/*.py"]}  # Include the subdirectory containing Python files

# Generating arguments for setup function
setup_args = generate_distutils_setup(packages=packages, package_dir=package_dir, package_data=package_data)
setup(**setup_args)
