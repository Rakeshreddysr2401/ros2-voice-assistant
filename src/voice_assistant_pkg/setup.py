from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'voice_assistant_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'vosk', 'sounddevice'],
    zip_safe=True,
    maintainer='your_name',
    maintainer_email='your_email@example.com',
    description='Voice Assistant Package for ROS2',
    license='Apache License 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'input_node = voice_assistant_pkg.input_node:main',
            'agent_node = voice_assistant_pkg.agent_node:main',
            'output_node = voice_assistant_pkg.output_node:main',
            'scene_service = voice_assistant_pkg.scene_service:main',
            'test_yolo_camera = voice_assistant_pkg.test_yolo_camera:main',
            'blip_server = voice_assistant_pkg.blip_server:main',
'llava_server = voice_assistant_pkg.llava_server:main',

        ],
    },
)