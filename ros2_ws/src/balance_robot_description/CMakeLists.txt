cmake_minimum_required(VERSION 3.8)
project(balance_robot_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY models meshes urdf rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
