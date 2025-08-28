CMAKE_MINIMUM_REQUIRED(VERSION 3.15.0)

# 设置 C++ 标准为 C++17
# SET(CMAKE_CXX_STANDARD 11)
# SET(CMAKE_CXX_STANDARD_REQUIRED ON)

# 设置项目名称
PROJECT(mysolver)

# 设置 IBAMR 的安装路径
SET(IBAMR_ROOT "/public21/home/sc93921/software-sc93921/IBAMR/IBAMR-0.16.0")
# 设置IBAMR_DIR指向安装目录中的配置文件位置
SET(IBAMR_DIR "${IBAMR_ROOT}/install3/lib64/cmake/ibamr")

# 添加 SILO 配置 - 使用静态库
SET(SILO_ROOT "/public21/home/sc93921/software-sc93921/silo")
SET(SILO_INCLUDE_DIRS "${SILO_ROOT}/include")
SET(SILO_LIBRARIES "${SILO_ROOT}/lib/libsilo.a")  # 使用静态库

# 添加调试信息
MESSAGE(STATUS "IBAMR_ROOT: ${IBAMR_ROOT}")
MESSAGE(STATUS "IBAMR_DIR: ${IBAMR_DIR}")
MESSAGE(STATUS "SILO_ROOT: ${SILO_ROOT}")
MESSAGE(STATUS "SILO_INCLUDE_DIRS: ${SILO_INCLUDE_DIRS}")
MESSAGE(STATUS "SILO_LIBRARIES: ${SILO_LIBRARIES}")

# 检查文件是否存在
IF(EXISTS "${SILO_LIBRARIES}")
    MESSAGE(STATUS "SILO library found: ${SILO_LIBRARIES}")
ELSE()
    MESSAGE(FATAL_ERROR "SILO library not found: ${SILO_LIBRARIES}")
ENDIF()

# 查找Python库
FIND_PACKAGE(PythonLibs 3 REQUIRED)

# 添加JSON库，并要求最低版本为 3.7.0
FIND_PACKAGE(nlohmann_json 3.7.0 REQUIRED)

# 列出源文件
SET(SOURCE_FILES main.cpp IBEELKinematics.cpp IBCylinderKinematics.cpp)

# 创建可执行文件
ADD_EXECUTABLE(main2d ${SOURCE_FILES})

# 查找 IBAMR 包
FIND_PACKAGE(IBAMR 0.16.0 REQUIRED HINTS ${IBAMR_DIR})

# 设置 PETSc 的安装路径
SET(PETSC_DIR "/public21/home/sc93921/software-sc93921/IBAMR/petsc-debugging/petsc-3.22.2/install")
SET(PETSC_ARCH "")

# 查找 PETSc 的配置文件
FIND_FILE(PETSC_VARIABLES_FILE petscvariables HINTS ${PETSC_DIR}
  PATH_SUFFIXES conf lib/petsc/conf)
IF(${PETSC_VARIABLES_FILE} STREQUAL "PETSC_VARIABLES_FILE-NOTFOUND")
  MESSAGE(FATAL_ERROR "Unable to find the petscvariables configuration file in ${PETSC_DIR}")
ELSE()
  # 提取 PETSc 的包含路径
  FILE(STRINGS ${PETSC_VARIABLES_FILE} _petsc_raw_includes REGEX "^PETSC_CC_INCLUDES =.*")
  IF ("${_petsc_raw_includes}" STREQUAL "")
    MESSAGE(FATAL_ERROR
      "The configuration script was unable to find the list of PETSc include directories in the file ${PETSC_VARIABLES_FILE}")
  ENDIF()

  # 处理包含路径
  STRING(REGEX REPLACE "^PETSC_CC_INCLUDES =(.*)" "\\1" _petsc_raw_includes ${_petsc_raw_includes})
  SEPARATE_ARGUMENTS(_petsc_raw_includes)
  FOREACH(_include ${_petsc_raw_includes})
    STRING(REGEX REPLACE "^-I" "" _directory "${_include}")
    LIST(APPEND PETSC_INCLUDE_DIRS ${_directory})
  ENDFOREACH()

  # 提取 PETSc 的库路径
  FILE(STRINGS ${PETSC_VARIABLES_FILE} PETSC_LIBRARIES REGEX "^PETSC_WITH_EXTERNAL_LIB =.*")
  STRING(REGEX REPLACE "^PETSC_WITH_EXTERNAL_LIB =(.*)" "\\1" PETSC_LIBRARIES ${PETSC_LIBRARIES})
  STRING(REGEX REPLACE "\\$.PETSC_DIR./\\$.PETSC_ARCH." "${PETSC_DIR}" PETSC_LIBRARIES ${PETSC_LIBRARIES})
  SEPARATE_ARGUMENTS(PETSC_LIBRARIES)

  # 查找 PETSc 主库
  FIND_LIBRARY(PETSC_LIBRARY REQUIRED NAMES "petsc" HINTS ${PETSC_DIR}/lib)
  LIST(PREPEND PETSC_LIBRARIES ${PETSC_LIBRARY})

  # 打印调试信息
  MESSAGE(STATUS "PETSC INCLUDE DIRS: ${PETSC_INCLUDE_DIRS}")
  MESSAGE(STATUS "PETSC LIBRARIES: ${PETSC_LIBRARIES}")
ENDIF()

# 设置 SAMRAI 的安装路径
SET(SAMRAI_DIR "/public21/home/sc93921/software-sc93921/IBAMR/petsc-debugging/IBSAMRAI2-2025.01.09/install")

# samrai libs:
SET(SAMRAI_INCLUDE_DIRS "${SAMRAI_DIR}/include")

# 查找所有需要的SAMRAI库
SET(_samrai_library_suffixes "algs" "appu" "geom" "hier" "math_std" "mesh" "pdat_std" "solv" "xfer")
SET(SAMRAI2d_LIBRARIES)

# 添加主SAMRAI库
FIND_LIBRARY(SAMRAI_path SAMRAI REQUIRED HINTS "${SAMRAI_DIR}/lib")
LIST(APPEND SAMRAI2d_LIBRARIES ${SAMRAI_path})

# 添加其他SAMRAI组件库
FOREACH(_suffix ${_samrai_library_suffixes})
    SET(_lib_name "SAMRAI2d_${_suffix}")
    FIND_LIBRARY(${_lib_name}_path ${_lib_name} REQUIRED HINTS "${SAMRAI_DIR}/lib")
    LIST(APPEND SAMRAI2d_LIBRARIES ${${_lib_name}_path})
ENDFOREACH()

# 设置 HDF5 的安装路径
SET(HDF5_DIR "/public21/home/sc93921/software-sc93921/IBAMR/hdf5/1.10.6")

# 查找 HDF5 包
FIND_PACKAGE(HDF5 REQUIRED COMPONENTS C)
IF(NOT HDF5_FOUND)
  MESSAGE(FATAL_ERROR "Unable to find a valid HDF5 installation.")
ENDIF()

# 将 IBAMR 的编译标志应用到你的项目
SET(CMAKE_CXX_FLAGS ${IBAMR_CXX_FLAGS})

# 添加包含目录
TARGET_INCLUDE_DIRECTORIES(main2d PRIVATE 
    ${CMAKE_SOURCE_DIR}
    ${PYTHON_INCLUDE_DIRS}
    ${PETSC_INCLUDE_DIRS}
    ${SAMRAI_INCLUDE_DIRS}
    ${HDF5_INCLUDE_DIRS}
    ${SILO_INCLUDE_DIRS}
)

# 在创建可执行文件后添加所有库链接
TARGET_LINK_LIBRARIES(main2d 
    PRIVATE
    IBAMR::IBAMR2d 
    nlohmann_json::nlohmann_json
    ${PYTHON_LIBRARIES}
    ${SAMRAI2d_LIBRARIES}
    ${PETSC_LIBRARIES}
    ${HDF5_LIBRARIES}
    ${SILO_LIBRARIES}
)