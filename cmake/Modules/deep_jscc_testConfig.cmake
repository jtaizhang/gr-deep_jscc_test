INCLUDE(FindPkgConfig)
PKG_CHECK_MODULES(PC_DEEP_JSCC_TEST deep_jscc_test)

FIND_PATH(
    DEEP_JSCC_TEST_INCLUDE_DIRS
    NAMES deep_jscc_test/api.h
    HINTS $ENV{DEEP_JSCC_TEST_DIR}/include
        ${PC_DEEP_JSCC_TEST_INCLUDEDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/include
          /usr/local/include
          /usr/include
)

FIND_LIBRARY(
    DEEP_JSCC_TEST_LIBRARIES
    NAMES gnuradio-deep_jscc_test
    HINTS $ENV{DEEP_JSCC_TEST_DIR}/lib
        ${PC_DEEP_JSCC_TEST_LIBDIR}
    PATHS ${CMAKE_INSTALL_PREFIX}/lib
          ${CMAKE_INSTALL_PREFIX}/lib64
          /usr/local/lib
          /usr/local/lib64
          /usr/lib
          /usr/lib64
          )

include("${CMAKE_CURRENT_LIST_DIR}/deep_jscc_testTarget.cmake")

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(DEEP_JSCC_TEST DEFAULT_MSG DEEP_JSCC_TEST_LIBRARIES DEEP_JSCC_TEST_INCLUDE_DIRS)
MARK_AS_ADVANCED(DEEP_JSCC_TEST_LIBRARIES DEEP_JSCC_TEST_INCLUDE_DIRS)
