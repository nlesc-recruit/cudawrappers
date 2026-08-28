
####### Expanded from @PACKAGE_INIT@ by configure_package_config_file() #######
####### Any changes to this file will be overwritten by the next CMake run ####
####### The input file was cudawrappers-config.cmake.in                            ########

get_filename_component(PACKAGE_PREFIX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../" ABSOLUTE)

####################################################################################

include(${PACKAGE_PREFIX_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/cudawrappers/cudawrappers-dependencies.cmake)
include(${PACKAGE_PREFIX_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/cudawrappers/cudawrappers-exported.cmake)
include(${PACKAGE_PREFIX_DIR}/${CMAKE_INSTALL_LIBDIR}/cmake/cudawrappers/cudawrappers-helper.cmake)
