project(core)
set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_CXX_FLAGS "-fPIC")


# find linpng
#find_package(PNG REQUIRED)
set(PNG_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/3rdParty/png/include)
include_directories(${PNG_INCLUDE_DIR})
add_definitions(${PNG_DEFINITIONS})
add_library(PNG_LIBRARIES SHARED IMPORTED)
set_property(TARGET PNG_LIBRARIES APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(PNG_LIBRARIES PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_IMPLIB_DEBUG ${CMAKE_SOURCE_DIR}/3rdParty/png/lib/Debug/libpng16.lib
  IMPORTED_LOCATION_DEBUG ${CMAKE_SOURCE_DIR}/3rdParty/png/bin/Debug/libpng16.dll)         
  
set_property(TARGET PNG_LIBRARIES APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(PNG_LIBRARIES PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_IMPLIB_DEBUG ${CMAKE_SOURCE_DIR}/3rdParty/png/lib/Release/libpng16.lib
  IMPORTED_LOCATION_RELEASE ${CMAKE_SOURCE_DIR}/3rdParty/png/bin/Release/libpng16.dll)   
 
 set(PNG_LIBRARY PNG_LIBRARIES)
#if(PNG_FOUND)
#    message("PNG found: ${PNG_LIBRARIES}")
#    include_directories(${PNG_INCLUDE_DIRS})
#   add_definitions(${PNG_DEFINITIONS})
#endif()


set(JPEG_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/3rdParty/jpeg/include)
include_directories(${JPEG_INCLUDE_DIR})
add_library(JPEG_LIBRARIES STATIC IMPORTED)
set_property(TARGET JPEG_LIBRARIES APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(JPEG_LIBRARIES PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG ${CMAKE_SOURCE_DIR}/3rdParty/jpeg/lib/Debug/jpeg.lib)         
  
set_property(TARGET JPEG_LIBRARIES APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(JPEG_LIBRARIES PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE ${CMAKE_SOURCE_DIR}/3rdParty/jpeg/lib/Release/jpeg.lib)  
 
 set(JPEG_LIBRARY JPEG_LIBRARIES)
# find libjpeg
#find_package(JPEG REQUIRED)
if(JPEG_FOUND)
    message("JPEG found: ${JPEG_LIBRARIES}")
    include_directories(${JPEG_INCLUDE_DIR})
endif()


set(TIFF_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/3rdParty/tiff/include)
include_directories(${TIFF_INCLUDE_DIR})
add_definitions(${TIFF_DEFINITIONS})
add_library(TIFF_LIBRARIES SHARED IMPORTED)
set_property(TARGET TIFF_LIBRARIES APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(TIFF_LIBRARIES PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_IMPLIB_DEBUG ${CMAKE_SOURCE_DIR}/3rdParty/tiff/lib/Debug/libtiff.lib
  IMPORTED_LOCATION_DEBUG ${CMAKE_SOURCE_DIR}/3rdParty/tiff/bin/Debug/libtiff.dll)         
  
set_property(TARGET TIFF_LIBRARIES APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(TIFF_LIBRARIES PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_IMPLIB_RELEASE ${CMAKE_SOURCE_DIR}/3rdParty/tiff/lib/Release/libtiff.lib
  IMPORTED_LOCATION_RELEASE ${CMAKE_SOURCE_DIR}/3rdParty/tiff/bin/Release/libtiff.dll)   

 set(TIFF_LIBRARY TIFF_LIBRARIES)
# find libtiff
#find_package(TIFF REQUIRED)
if(TIFF_FOUND)
    message("TIFF found: ${TIFF_LIBRARIES}")
    include_directories(${TIFF_INCLUDE_DIR})
endif()

include_directories("..")
set(HEADERS
       defines.h
        bundle.h
        bundle_io.h
        camera.h
        depthmap.h
        image.h
        image_base.h
        image_color.h
        image_drawing.h
        image_exif.h
        image_io.h
        image_tools.h
        scene.h
        view.h
        mesh.h
        mesh_info.h
        mesh_tools.h
        mesh_io_ply.h
        mesh_io_obj.h
        mesh_io_off.h
        mesh_io_npts.h
        mesh_io_smf.h
        mesh_io_pbrt.h
        )

set(SOURCE_FILES
        bundle.cc
        bundle_io.cc
        camera.cc
        depthmap.cc
        image_exif.cc
        image_io.cc
        image_tools.cc
        scene.cc
        view.cc
        mesh.cc
        mesh_info.cc
        mesh_tools.cc
        mesh_io.cc
        mesh_io_ply.cc
        mesh_io_obj.cc
        mesh_io_off.cc
        mesh_io_npts.cc
        mesh_io_smf.cc
        mesh_io_pbrt.cc
        )
add_library(core ${HEADERS} ${SOURCE_FILES})
target_link_libraries(core util ${PNG_LIBRARY} ${JPEG_LIBRARY} ${TIFF_LIBRARY})

ADD_DEFINITIONS(-DNOMINMAX)

