
CMAKE_POLICY(SET CMP0011 NEW)

FIND_PACKAGE(umpire REQUIRED)

KOKKOS_TPL_OPTION(UMPIRE On)

##LIST( APPEND KOKKOS_CORE_LINK_EXT_TPL umpire )

KOKKOS_LINK_TPL(kokkoscore PUBLIC IMPORTED_NAME umpire UMPIRE)
