CMAKE_POLICY(SET CMP0011 NEW)

FIND_PACKAGE(umpire REQUIRED)

KOKKOS_IMPORT_TPL(umpire INTERFACE)

