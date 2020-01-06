/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_UMPIRESPACE_HPP
#define KOKKOS_UMPIRESPACE_HPP

#include <cstring>
#include <string>
#include <iosfwd>
#include <typeinfo>

#include <Kokkos_Core_fwd.hpp>
#include <Kokkos_Concepts.hpp>
#include <Kokkos_MemoryTraits.hpp>
#include <Kokkos_HostSpace.hpp>

#include <impl/Kokkos_Traits.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_SharedAlloc.hpp>

/*--------------------------------------------------------------------------*/

namespace Kokkos {

/// \class UmpireSpace
/// \brief Memory management for host memory.
///
/// UmpireSpace is a memory space that governs host memory.  "Host"
/// memory means the usual CPU-accessible memory.
class UmpireSpace {
 public:
  //! Tag this class as a kokkos memory space
  typedef UmpireSpace memory_space;
  typedef size_t size_type;

  /// \typedef execution_space
  /// \brief Default execution space for this memory space.
  ///
  /// Every memory space has a default execution space.  This is
  /// useful for things like initializing a View (which happens in
  /// parallel using the View's default execution space).
#if defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_OPENMP)
  typedef Kokkos::OpenMP execution_space;
#elif defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_THREADS)
  typedef Kokkos::Threads execution_space;
#elif defined(KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_HPX)
  typedef Kokkos::Experimental::HPX execution_space;
//#elif defined( KOKKOS_ENABLE_DEFAULT_DEVICE_TYPE_QTHREADS )
//  typedef Kokkos::Qthreads  execution_space;
#elif defined(KOKKOS_ENABLE_OPENMP)
  typedef Kokkos::OpenMP execution_space;
#elif defined(KOKKOS_ENABLE_THREADS)
  typedef Kokkos::Threads execution_space;
//#elif defined( KOKKOS_ENABLE_QTHREADS )
//  typedef Kokkos::Qthreads  execution_space;
#elif defined(KOKKOS_ENABLE_HPX)
  typedef Kokkos::Experimental::HPX execution_space;
#elif defined(KOKKOS_ENABLE_SERIAL)
  typedef Kokkos::Serial execution_space;
#else
#error \
    "At least one of the following host execution spaces must be defined: Kokkos::OpenMP, Kokkos::Threads, Kokkos::Qthreads, or Kokkos::Serial.  You might be seeing this message if you disabled the Kokkos::Serial device explicitly using the Kokkos_ENABLE_Serial:BOOL=OFF CMake option, but did not enable any of the other host execution space devices."
#endif

  //! This memory space preferred device_type
  typedef Kokkos::Device<execution_space, memory_space> device_type;

  /**\brief  Default memory space instance */
  UmpireSpace();
  UmpireSpace(UmpireSpace&& rhs)      = default;
  UmpireSpace(const UmpireSpace& rhs) = default;
  UmpireSpace& operator=(UmpireSpace&&) = default;
  UmpireSpace& operator=(const UmpireSpace&) = default;
  ~UmpireSpace()                           = default;

  /**\brief  Allocate untracked memory in the space */
  void* allocate(const size_t arg_alloc_size) const;

  /**\brief  Deallocate untracked memory in the space */
  void deallocate(void* const arg_alloc_ptr, const size_t arg_alloc_size) const;

  /**\brief Return Name of the MemorySpace */
  static constexpr const char* name() { return m_name; }

 private:
  static constexpr const char* m_name = "Host";
  friend class Kokkos::Impl::SharedAllocationRecord<Kokkos::UmpireSpace, void>;
};

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {


template <>
struct MemorySpaceAccess<Kokkos::HostSpace, Kokkos::UmpireSpace> {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy = true };
};

template <>
struct MemorySpaceAccess<Kokkos::UmpireSpace, Kokkos::HostSpace> {
  enum { assignable = false };
  enum { accessible = false };
  enum { deepcopy = true };
};

}  // namespace Impl

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <>
class SharedAllocationRecord<Kokkos::UmpireSpace, void>
    : public SharedAllocationRecord<void, void> {
 private:
  friend Kokkos::UmpireSpace;

  typedef SharedAllocationRecord<void, void> RecordBase;

  SharedAllocationRecord(const SharedAllocationRecord&) = delete;
  SharedAllocationRecord& operator=(const SharedAllocationRecord&) = delete;

  static void deallocate(RecordBase*);

#ifdef KOKKOS_DEBUG
  /**\brief  Root record for tracked allocations from this UmpireSpace instance */
  static RecordBase s_root_record;
#endif

  const Kokkos::UmpireSpace m_space;

 protected:
  ~SharedAllocationRecord();
  SharedAllocationRecord() = default;

  SharedAllocationRecord(
      const Kokkos::UmpireSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size,
      const RecordBase::function_type arg_dealloc = &deallocate);

 public:
  inline std::string get_label() const {
    return std::string(RecordBase::head()->m_label);
  }

  KOKKOS_INLINE_FUNCTION static SharedAllocationRecord* allocate(
      const Kokkos::UmpireSpace& arg_space, const std::string& arg_label,
      const size_t arg_alloc_size) {
#if defined(KOKKOS_ACTIVE_EXECUTION_MEMORY_SPACE_HOST)
    return new SharedAllocationRecord(arg_space, arg_label, arg_alloc_size);
#else
    return (SharedAllocationRecord*)0;
#endif
  }

  /**\brief  Allocate tracked memory in the space */
  static void* allocate_tracked(const Kokkos::UmpireSpace& arg_space,
                                const std::string& arg_label,
                                const size_t arg_alloc_size);

  /**\brief  Reallocate tracked memory in the space */
  static void* reallocate_tracked(void* const arg_alloc_ptr,
                                  const size_t arg_alloc_size);

  /**\brief  Deallocate tracked memory in the space */
  static void deallocate_tracked(void* const arg_alloc_ptr);

  static SharedAllocationRecord* get_record(void* arg_alloc_ptr);

  static void print_records(std::ostream&, const Kokkos::UmpireSpace&,
                            bool detail = false);
};

}  // namespace Impl

}  // namespace Kokkos

//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

template <class ExecutionSpace>
struct DeepCopy<Kokkos::UmpireSpace, Kokkos::HostSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
     // Perform deep copy from host to umpire
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    // Perform deep copy from host to umpire
    exec.fence();
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::HostSpace, Kokkos::UmpireSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
     // Perform Deep Copy from umpire to host
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    // Perform Deep Copy from umpire to host
    exec.fence();
  }
};

template <class ExecutionSpace>
struct DeepCopy<Kokkos::UmpireSpace, Kokkos::UmpireSpace, ExecutionSpace> {
  DeepCopy(void* dst, const void* src, size_t n) {
     // Perform deep copy from host to umpire
  }

  DeepCopy(const ExecutionSpace& exec, void* dst, const void* src, size_t n) {
    exec.fence();
    // Perform deep copy from host to umpire
    exec.fence();
  }
};

}  // namespace Impl

}  // namespace Kokkos

#endif  // #define KOKKOS_UMPIRESPACE_HPP