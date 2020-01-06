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

#include <cstdio>
#include <algorithm>
#include <Kokkos_Macros.hpp>
#include <impl/Kokkos_Error.hpp>
#include <impl/Kokkos_MemorySpace.hpp>
#if defined(KOKKOS_ENABLE_PROFILING)
#include <impl/Kokkos_Profiling_Interface.hpp>
#endif

/*--------------------------------------------------------------------------*/

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>

#include <iostream>
#include <sstream>
#include <cstring>

#include <Kokkos_UmpireSpace.hpp>
#include <impl/Kokkos_Error.hpp>
#include <Kokkos_Atomic.hpp>

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

/* Default allocation mechanism */
UmpireSpace::UmpireSpace() {
}

void *UmpireSpace::allocate(const size_t arg_alloc_size) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  static_assert(
      Kokkos::Impl::is_integral_power_of_two(Kokkos::Impl::MEMORY_ALIGNMENT),
      "Memory alignment must be power of two");

  constexpr uintptr_t alignment      = Kokkos::Impl::MEMORY_ALIGNMENT;
  constexpr uintptr_t alignment_mask = alignment - 1;

  void *ptr = nullptr;

  if (arg_alloc_size) {
      // Over-allocate to and round up to guarantee proper alignment.
      size_t size_padded = arg_alloc_size + sizeof(void *) + alignment;

      void *alloc_ptr = malloc(size_padded);

      if (alloc_ptr) {
        auto address = reinterpret_cast<uintptr_t>(alloc_ptr);

        // offset enough to record the alloc_ptr
        address += sizeof(void *);
        uintptr_t rem    = address % alignment;
        uintptr_t offset = rem ? (alignment - rem) : 0u;
        address += offset;
        ptr = reinterpret_cast<void *>(address);
        // record the alloc'd pointer
        address -= sizeof(void *);
        *reinterpret_cast<void **>(address) = alloc_ptr;
      }
  }

  if ((ptr == nullptr) || (reinterpret_cast<uintptr_t>(ptr) == ~uintptr_t(0)) ||
      (reinterpret_cast<uintptr_t>(ptr) & alignment_mask)) {
    Experimental::RawMemoryAllocationFailure::FailureMode failure_mode =
        Experimental::RawMemoryAllocationFailure::FailureMode::
            AllocationNotAligned;
    if (ptr == nullptr) {
      failure_mode = Experimental::RawMemoryAllocationFailure::FailureMode::
          OutOfMemoryError;
    }

    throw Kokkos::Experimental::RawMemoryAllocationFailure(
        arg_alloc_size, alignment, failure_mode,
        Experimental::RawMemoryAllocationFailure::AllocationMechanism::StdMalloc);
  }

  return ptr;
}

void UmpireSpace::deallocate(void *const arg_alloc_ptr, const size_t
                           ) const {
  if (arg_alloc_ptr) {
      void *alloc_ptr = *(reinterpret_cast<void **>(arg_alloc_ptr) - 1);
      free(alloc_ptr);
  }
}

}  // namespace Kokkos

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {
namespace Impl {

#ifdef KOKKOS_DEBUG
SharedAllocationRecord<void, void>
    SharedAllocationRecord<Kokkos::UmpireSpace, void>::s_root_record;
#endif

void SharedAllocationRecord<Kokkos::UmpireSpace, void>::deallocate(
    SharedAllocationRecord<void, void> *arg_rec) {
  delete static_cast<SharedAllocationRecord *>(arg_rec);
}

SharedAllocationRecord<Kokkos::UmpireSpace, void>::~SharedAllocationRecord() {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::deallocateData(
        Kokkos::Profiling::SpaceHandle(Kokkos::UmpireSpace::name()),
        RecordBase::m_alloc_ptr->m_label, data(), size());
  }
#endif

  m_space.deallocate(SharedAllocationRecord<void, void>::m_alloc_ptr,
                     SharedAllocationRecord<void, void>::m_alloc_size);
}

SharedAllocationHeader *_do_allocation(Kokkos::UmpireSpace const &space,
                                       std::string const &label,
                                       size_t alloc_size) {
  try {
    return reinterpret_cast<SharedAllocationHeader *>(
        space.allocate(alloc_size));
  } catch (Experimental::RawMemoryAllocationFailure const &failure) {
    if (failure.failure_mode() == Experimental::RawMemoryAllocationFailure::
                                      FailureMode::AllocationNotAligned) {
      // TODO: delete the misaligned memory
    }

    std::cerr << "Kokkos failed to allocate memory for label \"" << label
              << "\".  Allocation using MemorySpace named \"" << space.name()
              << " failed with the following error:  ";
    failure.print_error_message(std::cerr);
    std::cerr.flush();
    Kokkos::Impl::throw_runtime_exception("Memory allocation failure");
  }
  return nullptr;  // unreachable
}

SharedAllocationRecord<Kokkos::UmpireSpace, void>::SharedAllocationRecord(
    const Kokkos::UmpireSpace &arg_space, const std::string &arg_label,
    const size_t arg_alloc_size,
    const SharedAllocationRecord<void, void>::function_type arg_dealloc)
    // Pass through allocated [ SharedAllocationHeader , user_memory ]
    // Pass through deallocation function
    : SharedAllocationRecord<void, void>(
#ifdef KOKKOS_DEBUG
          &SharedAllocationRecord<Kokkos::UmpireSpace, void>::s_root_record,
#endif
          Impl::checked_allocation_with_header(arg_space, arg_label,
                                               arg_alloc_size),
          sizeof(SharedAllocationHeader) + arg_alloc_size, arg_dealloc),
      m_space(arg_space) {
#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::allocateData(
        Kokkos::Profiling::SpaceHandle(arg_space.name()), arg_label, data(),
        arg_alloc_size);
  }
#endif
  // Fill in the Header information
  RecordBase::m_alloc_ptr->m_record =
      static_cast<SharedAllocationRecord<void, void> *>(this);

  strncpy(RecordBase::m_alloc_ptr->m_label, arg_label.c_str(),
          SharedAllocationHeader::maximum_label_length);
  // Set last element zero, in case c_str is too long
  RecordBase::m_alloc_ptr
      ->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;
}

//----------------------------------------------------------------------------

void *SharedAllocationRecord<Kokkos::UmpireSpace, void>::allocate_tracked(
    const Kokkos::UmpireSpace &arg_space, const std::string &arg_alloc_label,
    const size_t arg_alloc_size) {
  if (!arg_alloc_size) return (void *)nullptr;

  SharedAllocationRecord *const r =
      allocate(arg_space, arg_alloc_label, arg_alloc_size);

  RecordBase::increment(r);

  return r->data();
}

void SharedAllocationRecord<Kokkos::UmpireSpace, void>::deallocate_tracked(
    void *const arg_alloc_ptr) {
  if (arg_alloc_ptr != 0) {
    SharedAllocationRecord *const r = get_record(arg_alloc_ptr);

    RecordBase::decrement(r);
  }
}

void *SharedAllocationRecord<Kokkos::UmpireSpace, void>::reallocate_tracked(
    void *const arg_alloc_ptr, const size_t arg_alloc_size) {
  SharedAllocationRecord *const r_old = get_record(arg_alloc_ptr);
  SharedAllocationRecord *const r_new =
      allocate(r_old->m_space, r_old->get_label(), arg_alloc_size);

  Kokkos::Impl::DeepCopy<Kokkos::UmpireSpace, Kokkos::UmpireSpace>(
      r_new->data(), r_old->data(), std::min(r_old->size(), r_new->size()));

  RecordBase::increment(r_new);
  RecordBase::decrement(r_old);

  return r_new->data();
}

SharedAllocationRecord<Kokkos::UmpireSpace, void> *
SharedAllocationRecord<Kokkos::UmpireSpace, void>::get_record(void *alloc_ptr) {
  typedef SharedAllocationHeader Header;
  typedef SharedAllocationRecord<Kokkos::UmpireSpace, void> RecordHost;

  SharedAllocationHeader const *const head =
      alloc_ptr ? Header::get_header(alloc_ptr) : (SharedAllocationHeader *)0;
  RecordHost *const record =
      head ? static_cast<RecordHost *>(head->m_record) : (RecordHost *)0;

  if (!alloc_ptr || record->m_alloc_ptr != head) {
    Kokkos::Impl::throw_runtime_exception(
        std::string("Kokkos::Impl::SharedAllocationRecord< Kokkos::UmpireSpace , "
                    "void >::get_record ERROR"));
  }

  return record;
}

// Iterate records to print orphaned memory ...
#ifdef KOKKOS_DEBUG
void SharedAllocationRecord<Kokkos::UmpireSpace, void>::print_records(
    std::ostream &s, const Kokkos::UmpireSpace &, bool detail) {
  SharedAllocationRecord<void, void>::print_host_accessible_records(
      s, "UmpireSpace", &s_root_record, detail);
}
#else
void SharedAllocationRecord<Kokkos::UmpireSpace, void>::print_records(
    std::ostream &, const Kokkos::UmpireSpace &, bool) {
  throw_runtime_exception(
      "SharedAllocationRecord<UmpireSpace>::print_records only works with "
      "KOKKOS_DEBUG enabled");
}
#endif

} // Impl

} // Kokkos
