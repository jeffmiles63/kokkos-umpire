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

#include "umpire/op/MemoryOperationRegistry.hpp"

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

namespace Kokkos {

namespace Impl {

   void umpire_to_umpire_deep_copy ( void * dst, const void * src, size_t size, bool offset) {
      auto &rm = umpire::ResourceManager::getInstance();
      auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();

      Kokkos::Impl::SharedAllocationHeader * dst_header = 
                     (Kokkos::Impl::SharedAllocationHeader*)dst;

      Kokkos::Impl::SharedAllocationHeader * src_header = 
                     (Kokkos::Impl::SharedAllocationHeader*)src;

      if ( offset ) {
         src_header -= 1;
         dst_header -= 1;
      }

      auto src_alloc_record = rm.findAllocationRecord(src_header);
      std::ptrdiff_t src_offset = reinterpret_cast<char*>(src_header) - 
                                  reinterpret_cast<char*>(src_alloc_record->ptr) +
                                  offset ? sizeof(Kokkos::Impl::SharedAllocationHeader) : 0;

      std::size_t src_size = src_alloc_record->size - src_offset;

      auto dst_alloc_record = rm.findAllocationRecord(dst_header);
      std::ptrdiff_t dst_offset = reinterpret_cast<char*>(dst_header) - 
                                  reinterpret_cast<char*>(dst_alloc_record->ptr) +
                                  offset ? sizeof(Kokkos::Impl::SharedAllocationHeader) : 0;

      std::size_t dst_size = dst_alloc_record->size - dst_offset;

      UMPIRE_REPLAY(
          R"( "event": "copy", "payload": { "src": ")"
          << src_header
          << R"(", src_offset: ")"
          << src_offset
          << R"(", "dest": ")"
          << dst_header
          << R"(", dst_offset: ")"
          << dst_offset
          << R"(",  "size": )"
          << size
          << R"(, "src_allocator_ref": ")"
          << src_alloc_record->strategy
          << R"(", "dst_allocator_ref": ")"
          << dst_alloc_record->strategy
          << R"(" } )"
      );

      if (size > src_size) {
         UMPIRE_ERROR("Copy asks for more that resides in source copy: " << size << " -> " << src_size);
      }

      if (size > dst_size) {
         UMPIRE_ERROR("Not enough resource in destination for copy: " << size << " -> " << dst_size);
      }

      auto op = op_registry.find("COPY",
         src_alloc_record->strategy,
         dst_alloc_record->strategy);

      op->transform(const_cast<void*>(src), &dst, 
                    const_cast<umpire::util::AllocationRecord*>(src_alloc_record), 
                    const_cast<umpire::util::AllocationRecord*>(dst_alloc_record), 
                    size);
   }

   void host_to_umpire_deep_copy ( void * dst, const void * src, size_t size, bool offset) {
      auto &rm = umpire::ResourceManager::getInstance();
      auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();
      auto hostAllocator = rm.getAllocator("HOST");

      Kokkos::Impl::SharedAllocationHeader * dst_header = 
                     (Kokkos::Impl::SharedAllocationHeader*)dst;
      if (offset) dst_header -= 1;

      auto dst_alloc_record = rm.findAllocationRecord(dst_header);
      std::ptrdiff_t dst_offset = reinterpret_cast<char*>(dst_header) - 
                                  reinterpret_cast<char*>(dst_alloc_record->ptr) +
                                  offset ? sizeof(Kokkos::Impl::SharedAllocationHeader) : 0;

      std::size_t dst_size = dst_alloc_record->size - dst_offset;

      if (size > dst_size) {
         UMPIRE_ERROR("Copy asks for more that will fit in the destination: " << size << " -> " << dst_size);
      }

      umpire::util::AllocationRecord src_alloc_record{
        nullptr, size, hostAllocator.getAllocationStrategy()};

      auto op = op_registry.find("COPY",
         src_alloc_record.strategy,
         dst_alloc_record->strategy);

      op->transform(const_cast<void*>(src), &dst, 
                    const_cast<umpire::util::AllocationRecord*>(&src_alloc_record), 
                    const_cast<umpire::util::AllocationRecord*>(dst_alloc_record), 
                    size);
   }

   void umpire_to_host_deep_copy ( void * dst, const void * src, size_t size, bool offset) {
      auto &rm = umpire::ResourceManager::getInstance();
      auto& op_registry = umpire::op::MemoryOperationRegistry::getInstance();
      auto hostAllocator = rm.getAllocator("HOST");

      Kokkos::Impl::SharedAllocationHeader * src_header = 
                     (Kokkos::Impl::SharedAllocationHeader*)src;
      if (offset) src_header -= 1;

      auto src_alloc_record = rm.findAllocationRecord(src_header);
      std::ptrdiff_t src_offset = reinterpret_cast<char*>(src_header) - 
                                  reinterpret_cast<char*>(src_alloc_record->ptr) +
                                  offset ? sizeof(Kokkos::Impl::SharedAllocationHeader) : 0;

      std::size_t src_size = src_alloc_record->size - src_offset;

      if (size > src_size) {
         UMPIRE_ERROR("Copy asks for more that resides in source copy: " << size << " -> " << src_size);
      }

      umpire::util::AllocationRecord dst_alloc_record{
        nullptr, size, hostAllocator.getAllocationStrategy()};

      auto op = op_registry.find("COPY",
         src_alloc_record->strategy,
         dst_alloc_record.strategy);

      op->transform(const_cast<void*>(src), &dst, 
                    const_cast<umpire::util::AllocationRecord*>(src_alloc_record), 
                    const_cast<umpire::util::AllocationRecord*>(&dst_alloc_record), 
                    size);
   }
}  // Impl namespace

umpire::Allocator UmpireSpace::get_allocator(const char *name) {
  auto &rm = umpire::ResourceManager::getInstance();
  return rm.getAllocator(name);
}

/* Default allocation mechanism */
UmpireSpace::UmpireSpace() : m_AllocatorName("HOST") {}

void *UmpireSpace::allocate(const size_t arg_alloc_size) const {
  static_assert(sizeof(void *) == sizeof(uintptr_t),
                "Error sizeof(void*) != sizeof(uintptr_t)");

  static_assert(
      Kokkos::Impl::is_integral_power_of_two(Kokkos::Impl::MEMORY_ALIGNMENT),
      "Memory alignment must be power of two");

  constexpr uintptr_t alignment      = Kokkos::Impl::MEMORY_ALIGNMENT;

  void *ptr = nullptr;

  if (arg_alloc_size) {
    // Over-allocate to and round up to guarantee proper alignment.
    size_t size_padded = arg_alloc_size + sizeof(void *) + alignment;

    printf("obtaining umptire allocator: %s \n", m_AllocatorName);
    auto allocator  = get_allocator(m_AllocatorName);
    printf("Allocating umpire pointer: %s \n", m_AllocatorName);
    ptr = allocator.allocate(size_padded);
  }

  if (ptr == nullptr) {
    Experimental::RawMemoryAllocationFailure::FailureMode failure_mode =
         Experimental::RawMemoryAllocationFailure::FailureMode::OutOfMemoryError;

    throw Kokkos::Experimental::RawMemoryAllocationFailure(
        arg_alloc_size, alignment, failure_mode,
        Experimental::RawMemoryAllocationFailure::AllocationMechanism::
            StdMalloc);
  }

  printf("returning umpire pointer: %s \n", m_AllocatorName);
  return ptr;
}

void UmpireSpace::deallocate(void *const arg_alloc_ptr, const size_t) const {
  if (arg_alloc_ptr) {
    void *alloc_ptr = *(reinterpret_cast<void **>(arg_alloc_ptr) - 1);
    auto allocator  = get_allocator(m_AllocatorName);
    allocator.deallocate(alloc_ptr);
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

std::string SharedAllocationRecord<Kokkos::UmpireSpace, void>::get_label() const {
   if (m_space.is_host_accessible_space()) {
       return std::string(RecordBase::head()->m_label);
   } else {
      SharedAllocationHeader header;

    Kokkos::Impl::umpire_to_host_deep_copy(
                  &header, RecordBase::head(), sizeof(SharedAllocationHeader), false);

     return std::string(header.m_label);
   }
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

  if (m_space.is_host_accessible_space()) {
    // Fill in the Header information
    RecordBase::m_alloc_ptr->m_record =
        static_cast<SharedAllocationRecord<void, void> *>(this);

    strncpy(RecordBase::m_alloc_ptr->m_label, arg_label.c_str(),
            SharedAllocationHeader::maximum_label_length);
    // Set last element zero, in case c_str is too long
    RecordBase::m_alloc_ptr
      ->m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;
  } else {
    SharedAllocationHeader header;

    // Fill in the Header information
    header.m_record = static_cast<SharedAllocationRecord<void, void> *>(this);

    strncpy(header.m_label, arg_label.c_str(),
            SharedAllocationHeader::maximum_label_length);
    // Set last element zero, in case c_str is too long
    header.m_label[SharedAllocationHeader::maximum_label_length - 1] = (char)0;

    printf("copy shared alloc header to umpire space \n");

    // Copy to device memory
    Kokkos::Impl::host_to_umpire_deep_copy(RecordBase::m_alloc_ptr, &header,
                                               sizeof(SharedAllocationHeader), false);
  }
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
  using Header = SharedAllocationHeader;
  using RecordUmpire = SharedAllocationRecord<Kokkos::UmpireSpace, void>;

  // Copy the header from the allocation
  // cannot determine if it is host or device statically, so we will always deep
  // copy it....
  Header head;

  Header const *const head_dev =
      alloc_ptr ? Header::get_header(alloc_ptr) : (Header *)0;

  if (alloc_ptr) {
    Kokkos::Impl::umpire_to_host_deep_copy(
                      &head, head_dev, sizeof(SharedAllocationHeader), false);
  }

  RecordUmpire *const record =
    alloc_ptr ? static_cast<RecordUmpire *>(head.m_record) : (RecordUmpire *)0;

  if (!alloc_ptr || record->m_alloc_ptr != head_dev) {
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

}  // namespace Impl

}  // namespace Kokkos
