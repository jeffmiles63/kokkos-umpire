

namespace Test {

class TestUmpireAllocators {

   const int N = 100;
   using mem_space = Kokkos::UmpireSpace;
   using view_type = Kokkos::View<T*, mem_space>; 
   using default_device = Kokkos::DefaultExecutionSpace::memory_space;
   using default_host = Kokkos::DefaultHostExecutionSpace::memory_space;

   void run_tests( ) {

      // no allocator
      //
      mem_space no_alloc_host(mem_space::umpire_space_name(default_host));
      mem_space no_alloc_device(mem_space::umpire_space_name(default_device));

      view_type v1(no_alloc_device, "v1", N);
      view_type v2(no_alloc_host, "v2", N);

      auto h_v1 = Kokkos::create_host_mirror(v1);
      auto h_v2 = Kokkos::create_host_mirror(v2);

      for (int i = 0; i < N; i++) {
         h_v1 = i;
         h_v2 = i * 2;
      }

      Kokkos::deep_copy(v1, h_v1);
      Kokkos::deep_copy(v1, h_v1);

      // pooled allocator
      //

      // typed allocator 
      //

   }
};

TEST(TEST_CATEGORY, umpire_space_view_allocators) {
  TestUmpireAllocators f();
  f.run_tests();
}

}  // namespace Test
