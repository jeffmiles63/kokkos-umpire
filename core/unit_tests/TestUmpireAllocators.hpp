

namespace Test {

template<class T>
struct TestUmpireAllocators {

   const int N = 100;
   using mem_space = Kokkos::UmpireSpace;
   using view_type = Kokkos::View<T*, mem_space>; 
   using default_device = Kokkos::DefaultExecutionSpace::memory_space;
   using default_host = Kokkos::DefaultHostExecutionSpace::memory_space;
   using view_ctor_prop = Kokkos::Impl::ViewCtorProp<std::string, mem_space>;

   void run_tests( ) {

      // no allocator
      //
      printf("creating host allocator\n");
      mem_space no_alloc_host(mem_space::umpire_space_name(default_host()));
      printf("creating device allocator\n");
      mem_space no_alloc_device(mem_space::umpire_space_name(default_device()));

      printf("manually set device pointer \n"); 
      double * ptr = (double*)no_alloc_device.allocate(N*sizeof(double));
      Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,N), KOKKOS_LAMBDA (const int i) {
         ptr[i] = (double)i;
      });

      printf("creating device view\n");
      view_type v1(view_ctor_prop("v1", no_alloc_device), N);
      printf("creating host view\n");
      view_type v2(view_ctor_prop("v2", no_alloc_host), N);

      auto h_v1 = Kokkos::create_mirror(Kokkos::HostSpace(), v1);
      auto h_v2 = Kokkos::create_mirror(Kokkos::HostSpace(), v2);

      printf("initializing host data \n");
      for (int i = 0; i < N; i++) {
         h_v1(i) = i;
         h_v2(i) = i * 2;
      }

      printf("copying host data to umpire space \n");
      Kokkos::deep_copy(v1, h_v1);
      Kokkos::deep_copy(v2, h_v2);

      printf("device kernel \n");
      Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(0,N), KOKKOS_LAMBDA (const int i) {
         v1(i) *= 2;
      });
      Kokkos::fence();
      fflush(stdout);
      printf("host kernel \n");
      Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0,N), KOKKOS_LAMBDA (const int i) {
         v2(i) *= 2;
      });

      printf("copy data from umpire back to host \n"); 
      Kokkos::deep_copy(h_v1, v1);
      Kokkos::deep_copy(h_v2, v2);

      for (int i = 0; i < N; i++) {
         ASSERT_EQ( h_v1(i), 2*i );
         ASSERT_EQ( h_v2(i), i*4 );
      }
      // pooled allocator
      //

      // typed allocator 
      //

   }
};

TEST(TEST_CATEGORY, umpire_space_view_allocators) {
  TestUmpireAllocators<double> f{};
  f.run_tests();
}

}  // namespace Test
