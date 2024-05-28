! BSD 3-Clause No Military License
! Copyright © 2024, Piotr Bajdek. All Rights Reserved.

program knn_evolutionary
  implicit none
  character(len=256) :: arg_1
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(8), dimension(:), allocatable :: sequence
  integer :: seq_len, i, j, k, m, g, best_k
  real(8) :: y_pred, min_dist, fitness, best_fitness
  real(8), dimension(:,:), allocatable :: training_data
  real(8), dimension(:), allocatable :: distances
  integer, parameter :: pop_size = 1000, generations = 100
  integer, dimension(pop_size) :: population, new_population
  real(8), dimension(pop_size) :: fitness_values
  real(8) :: rand_real
  real(8), dimension(:), allocatable :: x_new

  if (command_argument_count() == 0) then
      write(*,'(a)') 'No input file selected!'
      call exit(0)
  end if

  call get_command_argument(1, arg_1)

  open(10, file=arg_1, status='old', action='read')
  seq_len = 0
  do
      read(10,*,iostat=i)
      if (i /= 0) exit
      seq_len = seq_len + 1
  end do
  allocate(sequence(seq_len))
  rewind(10)
  do i=1, seq_len
      read(10,*) sequence(i)
  end do
  close(10)

  allocate(distances(seq_len-1))

  call random_seed()
  call initialize_population(population, pop_size, seq_len-1)

  do g = 1, generations
      do i = 1, pop_size
          fitness_values(i) = evaluate_fitness(sequence, distances, seq_len, population(i))
      end do
      call select_population(population, fitness_values, new_population, pop_size)
      call mutate_population(new_population, pop_size, seq_len-1)
      population = new_population
  end do

  best_fitness = fitness_values(1)
  best_k = population(1)
  do i = 2, pop_size
      if (fitness_values(i) < best_fitness) then
          best_fitness = fitness_values(i)
          best_k = population(i)
      end if
  end do

  k = best_k
  allocate(x_new(k))
  x_new = sequence(seq_len-k+1:seq_len)
  
  allocate(training_data(seq_len-k, k+1))
  do i=1, seq_len-k
      training_data(i,1:k) = sequence(i:i+k-1)
      training_data(i,k+1) = sequence(i+k)
  end do

  y_pred = knn_predict(training_data, distances, x_new, k)
  write(*,'(a,I0,a,F0.6)') 'Optimal k = ', k, ' Predicted value = ', y_pred

  deallocate(sequence, distances, x_new, training_data)

contains

  subroutine initialize_population(population, size, max_k)
    integer, dimension(:), intent(out) :: population
    integer, intent(in) :: size, max_k
    integer :: i
    real(8) :: rand_real
    do i = 1, size
        call random_number(rand_real)
        population(i) = int(rand_real * max_k) + 1
    end do
  end subroutine initialize_population

  real(8) function evaluate_fitness(sequence, distances, seq_len, k)
    real(8), dimension(:), intent(in) :: sequence
    real(8), dimension(:), intent(inout) :: distances
    integer, intent(in) :: seq_len, k
    integer :: i, j
    real(8) :: y_true, y_pred, error
    real(8), dimension(:), allocatable :: x_new
    real(8), dimension(:,:), allocatable :: training_data

    allocate(training_data(seq_len-k, k+1))
    do i=1, seq_len-k
        training_data(i,1:k) = sequence(i:i+k-1)
        training_data(i,k+1) = sequence(i+k)
    end do

    allocate(x_new(k))
    x_new = sequence(seq_len-k+1:seq_len)
    y_true = sequence(seq_len)
    y_pred = knn_predict(training_data, distances, x_new, k)
    error = abs(y_true - y_pred)
    deallocate(x_new, training_data)
    evaluate_fitness = error
  end function evaluate_fitness

real(8) function knn_predict(training_data, distances, x_new, k)
    real(8), dimension(:,:), intent(in) :: training_data
    real(8), dimension(:), intent(inout) :: distances
    real(8), dimension(:), intent(in) :: x_new
    integer, intent(in) :: k
    integer :: i, j, m, min_idx
    real(8) :: y_pred, min_dist

    do i=1, size(training_data, 1)
        distances(i) = sqrt(sum((training_data(i,1:size(x_new)) - x_new)**2))
    end do

    y_pred = 0.0d0
    do m = 1, k
        min_dist = huge(1.0d0)
        do i = 1, size(training_data, 1)
            if (distances(i) < min_dist) then
                min_dist = distances(i)
                min_idx = i
            end if
        end do
        y_pred = y_pred + training_data(min_idx, size(x_new) + 1)
        distances(min_idx) = huge(1.0d0)
    end do

    knn_predict = y_pred / k
  end function knn_predict

  subroutine select_population(population, fitness_values, new_population, size)
    integer, dimension(:), intent(in) :: population
    real(8), dimension(:), intent(in) :: fitness_values
    integer, dimension(:), intent(out) :: new_population
    integer, intent(in) :: size
    integer :: i, idx
    real(8) :: rand_real

    do i = 1, size
        call random_number(rand_real)
        idx = int(rand_real * size) + 1
        new_population(i) = population(idx)
    end do
  end subroutine select_population

  subroutine mutate_population(population, size, max_k)
    integer, dimension(:), intent(inout) :: population
    integer, intent(in) :: size, max_k
    integer :: i
    real(8) :: rand_real

    do i = 1, size
        call random_number(rand_real)
        if (rand_real < 0.1d0) then
            call random_number(rand_real)
            population(i) = int(rand_real * max_k) + 1
        end if
    end do
  end subroutine mutate_population

end program knn_evolutionary
