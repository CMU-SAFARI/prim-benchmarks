; ModuleID = 'app_baseline.bc'
source_filename = "app_baseline.c"
target datalayout = "e-m:e-i8:8:32-i16:16:32-i64:64-i128:128-n32:64-S128"
target triple = "aarch64-unknown-linux-gnu"

%struct._IO_FILE = type { i32, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, i8*, %struct._IO_marker*, %struct._IO_FILE*, i32, i32, i64, i16, i8, [1 x i8], i8*, i64, %struct._IO_codecvt*, %struct._IO_wide_data*, %struct._IO_FILE*, i8*, i64, i32, [20 x i8] }
%struct._IO_marker = type opaque
%struct._IO_codecvt = type opaque
%struct._IO_wide_data = type opaque
%struct.Timer = type { [4 x %struct.timeval], [4 x %struct.timeval], [4 x double] }
%struct.timeval = type { i64, i64 }
%struct.Params = type { i32, i32, i32, i32 }

@.str = private unnamed_addr constant [15 x i8] c"Time (ms): %f\09\00", align 1
@.str.1 = private unnamed_addr constant [16 x i8] c"nr_elements\09%u\09\00", align 1
@A = internal global i32* null, align 8
@B = internal global i32* null, align 8
@C = internal global i32* null, align 8
@stderr = external dso_local global %struct._IO_FILE*, align 8
@.str.2 = private unnamed_addr constant [298 x i8] c"\0AUsage:  ./program [options]\0A\0AGeneral options:\0A    -h        help\0A    -t <T>    # of threads (default=8)\0A    -w <W>    # of untimed warmup iterations (default=1)\0A    -e <E>    # of timed repetition iterations (default=3)\0A\0ABenchmark-specific options:\0A    -i <I>    input size (default=8M elements)\0A\00", align 1
@.str.3 = private unnamed_addr constant [10 x i8] c"hi:w:e:t:\00", align 1
@optarg = external dso_local global i8*, align 8
@.str.4 = private unnamed_addr constant [23 x i8] c"\0AUnrecognized option!\0A\00", align 1
@.str.5 = private unnamed_addr constant [20 x i8] c"Invalid # of ranks!\00", align 1
@.str.6 = private unnamed_addr constant [41 x i8] c"p.n_threads > 0 && \22Invalid # of ranks!\22\00", align 1
@.str.7 = private unnamed_addr constant [15 x i8] c"app_baseline.c\00", align 1
@__PRETTY_FUNCTION__.input_params = private unnamed_addr constant [41 x i8] c"struct Params input_params(int, char **)\00", align 1
@.str.8 = private unnamed_addr constant [8 x i8] c"Kernel \00", align 1
@.str.9 = private unnamed_addr constant [2 x i8] c"\0A\00", align 1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @start(%struct.Timer* %0, i32 %1, i32 %2) #0 {
  call void @updateInstrInfo1(i32 31)
  %4 = alloca %struct.Timer*, align 8
  call void @updateInstrInfo1(i32 31)
  %5 = alloca i32, align 4
  call void @updateInstrInfo1(i32 31)
  %6 = alloca i32, align 4
  call void @updateInstrInfo1(i32 33)
  store %struct.Timer* %0, %struct.Timer** %4, align 8
  call void @updateInstrInfo1(i32 33)
  store i32 %1, i32* %5, align 4
  call void @updateInstrInfo1(i32 33)
  store i32 %2, i32* %6, align 4
  call void @updateInstrInfo1(i32 32)
  %7 = load i32, i32* %6, align 4
  call void @updateInstrInfo1(i32 53)
  %8 = icmp eq i32 %7, 0
  call void @updateInstrInfo1(i32 2)
  br i1 %8, label %9, label %15

9:                                                ; preds = %3
  call void @updateInstrInfo1(i32 32)
  %10 = load %struct.Timer*, %struct.Timer** %4, align 8
  call void @updateInstrInfo1(i32 34)
  %11 = getelementptr inbounds %struct.Timer, %struct.Timer* %10, i32 0, i32 2
  call void @updateInstrInfo1(i32 32)
  %12 = load i32, i32* %5, align 4
  call void @updateInstrInfo1(i32 40)
  %13 = sext i32 %12 to i64
  call void @updateInstrInfo1(i32 34)
  %14 = getelementptr inbounds [4 x double], [4 x double]* %11, i64 0, i64 %13
  call void @updateInstrInfo1(i32 33)
  store double 0.000000e+00, double* %14, align 8
  call void @updateInstrInfo1(i32 2)
  br label %15

15:                                               ; preds = %9, %3
  call void @updateInstrInfo1(i32 32)
  %16 = load %struct.Timer*, %struct.Timer** %4, align 8
  call void @updateInstrInfo1(i32 34)
  %17 = getelementptr inbounds %struct.Timer, %struct.Timer* %16, i32 0, i32 0
  call void @updateInstrInfo1(i32 32)
  %18 = load i32, i32* %5, align 4
  call void @updateInstrInfo1(i32 40)
  %19 = sext i32 %18 to i64
  call void @updateInstrInfo1(i32 34)
  %20 = getelementptr inbounds [4 x %struct.timeval], [4 x %struct.timeval]* %17, i64 0, i64 %19
  call void @updateInstrInfo1(i32 56)
  %21 = call i32 @gettimeofday(%struct.timeval* %20, i8* null) #5
  call void @updateInstrInfo1(i32 1)
  ret void
}

; Function Attrs: nounwind
declare dso_local i32 @gettimeofday(%struct.timeval*, i8*) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @stop(%struct.Timer* %0, i32 %1) #0 {
  call void @updateInstrInfo1(i32 31)
  %3 = alloca %struct.Timer*, align 8
  call void @updateInstrInfo1(i32 31)
  %4 = alloca i32, align 4
  call void @updateInstrInfo1(i32 33)
  store %struct.Timer* %0, %struct.Timer** %3, align 8
  call void @updateInstrInfo1(i32 33)
  store i32 %1, i32* %4, align 4
  call void @updateInstrInfo1(i32 32)
  %5 = load %struct.Timer*, %struct.Timer** %3, align 8
  call void @updateInstrInfo1(i32 34)
  %6 = getelementptr inbounds %struct.Timer, %struct.Timer* %5, i32 0, i32 1
  call void @updateInstrInfo1(i32 32)
  %7 = load i32, i32* %4, align 4
  call void @updateInstrInfo1(i32 40)
  %8 = sext i32 %7 to i64
  call void @updateInstrInfo1(i32 34)
  %9 = getelementptr inbounds [4 x %struct.timeval], [4 x %struct.timeval]* %6, i64 0, i64 %8
  call void @updateInstrInfo1(i32 56)
  %10 = call i32 @gettimeofday(%struct.timeval* %9, i8* null) #5
  call void @updateInstrInfo1(i32 32)
  %11 = load %struct.Timer*, %struct.Timer** %3, align 8
  call void @updateInstrInfo1(i32 34)
  %12 = getelementptr inbounds %struct.Timer, %struct.Timer* %11, i32 0, i32 1
  call void @updateInstrInfo1(i32 32)
  %13 = load i32, i32* %4, align 4
  call void @updateInstrInfo1(i32 40)
  %14 = sext i32 %13 to i64
  call void @updateInstrInfo1(i32 34)
  %15 = getelementptr inbounds [4 x %struct.timeval], [4 x %struct.timeval]* %12, i64 0, i64 %14
  call void @updateInstrInfo1(i32 34)
  %16 = getelementptr inbounds %struct.timeval, %struct.timeval* %15, i32 0, i32 0
  call void @updateInstrInfo1(i32 32)
  %17 = load i64, i64* %16, align 8
  call void @updateInstrInfo1(i32 32)
  %18 = load %struct.Timer*, %struct.Timer** %3, align 8
  call void @updateInstrInfo1(i32 34)
  %19 = getelementptr inbounds %struct.Timer, %struct.Timer* %18, i32 0, i32 0
  call void @updateInstrInfo1(i32 32)
  %20 = load i32, i32* %4, align 4
  call void @updateInstrInfo1(i32 40)
  %21 = sext i32 %20 to i64
  call void @updateInstrInfo1(i32 34)
  %22 = getelementptr inbounds [4 x %struct.timeval], [4 x %struct.timeval]* %19, i64 0, i64 %21
  call void @updateInstrInfo1(i32 34)
  %23 = getelementptr inbounds %struct.timeval, %struct.timeval* %22, i32 0, i32 0
  call void @updateInstrInfo1(i32 32)
  %24 = load i64, i64* %23, align 8
  call void @updateInstrInfo1(i32 15)
  %25 = sub nsw i64 %17, %24
  call void @updateInstrInfo1(i32 44)
  %26 = sitofp i64 %25 to double
  call void @updateInstrInfo1(i32 18)
  %27 = fmul double %26, 1.000000e+06
  call void @updateInstrInfo1(i32 32)
  %28 = load %struct.Timer*, %struct.Timer** %3, align 8
  call void @updateInstrInfo1(i32 34)
  %29 = getelementptr inbounds %struct.Timer, %struct.Timer* %28, i32 0, i32 1
  call void @updateInstrInfo1(i32 32)
  %30 = load i32, i32* %4, align 4
  call void @updateInstrInfo1(i32 40)
  %31 = sext i32 %30 to i64
  call void @updateInstrInfo1(i32 34)
  %32 = getelementptr inbounds [4 x %struct.timeval], [4 x %struct.timeval]* %29, i64 0, i64 %31
  call void @updateInstrInfo1(i32 34)
  %33 = getelementptr inbounds %struct.timeval, %struct.timeval* %32, i32 0, i32 1
  call void @updateInstrInfo1(i32 32)
  %34 = load i64, i64* %33, align 8
  call void @updateInstrInfo1(i32 32)
  %35 = load %struct.Timer*, %struct.Timer** %3, align 8
  call void @updateInstrInfo1(i32 34)
  %36 = getelementptr inbounds %struct.Timer, %struct.Timer* %35, i32 0, i32 0
  call void @updateInstrInfo1(i32 32)
  %37 = load i32, i32* %4, align 4
  call void @updateInstrInfo1(i32 40)
  %38 = sext i32 %37 to i64
  call void @updateInstrInfo1(i32 34)
  %39 = getelementptr inbounds [4 x %struct.timeval], [4 x %struct.timeval]* %36, i64 0, i64 %38
  call void @updateInstrInfo1(i32 34)
  %40 = getelementptr inbounds %struct.timeval, %struct.timeval* %39, i32 0, i32 1
  call void @updateInstrInfo1(i32 32)
  %41 = load i64, i64* %40, align 8
  call void @updateInstrInfo1(i32 15)
  %42 = sub nsw i64 %34, %41
  call void @updateInstrInfo1(i32 44)
  %43 = sitofp i64 %42 to double
  call void @updateInstrInfo1(i32 14)
  %44 = fadd double %27, %43
  call void @updateInstrInfo1(i32 32)
  %45 = load %struct.Timer*, %struct.Timer** %3, align 8
  call void @updateInstrInfo1(i32 34)
  %46 = getelementptr inbounds %struct.Timer, %struct.Timer* %45, i32 0, i32 2
  call void @updateInstrInfo1(i32 32)
  %47 = load i32, i32* %4, align 4
  call void @updateInstrInfo1(i32 40)
  %48 = sext i32 %47 to i64
  call void @updateInstrInfo1(i32 34)
  %49 = getelementptr inbounds [4 x double], [4 x double]* %46, i64 0, i64 %48
  call void @updateInstrInfo1(i32 32)
  %50 = load double, double* %49, align 8
  call void @updateInstrInfo1(i32 14)
  %51 = fadd double %50, %44
  call void @updateInstrInfo1(i32 33)
  store double %51, double* %49, align 8
  call void @updateInstrInfo1(i32 1)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @print(%struct.Timer* %0, i32 %1, i32 %2) #0 {
  call void @updateInstrInfo1(i32 31)
  %4 = alloca %struct.Timer*, align 8
  call void @updateInstrInfo1(i32 31)
  %5 = alloca i32, align 4
  call void @updateInstrInfo1(i32 31)
  %6 = alloca i32, align 4
  call void @updateInstrInfo1(i32 33)
  store %struct.Timer* %0, %struct.Timer** %4, align 8
  call void @updateInstrInfo1(i32 33)
  store i32 %1, i32* %5, align 4
  call void @updateInstrInfo1(i32 33)
  store i32 %2, i32* %6, align 4
  call void @updateInstrInfo1(i32 32)
  %7 = load %struct.Timer*, %struct.Timer** %4, align 8
  call void @updateInstrInfo1(i32 34)
  %8 = getelementptr inbounds %struct.Timer, %struct.Timer* %7, i32 0, i32 2
  call void @updateInstrInfo1(i32 32)
  %9 = load i32, i32* %5, align 4
  call void @updateInstrInfo1(i32 40)
  %10 = sext i32 %9 to i64
  call void @updateInstrInfo1(i32 34)
  %11 = getelementptr inbounds [4 x double], [4 x double]* %8, i64 0, i64 %10
  call void @updateInstrInfo1(i32 32)
  %12 = load double, double* %11, align 8
  call void @updateInstrInfo1(i32 32)
  %13 = load i32, i32* %6, align 4
  call void @updateInstrInfo1(i32 17)
  %14 = mul nsw i32 1000, %13
  call void @updateInstrInfo1(i32 44)
  %15 = sitofp i32 %14 to double
  call void @updateInstrInfo1(i32 21)
  %16 = fdiv double %12, %15
  call void @updateInstrInfo1(i32 56)
  %17 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str, i64 0, i64 0), double %16)
  call void @updateInstrInfo1(i32 1)
  ret void
}

declare dso_local i32 @printf(i8*, ...) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @create_test_file(i32 %0) #0 {
  call void @updateInstrInfo1(i32 31)
  %2 = alloca i32, align 4
  call void @updateInstrInfo1(i32 31)
  %3 = alloca i32, align 4
  call void @updateInstrInfo1(i32 33)
  store i32 %0, i32* %2, align 4
  call void @updateInstrInfo1(i32 56)
  call void @srand(i32 0) #5
  call void @updateInstrInfo1(i32 32)
  %4 = load i32, i32* %2, align 4
  call void @updateInstrInfo1(i32 56)
  %5 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str.1, i64 0, i64 0), i32 %4)
  call void @updateInstrInfo1(i32 32)
  %6 = load i32, i32* %2, align 4
  call void @updateInstrInfo1(i32 39)
  %7 = zext i32 %6 to i64
  call void @updateInstrInfo1(i32 17)
  %8 = mul i64 %7, 4
  call void @updateInstrInfo1(i32 56)
  %9 = call noalias align 16 i8* @malloc(i64 %8) #5
  call void @updateInstrInfo1(i32 49)
  %10 = bitcast i8* %9 to i32*
  call void @updateInstrInfo1(i32 33)
  store i32* %10, i32** @A, align 8
  call void @updateInstrInfo1(i32 32)
  %11 = load i32, i32* %2, align 4
  call void @updateInstrInfo1(i32 39)
  %12 = zext i32 %11 to i64
  call void @updateInstrInfo1(i32 17)
  %13 = mul i64 %12, 4
  call void @updateInstrInfo1(i32 56)
  %14 = call noalias align 16 i8* @malloc(i64 %13) #5
  call void @updateInstrInfo1(i32 49)
  %15 = bitcast i8* %14 to i32*
  call void @updateInstrInfo1(i32 33)
  store i32* %15, i32** @B, align 8
  call void @updateInstrInfo1(i32 32)
  %16 = load i32, i32* %2, align 4
  call void @updateInstrInfo1(i32 39)
  %17 = zext i32 %16 to i64
  call void @updateInstrInfo1(i32 17)
  %18 = mul i64 %17, 4
  call void @updateInstrInfo1(i32 56)
  %19 = call noalias align 16 i8* @malloc(i64 %18) #5
  call void @updateInstrInfo1(i32 49)
  %20 = bitcast i8* %19 to i32*
  call void @updateInstrInfo1(i32 33)
  store i32* %20, i32** @C, align 8
  call void @updateInstrInfo1(i32 33)
  store i32 0, i32* %3, align 4
  call void @updateInstrInfo1(i32 2)
  br label %21

21:                                               ; preds = %36, %1
  call void @updateInstrInfo1(i32 32)
  %22 = load i32, i32* %3, align 4
  call void @updateInstrInfo1(i32 32)
  %23 = load i32, i32* %2, align 4
  call void @updateInstrInfo1(i32 53)
  %24 = icmp ult i32 %22, %23
  call void @updateInstrInfo1(i32 2)
  br i1 %24, label %25, label %39

25:                                               ; preds = %21
  call void @updateInstrInfo1(i32 56)
  %26 = call i32 @rand() #5
  call void @updateInstrInfo1(i32 32)
  %27 = load i32*, i32** @A, align 8
  call void @updateInstrInfo1(i32 32)
  %28 = load i32, i32* %3, align 4
  call void @updateInstrInfo1(i32 40)
  %29 = sext i32 %28 to i64
  call void @updateInstrInfo1(i32 34)
  %30 = getelementptr inbounds i32, i32* %27, i64 %29
  call void @updateInstrInfo1(i32 33)
  store i32 %26, i32* %30, align 4
  call void @updateInstrInfo1(i32 56)
  %31 = call i32 @rand() #5
  call void @updateInstrInfo1(i32 32)
  %32 = load i32*, i32** @B, align 8
  call void @updateInstrInfo1(i32 32)
  %33 = load i32, i32* %3, align 4
  call void @updateInstrInfo1(i32 40)
  %34 = sext i32 %33 to i64
  call void @updateInstrInfo1(i32 34)
  %35 = getelementptr inbounds i32, i32* %32, i64 %34
  call void @updateInstrInfo1(i32 33)
  store i32 %31, i32* %35, align 4
  call void @updateInstrInfo1(i32 2)
  br label %36

36:                                               ; preds = %25
  call void @updateInstrInfo1(i32 32)
  %37 = load i32, i32* %3, align 4
  call void @updateInstrInfo1(i32 13)
  %38 = add nsw i32 %37, 1
  call void @updateInstrInfo1(i32 33)
  store i32 %38, i32* %3, align 4
  call void @updateInstrInfo1(i32 2)
  br label %21, !llvm.loop !9

39:                                               ; preds = %21
  call void @updateInstrInfo1(i32 1)
  ret void
}

; Function Attrs: nounwind
declare dso_local void @srand(i32) #1

; Function Attrs: nounwind
declare dso_local noalias align 16 i8* @malloc(i64) #1

; Function Attrs: nounwind
declare dso_local i32 @rand() #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @usage() #0 {
  call void @updateInstrInfo1(i32 32)
  %1 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  call void @updateInstrInfo1(i32 56)
  %2 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %1, i8* getelementptr inbounds ([298 x i8], [298 x i8]* @.str.2, i64 0, i64 0))
  call void @updateInstrInfo1(i32 1)
  ret void
}

declare dso_local i32 @fprintf(%struct._IO_FILE*, i8*, ...) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local [2 x i64] @input_params(i32 %0, i8** %1) #0 {
  call void @updateInstrInfo1(i32 31)
  %3 = alloca %struct.Params, align 4
  call void @updateInstrInfo1(i32 31)
  %4 = alloca i32, align 4
  call void @updateInstrInfo1(i32 31)
  %5 = alloca i8**, align 8
  call void @updateInstrInfo1(i32 31)
  %6 = alloca i32, align 4
  call void @updateInstrInfo1(i32 33)
  store i32 %0, i32* %4, align 4
  call void @updateInstrInfo1(i32 33)
  store i8** %1, i8*** %5, align 8
  call void @updateInstrInfo1(i32 34)
  %7 = getelementptr inbounds %struct.Params, %struct.Params* %3, i32 0, i32 0
  call void @updateInstrInfo1(i32 33)
  store i32 16777216, i32* %7, align 4
  call void @updateInstrInfo1(i32 34)
  %8 = getelementptr inbounds %struct.Params, %struct.Params* %3, i32 0, i32 1
  call void @updateInstrInfo1(i32 33)
  store i32 1, i32* %8, align 4
  call void @updateInstrInfo1(i32 34)
  %9 = getelementptr inbounds %struct.Params, %struct.Params* %3, i32 0, i32 2
  call void @updateInstrInfo1(i32 33)
  store i32 3, i32* %9, align 4
  call void @updateInstrInfo1(i32 34)
  %10 = getelementptr inbounds %struct.Params, %struct.Params* %3, i32 0, i32 3
  call void @updateInstrInfo1(i32 33)
  store i32 5, i32* %10, align 4
  call void @updateInstrInfo1(i32 2)
  br label %11

11:                                               ; preds = %38, %2
  call void @updateInstrInfo1(i32 32)
  %12 = load i32, i32* %4, align 4
  call void @updateInstrInfo1(i32 32)
  %13 = load i8**, i8*** %5, align 8
  call void @updateInstrInfo1(i32 56)
  %14 = call i32 @getopt(i32 %12, i8** %13, i8* getelementptr inbounds ([10 x i8], [10 x i8]* @.str.3, i64 0, i64 0)) #5
  call void @updateInstrInfo1(i32 33)
  store i32 %14, i32* %6, align 4
  call void @updateInstrInfo1(i32 53)
  %15 = icmp sge i32 %14, 0
  call void @updateInstrInfo1(i32 2)
  br i1 %15, label %16, label %39

16:                                               ; preds = %11
  call void @updateInstrInfo1(i32 32)
  %17 = load i32, i32* %6, align 4
  call void @updateInstrInfo1(i32 3)
  switch i32 %17, label %35 [
    i32 104, label %18
    i32 105, label %19
    i32 119, label %23
    i32 101, label %27
    i32 116, label %31
  ]

18:                                               ; preds = %16
  call void @updateInstrInfo1(i32 56)
  call void @usage()
  call void @updateInstrInfo1(i32 56)
  call void @exit(i32 0) #6
  call void @updateInstrInfo1(i32 7)
  unreachable

19:                                               ; preds = %16
  call void @updateInstrInfo1(i32 32)
  %20 = load i8*, i8** @optarg, align 8
  call void @updateInstrInfo1(i32 56)
  %21 = call i32 @atoi(i8* %20) #7
  call void @updateInstrInfo1(i32 34)
  %22 = getelementptr inbounds %struct.Params, %struct.Params* %3, i32 0, i32 0
  call void @updateInstrInfo1(i32 33)
  store i32 %21, i32* %22, align 4
  call void @updateInstrInfo1(i32 2)
  br label %38

23:                                               ; preds = %16
  call void @updateInstrInfo1(i32 32)
  %24 = load i8*, i8** @optarg, align 8
  call void @updateInstrInfo1(i32 56)
  %25 = call i32 @atoi(i8* %24) #7
  call void @updateInstrInfo1(i32 34)
  %26 = getelementptr inbounds %struct.Params, %struct.Params* %3, i32 0, i32 1
  call void @updateInstrInfo1(i32 33)
  store i32 %25, i32* %26, align 4
  call void @updateInstrInfo1(i32 2)
  br label %38

27:                                               ; preds = %16
  call void @updateInstrInfo1(i32 32)
  %28 = load i8*, i8** @optarg, align 8
  call void @updateInstrInfo1(i32 56)
  %29 = call i32 @atoi(i8* %28) #7
  call void @updateInstrInfo1(i32 34)
  %30 = getelementptr inbounds %struct.Params, %struct.Params* %3, i32 0, i32 2
  call void @updateInstrInfo1(i32 33)
  store i32 %29, i32* %30, align 4
  call void @updateInstrInfo1(i32 2)
  br label %38

31:                                               ; preds = %16
  call void @updateInstrInfo1(i32 32)
  %32 = load i8*, i8** @optarg, align 8
  call void @updateInstrInfo1(i32 56)
  %33 = call i32 @atoi(i8* %32) #7
  call void @updateInstrInfo1(i32 34)
  %34 = getelementptr inbounds %struct.Params, %struct.Params* %3, i32 0, i32 3
  call void @updateInstrInfo1(i32 33)
  store i32 %33, i32* %34, align 4
  call void @updateInstrInfo1(i32 2)
  br label %38

35:                                               ; preds = %16
  call void @updateInstrInfo1(i32 32)
  %36 = load %struct._IO_FILE*, %struct._IO_FILE** @stderr, align 8
  call void @updateInstrInfo1(i32 56)
  %37 = call i32 (%struct._IO_FILE*, i8*, ...) @fprintf(%struct._IO_FILE* %36, i8* getelementptr inbounds ([23 x i8], [23 x i8]* @.str.4, i64 0, i64 0))
  call void @updateInstrInfo1(i32 56)
  call void @usage()
  call void @updateInstrInfo1(i32 56)
  call void @exit(i32 0) #6
  call void @updateInstrInfo1(i32 7)
  unreachable

38:                                               ; preds = %31, %27, %23, %19
  call void @updateInstrInfo1(i32 2)
  br label %11, !llvm.loop !11

39:                                               ; preds = %11
  call void @updateInstrInfo1(i32 34)
  %40 = getelementptr inbounds %struct.Params, %struct.Params* %3, i32 0, i32 3
  call void @updateInstrInfo1(i32 32)
  %41 = load i32, i32* %40, align 4
  call void @updateInstrInfo1(i32 53)
  %42 = icmp sgt i32 %41, 0
  call void @updateInstrInfo1(i32 2)
  br i1 %42, label %43, label %45

43:                                               ; preds = %39
  call void @updateInstrInfo1(i32 2)
  br i1 true, label %44, label %45

44:                                               ; preds = %43
  call void @updateInstrInfo1(i32 2)
  br label %46

45:                                               ; preds = %43, %39
  call void @updateInstrInfo1(i32 56)
  call void @__assert_fail(i8* getelementptr inbounds ([41 x i8], [41 x i8]* @.str.6, i64 0, i64 0), i8* getelementptr inbounds ([15 x i8], [15 x i8]* @.str.7, i64 0, i64 0), i32 110, i8* getelementptr inbounds ([41 x i8], [41 x i8]* @__PRETTY_FUNCTION__.input_params, i64 0, i64 0)) #6
  call void @updateInstrInfo1(i32 7)
  unreachable

46:                                               ; preds = %44
  call void @updateInstrInfo1(i32 49)
  %47 = bitcast %struct.Params* %3 to [2 x i64]*
  call void @updateInstrInfo1(i32 32)
  %48 = load [2 x i64], [2 x i64]* %47, align 4
  call void @updateInstrInfo1(i32 1)
  ret [2 x i64] %48
}

; Function Attrs: nounwind
declare dso_local i32 @getopt(i32, i8**, i8*) #1

; Function Attrs: noreturn nounwind
declare dso_local void @exit(i32) #3

; Function Attrs: nounwind readonly willreturn
declare dso_local i32 @atoi(i8*) #4

; Function Attrs: noreturn nounwind
declare dso_local void @__assert_fail(i8*, i8*, i32, i8*) #3

; Function Attrs: noinline nounwind optnone uwtable
define dso_local i32 @main(i32 %0, i8** %1) #0 {
  call void @updateInstrInfo1(i32 31)
  %3 = alloca i32, align 4
  call void @updateInstrInfo1(i32 31)
  %4 = alloca i32, align 4
  call void @updateInstrInfo1(i32 31)
  %5 = alloca i8**, align 8
  call void @updateInstrInfo1(i32 31)
  %6 = alloca %struct.Params, align 4
  call void @updateInstrInfo1(i32 31)
  %7 = alloca i32, align 4
  call void @updateInstrInfo1(i32 31)
  %8 = alloca %struct.Timer, align 8
  call void @updateInstrInfo1(i32 33)
  store i32 0, i32* %3, align 4
  call void @updateInstrInfo1(i32 33)
  store i32 %0, i32* %4, align 4
  call void @updateInstrInfo1(i32 33)
  store i8** %1, i8*** %5, align 8
  call void @updateInstrInfo1(i32 32)
  %9 = load i32, i32* %4, align 4
  call void @updateInstrInfo1(i32 32)
  %10 = load i8**, i8*** %5, align 8
  call void @updateInstrInfo1(i32 56)
  %11 = call [2 x i64] @input_params(i32 %9, i8** %10)
  call void @updateInstrInfo1(i32 49)
  %12 = bitcast %struct.Params* %6 to [2 x i64]*
  call void @updateInstrInfo1(i32 33)
  store [2 x i64] %11, [2 x i64]* %12, align 4
  call void @updateInstrInfo1(i32 34)
  %13 = getelementptr inbounds %struct.Params, %struct.Params* %6, i32 0, i32 0
  call void @updateInstrInfo1(i32 32)
  %14 = load i32, i32* %13, align 4
  call void @updateInstrInfo1(i32 33)
  store i32 %14, i32* %7, align 4
  call void @updateInstrInfo1(i32 32)
  %15 = load i32, i32* %7, align 4
  call void @updateInstrInfo1(i32 56)
  call void @create_test_file(i32 %15)
  call void @updateInstrInfo1(i32 56)
  call void @start(%struct.Timer* %8, i32 0, i32 0)
  call void @updateInstrInfo1(i32 32)
  %16 = load i32, i32* %7, align 4
  call void @updateInstrInfo1(i32 34)
  %17 = getelementptr inbounds %struct.Params, %struct.Params* %6, i32 0, i32 3
  call void @updateInstrInfo1(i32 32)
  %18 = load i32, i32* %17, align 4
  call void @updateInstrInfo1(i32 56)
  call void @vector_addition_host(i32 %16, i32 %18)
  call void @updateInstrInfo1(i32 56)
  call void @stop(%struct.Timer* %8, i32 0)
  call void @updateInstrInfo1(i32 56)
  %19 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([8 x i8], [8 x i8]* @.str.8, i64 0, i64 0))
  call void @updateInstrInfo1(i32 56)
  call void @print(%struct.Timer* %8, i32 0, i32 1)
  call void @updateInstrInfo1(i32 56)
  %20 = call i32 (i8*, ...) @printf(i8* getelementptr inbounds ([2 x i8], [2 x i8]* @.str.9, i64 0, i64 0))
  call void @updateInstrInfo1(i32 32)
  %21 = load i32*, i32** @A, align 8
  call void @updateInstrInfo1(i32 49)
  %22 = bitcast i32* %21 to i8*
  call void @updateInstrInfo1(i32 56)
  call void @free(i8* %22) #5
  call void @updateInstrInfo1(i32 32)
  %23 = load i32*, i32** @B, align 8
  call void @updateInstrInfo1(i32 49)
  %24 = bitcast i32* %23 to i8*
  call void @updateInstrInfo1(i32 56)
  call void @free(i8* %24) #5
  call void @updateInstrInfo1(i32 32)
  %25 = load i32*, i32** @C, align 8
  call void @updateInstrInfo1(i32 49)
  %26 = bitcast i32* %25 to i8*
  call void @updateInstrInfo1(i32 56)
  call void @free(i8* %26) #5
  call void @updateInstrInfo1(i32 1)
  call void @printOutInstrInfo1()
  ret i32 0
}

; Function Attrs: noinline nounwind optnone uwtable
define internal void @vector_addition_host(i32 %0, i32 %1) #0 {
  call void @updateInstrInfo1(i32 31)
  %3 = alloca i32, align 4
  call void @updateInstrInfo1(i32 31)
  %4 = alloca i32, align 4
  call void @updateInstrInfo1(i32 31)
  %5 = alloca i32, align 4
  call void @updateInstrInfo1(i32 33)
  store i32 %0, i32* %3, align 4
  call void @updateInstrInfo1(i32 33)
  store i32 %1, i32* %4, align 4
  call void @updateInstrInfo1(i32 56)
  call void bitcast (void (...)* @start_region to void ()*)()
  call void @updateInstrInfo1(i32 33)
  store i32 0, i32* %5, align 4
  call void @updateInstrInfo1(i32 2)
  br label %6

6:                                                ; preds = %26, %2
  call void @updateInstrInfo1(i32 32)
  %7 = load i32, i32* %5, align 4
  call void @updateInstrInfo1(i32 32)
  %8 = load i32, i32* %3, align 4
  call void @updateInstrInfo1(i32 53)
  %9 = icmp ult i32 %7, %8
  call void @updateInstrInfo1(i32 2)
  br i1 %9, label %10, label %29

10:                                               ; preds = %6
  call void @updateInstrInfo1(i32 32)
  %11 = load i32*, i32** @A, align 8
  call void @updateInstrInfo1(i32 32)
  %12 = load i32, i32* %5, align 4
  call void @updateInstrInfo1(i32 40)
  %13 = sext i32 %12 to i64
  call void @updateInstrInfo1(i32 34)
  %14 = getelementptr inbounds i32, i32* %11, i64 %13
  call void @updateInstrInfo1(i32 32)
  %15 = load i32, i32* %14, align 4
  call void @updateInstrInfo1(i32 32)
  %16 = load i32*, i32** @B, align 8
  call void @updateInstrInfo1(i32 32)
  %17 = load i32, i32* %5, align 4
  call void @updateInstrInfo1(i32 40)
  %18 = sext i32 %17 to i64
  call void @updateInstrInfo1(i32 34)
  %19 = getelementptr inbounds i32, i32* %16, i64 %18
  call void @updateInstrInfo1(i32 32)
  %20 = load i32, i32* %19, align 4
  call void @updateInstrInfo1(i32 13)
  %21 = add nsw i32 %15, %20
  call void @updateInstrInfo1(i32 32)
  %22 = load i32*, i32** @C, align 8
  call void @updateInstrInfo1(i32 32)
  %23 = load i32, i32* %5, align 4
  call void @updateInstrInfo1(i32 40)
  %24 = sext i32 %23 to i64
  call void @updateInstrInfo1(i32 34)
  %25 = getelementptr inbounds i32, i32* %22, i64 %24
  call void @updateInstrInfo1(i32 33)
  store i32 %21, i32* %25, align 4
  call void @updateInstrInfo1(i32 2)
  br label %26

26:                                               ; preds = %10
  call void @updateInstrInfo1(i32 32)
  %27 = load i32, i32* %5, align 4
  call void @updateInstrInfo1(i32 13)
  %28 = add nsw i32 %27, 1
  call void @updateInstrInfo1(i32 33)
  store i32 %28, i32* %5, align 4
  call void @updateInstrInfo1(i32 2)
  br label %6, !llvm.loop !12

29:                                               ; preds = %6
  call void @updateInstrInfo1(i32 56)
  call void bitcast (void (...)* @end_region to void ()*)()
  call void @updateInstrInfo1(i32 1)
  ret void
}

; Function Attrs: nounwind
declare dso_local void @free(i8*) #1

declare dso_local void @start_region(...) #2

declare dso_local void @end_region(...) #2

declare void @updateInstrInfo1(i32)

declare void @printOutInstrInfo1()

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="non-leaf" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" }
attributes #1 = { nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" }
attributes #2 = { "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" }
attributes #3 = { noreturn nounwind "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" }
attributes #4 = { nounwind readonly willreturn "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="generic" "target-features"="+neon" }
attributes #5 = { nounwind }
attributes #6 = { noreturn nounwind }
attributes #7 = { nounwind readonly willreturn }

!llvm.module.flags = !{!0, !1, !2, !3, !4, !5, !6, !7}
!llvm.ident = !{!8}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, !"branch-target-enforcement", i32 0}
!2 = !{i32 1, !"sign-return-address", i32 0}
!3 = !{i32 1, !"sign-return-address-all", i32 0}
!4 = !{i32 1, !"sign-return-address-with-bkey", i32 0}
!5 = !{i32 7, !"openmp", i32 50}
!6 = !{i32 7, !"uwtable", i32 1}
!7 = !{i32 7, !"frame-pointer", i32 1}
!8 = !{!"Ubuntu clang version 13.0.1-++20220120110924+75e33f71c2da-1~exp1~20220120231001.58"}
!9 = distinct !{!9, !10}
!10 = !{!"llvm.loop.mustprogress"}
!11 = distinct !{!11, !10}
!12 = distinct !{!12, !10}
