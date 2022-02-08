; ModuleID = 'app_baseline_streamlined.c'
source_filename = "app_baseline_streamlined.c"
target datalayout = "e-m:e-p:32:32-i1:8:32-i8:8:32-i16:16:32-i32:32:32-i64:64:64-n32"
target triple = "dpu-upmem-dpurte"

@.str = private unnamed_addr constant [16 x i8] c"nr_elements\09%u\09\00", align 1
@A = internal global i32* null, align 4, !dbg !0
@B = internal global i32* null, align 4, !dbg !11
@C = internal global i32* null, align 4, !dbg !13

; Function Attrs: noinline nounwind optnone
define dso_local void @create_test_file(i32 %0) #0 !dbg !19 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  call void @llvm.dbg.declare(metadata i32* %2, metadata !23, metadata !DIExpression()), !dbg !24
  %4 = call i32 bitcast (i32 (...)* @srand to i32 (i32)*)(i32 0), !dbg !25
  %5 = load i32, i32* %2, align 4, !dbg !26
  call void (i8*, ...) @printf(i8* getelementptr inbounds ([16 x i8], [16 x i8]* @.str, i32 0, i32 0), i32 %5), !dbg !27
  %6 = load i32, i32* %2, align 4, !dbg !28
  %7 = mul i32 %6, 4, !dbg !29
  %8 = call i8* @malloc(i32 %7), !dbg !30
  %9 = bitcast i8* %8 to i32*, !dbg !31
  store i32* %9, i32** @A, align 4, !dbg !32
  %10 = load i32, i32* %2, align 4, !dbg !33
  %11 = mul i32 %10, 4, !dbg !34
  %12 = call i8* @malloc(i32 %11), !dbg !35
  %13 = bitcast i8* %12 to i32*, !dbg !36
  store i32* %13, i32** @B, align 4, !dbg !37
  %14 = load i32, i32* %2, align 4, !dbg !38
  %15 = mul i32 %14, 4, !dbg !39
  %16 = call i8* @malloc(i32 %15), !dbg !40
  %17 = bitcast i8* %16 to i32*, !dbg !41
  store i32* %17, i32** @C, align 4, !dbg !42
  call void @llvm.dbg.declare(metadata i32* %3, metadata !43, metadata !DIExpression()), !dbg !45
  store i32 0, i32* %3, align 4, !dbg !45
  br label %18, !dbg !46

18:                                               ; preds = %31, %1
  %19 = load i32, i32* %3, align 4, !dbg !47
  %20 = load i32, i32* %2, align 4, !dbg !49
  %21 = icmp ult i32 %19, %20, !dbg !50
  br i1 %21, label %22, label %34, !dbg !51

22:                                               ; preds = %18
  %23 = call i32 bitcast (i32 (...)* @rand to i32 ()*)(), !dbg !52
  %24 = load i32*, i32** @A, align 4, !dbg !54
  %25 = load i32, i32* %3, align 4, !dbg !55
  %26 = getelementptr inbounds i32, i32* %24, i32 %25, !dbg !54
  store i32 %23, i32* %26, align 4, !dbg !56
  %27 = call i32 bitcast (i32 (...)* @rand to i32 ()*)(), !dbg !57
  %28 = load i32*, i32** @B, align 4, !dbg !58
  %29 = load i32, i32* %3, align 4, !dbg !59
  %30 = getelementptr inbounds i32, i32* %28, i32 %29, !dbg !58
  store i32 %27, i32* %30, align 4, !dbg !60
  br label %31, !dbg !61

31:                                               ; preds = %22
  %32 = load i32, i32* %3, align 4, !dbg !62
  %33 = add nsw i32 %32, 1, !dbg !62
  store i32 %33, i32* %3, align 4, !dbg !62
  br label %18, !dbg !63, !llvm.loop !64

34:                                               ; preds = %18
  ret void, !dbg !67
}

; Function Attrs: nofree nosync nounwind readnone speculatable willreturn
declare void @llvm.dbg.declare(metadata, metadata, metadata) #1

declare dso_local i32 @srand(...) #2

declare dso_local void @printf(i8*, ...) #2

declare dso_local i8* @malloc(i32) #2

declare dso_local i32 @rand(...) #2

; Function Attrs: noinline nounwind optnone
define dso_local i32 @main() #0 !dbg !68 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  call void @llvm.dbg.declare(metadata i32* %2, metadata !71, metadata !DIExpression()), !dbg !73
  store i32 4, i32* %2, align 4, !dbg !73
  call void @create_test_file(i32 4), !dbg !74
  call void @vector_addition_host(i32 4), !dbg !75
  %3 = load i32*, i32** @A, align 4, !dbg !76
  %4 = bitcast i32* %3 to i8*, !dbg !76
  call void @free(i8* %4), !dbg !77
  %5 = load i32*, i32** @B, align 4, !dbg !78
  %6 = bitcast i32* %5 to i8*, !dbg !78
  call void @free(i8* %6), !dbg !79
  %7 = load i32*, i32** @C, align 4, !dbg !80
  %8 = bitcast i32* %7 to i8*, !dbg !80
  call void @free(i8* %8), !dbg !81
  ret i32 0, !dbg !82
}

; Function Attrs: noinline nounwind optnone
define internal void @vector_addition_host(i32 %0) #0 !dbg !83 {
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  call void @llvm.dbg.declare(metadata i32* %2, metadata !84, metadata !DIExpression()), !dbg !85
  %4 = call i32 bitcast (i32 (...)* @start_region to i32 ()*)(), !dbg !86
  call void @llvm.dbg.declare(metadata i32* %3, metadata !87, metadata !DIExpression()), !dbg !89
  store i32 0, i32* %3, align 4, !dbg !89
  br label %5, !dbg !90

5:                                                ; preds = %22, %1
  %6 = load i32, i32* %3, align 4, !dbg !91
  %7 = load i32, i32* %2, align 4, !dbg !93
  %8 = icmp ult i32 %6, %7, !dbg !94
  br i1 %8, label %9, label %25, !dbg !95

9:                                                ; preds = %5
  %10 = load i32*, i32** @A, align 4, !dbg !96
  %11 = load i32, i32* %3, align 4, !dbg !98
  %12 = getelementptr inbounds i32, i32* %10, i32 %11, !dbg !96
  %13 = load i32, i32* %12, align 4, !dbg !96
  %14 = load i32*, i32** @B, align 4, !dbg !99
  %15 = load i32, i32* %3, align 4, !dbg !100
  %16 = getelementptr inbounds i32, i32* %14, i32 %15, !dbg !99
  %17 = load i32, i32* %16, align 4, !dbg !99
  %18 = add nsw i32 %13, %17, !dbg !101
  %19 = load i32*, i32** @C, align 4, !dbg !102
  %20 = load i32, i32* %3, align 4, !dbg !103
  %21 = getelementptr inbounds i32, i32* %19, i32 %20, !dbg !102
  store i32 %18, i32* %21, align 4, !dbg !104
  br label %22, !dbg !105

22:                                               ; preds = %9
  %23 = load i32, i32* %3, align 4, !dbg !106
  %24 = add nsw i32 %23, 1, !dbg !106
  store i32 %24, i32* %3, align 4, !dbg !106
  br label %5, !dbg !107, !llvm.loop !108

25:                                               ; preds = %5
  %26 = call i32 bitcast (i32 (...)* @end_region to i32 ()*)(), !dbg !110
  ret void, !dbg !111
}

declare dso_local void @free(i8*) #2

declare dso_local i32 @start_region(...) #2

declare dso_local i32 @end_region(...) #2

attributes #0 = { noinline nounwind optnone "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { nofree nosync nounwind readnone speculatable willreturn }
attributes #2 = { "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.dbg.cu = !{!2}
!llvm.module.flags = !{!15, !16, !17}
!llvm.ident = !{!18}

!0 = !DIGlobalVariableExpression(var: !1, expr: !DIExpression())
!1 = distinct !DIGlobalVariable(name: "A", scope: !2, file: !3, line: 19, type: !6, isLocal: true, isDefinition: true)
!2 = distinct !DICompileUnit(language: DW_LANG_C99, file: !3, producer: "clang version 12.0.0 (https://github.com/upmem/llvm-project.git d36425841d9a4d1420b7aa155675f6ae8bcf9f08)", isOptimized: false, runtimeVersion: 0, emissionKind: FullDebug, enums: !4, retainedTypes: !5, globals: !10, splitDebugInlining: false, nameTableKind: None)
!3 = !DIFile(filename: "app_baseline_streamlined.c", directory: "/home/pim-pm/workloads/prim-benchmarks/Benchmarks/VA/baselines/cpu")
!4 = !{}
!5 = !{!6, !9}
!6 = !DIDerivedType(tag: DW_TAG_pointer_type, baseType: !7, size: 32)
!7 = !DIDerivedType(tag: DW_TAG_typedef, name: "int32_t", file: !8, line: 29, baseType: !9)
!8 = !DIFile(filename: "/usr/bin/../share/upmem/include/stdlib/stdint.h", directory: "")
!9 = !DIBasicType(name: "int", size: 32, encoding: DW_ATE_signed)
!10 = !{!0, !11, !13}
!11 = !DIGlobalVariableExpression(var: !12, expr: !DIExpression())
!12 = distinct !DIGlobalVariable(name: "B", scope: !2, file: !3, line: 20, type: !6, isLocal: true, isDefinition: true)
!13 = !DIGlobalVariableExpression(var: !14, expr: !DIExpression())
!14 = distinct !DIGlobalVariable(name: "C", scope: !2, file: !3, line: 21, type: !6, isLocal: true, isDefinition: true)
!15 = !{i32 7, !"Dwarf Version", i32 4}
!16 = !{i32 2, !"Debug Info Version", i32 3}
!17 = !{i32 1, !"wchar_size", i32 1}
!18 = !{!"clang version 12.0.0 (https://github.com/upmem/llvm-project.git d36425841d9a4d1420b7aa155675f6ae8bcf9f08)"}
!19 = distinct !DISubprogram(name: "create_test_file", scope: !3, file: !3, line: 30, type: !20, scopeLine: 30, flags: DIFlagPrototyped, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!20 = !DISubroutineType(types: !21)
!21 = !{null, !22}
!22 = !DIBasicType(name: "unsigned int", size: 32, encoding: DW_ATE_unsigned)
!23 = !DILocalVariable(name: "nr_elements", arg: 1, scope: !19, file: !3, line: 30, type: !22)
!24 = !DILocation(line: 30, column: 36, scope: !19)
!25 = !DILocation(line: 31, column: 5, scope: !19)
!26 = !DILocation(line: 32, column: 33, scope: !19)
!27 = !DILocation(line: 32, column: 5, scope: !19)
!28 = !DILocation(line: 33, column: 27, scope: !19)
!29 = !DILocation(line: 33, column: 39, scope: !19)
!30 = !DILocation(line: 33, column: 20, scope: !19)
!31 = !DILocation(line: 33, column: 9, scope: !19)
!32 = !DILocation(line: 33, column: 7, scope: !19)
!33 = !DILocation(line: 34, column: 27, scope: !19)
!34 = !DILocation(line: 34, column: 39, scope: !19)
!35 = !DILocation(line: 34, column: 20, scope: !19)
!36 = !DILocation(line: 34, column: 9, scope: !19)
!37 = !DILocation(line: 34, column: 7, scope: !19)
!38 = !DILocation(line: 35, column: 27, scope: !19)
!39 = !DILocation(line: 35, column: 39, scope: !19)
!40 = !DILocation(line: 35, column: 20, scope: !19)
!41 = !DILocation(line: 35, column: 9, scope: !19)
!42 = !DILocation(line: 35, column: 7, scope: !19)
!43 = !DILocalVariable(name: "i", scope: !44, file: !3, line: 37, type: !9)
!44 = distinct !DILexicalBlock(scope: !19, file: !3, line: 37, column: 5)
!45 = !DILocation(line: 37, column: 14, scope: !44)
!46 = !DILocation(line: 37, column: 10, scope: !44)
!47 = !DILocation(line: 37, column: 21, scope: !48)
!48 = distinct !DILexicalBlock(scope: !44, file: !3, line: 37, column: 5)
!49 = !DILocation(line: 37, column: 25, scope: !48)
!50 = !DILocation(line: 37, column: 23, scope: !48)
!51 = !DILocation(line: 37, column: 5, scope: !44)
!52 = !DILocation(line: 38, column: 22, scope: !53)
!53 = distinct !DILexicalBlock(scope: !48, file: !3, line: 37, column: 43)
!54 = !DILocation(line: 38, column: 9, scope: !53)
!55 = !DILocation(line: 38, column: 11, scope: !53)
!56 = !DILocation(line: 38, column: 14, scope: !53)
!57 = !DILocation(line: 39, column: 22, scope: !53)
!58 = !DILocation(line: 39, column: 9, scope: !53)
!59 = !DILocation(line: 39, column: 11, scope: !53)
!60 = !DILocation(line: 39, column: 14, scope: !53)
!61 = !DILocation(line: 40, column: 5, scope: !53)
!62 = !DILocation(line: 37, column: 39, scope: !48)
!63 = !DILocation(line: 37, column: 5, scope: !48)
!64 = distinct !{!64, !51, !65, !66}
!65 = !DILocation(line: 40, column: 5, scope: !44)
!66 = !{!"llvm.loop.mustprogress"}
!67 = !DILocation(line: 41, column: 1, scope: !19)
!68 = distinct !DISubprogram(name: "main", scope: !3, file: !3, line: 128, type: !69, scopeLine: 128, spFlags: DISPFlagDefinition, unit: !2, retainedNodes: !4)
!69 = !DISubroutineType(types: !70)
!70 = !{!9}
!71 = !DILocalVariable(name: "file_size", scope: !68, file: !3, line: 133, type: !72)
!72 = !DIDerivedType(tag: DW_TAG_const_type, baseType: !22)
!73 = !DILocation(line: 133, column: 24, scope: !68)
!74 = !DILocation(line: 136, column: 5, scope: !68)
!75 = !DILocation(line: 142, column: 5, scope: !68)
!76 = !DILocation(line: 149, column: 10, scope: !68)
!77 = !DILocation(line: 149, column: 5, scope: !68)
!78 = !DILocation(line: 150, column: 10, scope: !68)
!79 = !DILocation(line: 150, column: 5, scope: !68)
!80 = !DILocation(line: 151, column: 10, scope: !68)
!81 = !DILocation(line: 151, column: 5, scope: !68)
!82 = !DILocation(line: 153, column: 5, scope: !68)
!83 = distinct !DISubprogram(name: "vector_addition_host", scope: !3, file: !3, line: 55, type: !20, scopeLine: 55, flags: DIFlagPrototyped, spFlags: DISPFlagLocalToUnit | DISPFlagDefinition, unit: !2, retainedNodes: !4)
!84 = !DILocalVariable(name: "nr_elements", arg: 1, scope: !83, file: !3, line: 55, type: !22)
!85 = !DILocation(line: 55, column: 47, scope: !83)
!86 = !DILocation(line: 58, column: 5, scope: !83)
!87 = !DILocalVariable(name: "i", scope: !88, file: !3, line: 59, type: !9)
!88 = distinct !DILexicalBlock(scope: !83, file: !3, line: 59, column: 5)
!89 = !DILocation(line: 59, column: 14, scope: !88)
!90 = !DILocation(line: 59, column: 10, scope: !88)
!91 = !DILocation(line: 59, column: 21, scope: !92)
!92 = distinct !DILexicalBlock(scope: !88, file: !3, line: 59, column: 5)
!93 = !DILocation(line: 59, column: 25, scope: !92)
!94 = !DILocation(line: 59, column: 23, scope: !92)
!95 = !DILocation(line: 59, column: 5, scope: !88)
!96 = !DILocation(line: 60, column: 16, scope: !97)
!97 = distinct !DILexicalBlock(scope: !92, file: !3, line: 59, column: 43)
!98 = !DILocation(line: 60, column: 18, scope: !97)
!99 = !DILocation(line: 60, column: 23, scope: !97)
!100 = !DILocation(line: 60, column: 25, scope: !97)
!101 = !DILocation(line: 60, column: 21, scope: !97)
!102 = !DILocation(line: 60, column: 9, scope: !97)
!103 = !DILocation(line: 60, column: 11, scope: !97)
!104 = !DILocation(line: 60, column: 14, scope: !97)
!105 = !DILocation(line: 61, column: 5, scope: !97)
!106 = !DILocation(line: 59, column: 39, scope: !92)
!107 = !DILocation(line: 59, column: 5, scope: !92)
!108 = distinct !{!108, !95, !109, !66}
!109 = !DILocation(line: 61, column: 5, scope: !88)
!110 = !DILocation(line: 62, column: 5, scope: !83)
!111 = !DILocation(line: 63, column: 1, scope: !83)
