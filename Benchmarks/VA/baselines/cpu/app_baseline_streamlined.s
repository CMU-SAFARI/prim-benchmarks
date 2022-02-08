	.text
	.file	"app_baseline_streamlined.c"
	.file	1 "/usr/bin/../share/upmem/include/stdlib" "stdint.h"
	.file	2 "/home/joao/Git/PiM-Performance-Model/workloads/prim-benchmarks/Benchmarks/VA/baselines/cpu" "app_baseline_streamlined.c"
	.section	.text.create_test_file,"ax",@progbits
	.globl	create_test_file                // -- Begin function create_test_file
	.type	create_test_file,@function
create_test_file:                       // @create_test_file
.Lfunc_begin0:
	.loc	2 30 0                          // app_baseline_streamlined.c:30:0
	.cfi_sections .debug_frame
	.cfi_startproc
// %bb.0:
	.cfi_def_cfa_offset -24
	.cfi_offset 23, -8
	.cfi_offset 22, -4
	sd r22, 16, d22
	add r22, r22, 24
	sw r22, -24, r0
	move r0, 0
.Ltmp0:
	.loc	2 31 5 prologue_end             // app_baseline_streamlined.c:31:5
	call r23, srand
	.loc	2 32 33                         // app_baseline_streamlined.c:32:33
	lw r1, r22, -24
	.loc	2 32 5 is_stmt 0                // app_baseline_streamlined.c:32:5
	move r0, r22
	sw r0, -12, r1
	move r0, .L.str
	call r23, printf
	.loc	2 33 27 is_stmt 1               // app_baseline_streamlined.c:33:27
	lw r0, r22, -24
	.loc	2 33 39 is_stmt 0               // app_baseline_streamlined.c:33:39
	lsl r0, r0, 2
	.loc	2 33 20                         // app_baseline_streamlined.c:33:20
	call r23, malloc
	.loc	2 33 7                          // app_baseline_streamlined.c:33:7
	sw zero, A, r0
	.loc	2 34 27 is_stmt 1               // app_baseline_streamlined.c:34:27
	lw r0, r22, -24
	.loc	2 34 39 is_stmt 0               // app_baseline_streamlined.c:34:39
	lsl r0, r0, 2
	.loc	2 34 20                         // app_baseline_streamlined.c:34:20
	call r23, malloc
	.loc	2 34 7                          // app_baseline_streamlined.c:34:7
	sw zero, B, r0
	.loc	2 35 27 is_stmt 1               // app_baseline_streamlined.c:35:27
	lw r0, r22, -24
	.loc	2 35 39 is_stmt 0               // app_baseline_streamlined.c:35:39
	lsl r0, r0, 2
	.loc	2 35 20                         // app_baseline_streamlined.c:35:20
	call r23, malloc
	.loc	2 35 7                          // app_baseline_streamlined.c:35:7
	sw zero, C, r0
.Ltmp1:
	.loc	2 37 14 is_stmt 1               // app_baseline_streamlined.c:37:14
	sw r22, -20, 0
	.loc	2 37 10 is_stmt 0               // app_baseline_streamlined.c:37:10
	jump .LBB0_1
.LBB0_1:                                // =>This Inner Loop Header: Depth=1
.Ltmp2:
	.loc	2 37 21                         // app_baseline_streamlined.c:37:21
	lw r0, r22, -20
	.loc	2 37 25                         // app_baseline_streamlined.c:37:25
	lw r1, r22, -24
.Ltmp3:
	.loc	2 37 5                          // app_baseline_streamlined.c:37:5
	jgeu r0, r1, .LBB0_4
	jump .LBB0_2
.LBB0_2:                                //   in Loop: Header=BB0_1 Depth=1
.Ltmp4:
	.loc	2 38 22 is_stmt 1               // app_baseline_streamlined.c:38:22
	call r23, rand
	move r1, r0
	.loc	2 38 9 is_stmt 0                // app_baseline_streamlined.c:38:9
	lw r0, zero, A
	.loc	2 38 11                         // app_baseline_streamlined.c:38:11
	lw r2, r22, -20
	.loc	2 38 9                          // app_baseline_streamlined.c:38:9
	lsl_add r0, r0, r2, 2
	.loc	2 38 14                         // app_baseline_streamlined.c:38:14
	sw r0, 0, r1
	.loc	2 39 22 is_stmt 1               // app_baseline_streamlined.c:39:22
	call r23, rand
	move r1, r0
	.loc	2 39 9 is_stmt 0                // app_baseline_streamlined.c:39:9
	lw r0, zero, B
	.loc	2 39 11                         // app_baseline_streamlined.c:39:11
	lw r2, r22, -20
	.loc	2 39 9                          // app_baseline_streamlined.c:39:9
	lsl_add r0, r0, r2, 2
	.loc	2 39 14                         // app_baseline_streamlined.c:39:14
	sw r0, 0, r1
	.loc	2 40 5 is_stmt 1                // app_baseline_streamlined.c:40:5
	jump .LBB0_3
.Ltmp5:
.LBB0_3:                                //   in Loop: Header=BB0_1 Depth=1
	.loc	2 37 39                         // app_baseline_streamlined.c:37:39
	lw r0, r22, -20
	add r0, r0, 1
	sw r22, -20, r0
	.loc	2 37 5 is_stmt 0                // app_baseline_streamlined.c:37:5
	jump .LBB0_1
.Ltmp6:
.LBB0_4:
	.loc	2 41 1 is_stmt 1                // app_baseline_streamlined.c:41:1
	ld d22, r22, -8
	jump r23
.Ltmp7:
.Lfunc_end0:
	.size	create_test_file, .Lfunc_end0-create_test_file
	.cfi_endproc
	.section	.stack_sizes,"o",@progbits,.text.create_test_file
	.long	.Lfunc_begin0
	.byte	24
	.section	.text.create_test_file,"ax",@progbits
                                        // -- End function
	.section	.text.main,"ax",@progbits
	.globl	main                            // -- Begin function main
	.type	main,@function
main:                                   // @main
.Lfunc_begin1:
	.loc	2 128 0                         // app_baseline_streamlined.c:128:0
	.cfi_startproc
// %bb.0:
	.cfi_def_cfa_offset -24
	.cfi_offset 23, -8
	.cfi_offset 22, -4
	sd r22, 16, d22
	add r22, r22, 24
	sw r22, -24, 0
.Ltmp8:
	.loc	2 133 24 prologue_end           // app_baseline_streamlined.c:133:24
	sw r22, -20, 4
	move r0, 4
	sw r22, -16, r0
	.loc	2 136 5                         // app_baseline_streamlined.c:136:5
	call r23, create_test_file
	lw r0, r22, -16
	.loc	2 142 5                         // app_baseline_streamlined.c:142:5
	call r23, vector_addition_host
	.loc	2 149 10                        // app_baseline_streamlined.c:149:10
	lw r0, zero, A
	.loc	2 149 5 is_stmt 0               // app_baseline_streamlined.c:149:5
	call r23, free
	.loc	2 150 10 is_stmt 1              // app_baseline_streamlined.c:150:10
	lw r0, zero, B
	.loc	2 150 5 is_stmt 0               // app_baseline_streamlined.c:150:5
	call r23, free
	.loc	2 151 10 is_stmt 1              // app_baseline_streamlined.c:151:10
	lw r0, zero, C
	.loc	2 151 5 is_stmt 0               // app_baseline_streamlined.c:151:5
	call r23, free
	move r0, 0
	.loc	2 153 5 is_stmt 1               // app_baseline_streamlined.c:153:5
	ld d22, r22, -8
	jump r23
.Ltmp9:
.Lfunc_end1:
	.size	main, .Lfunc_end1-main
	.cfi_endproc
	.section	.stack_sizes,"o",@progbits,.text.main
	.long	.Lfunc_begin1
	.byte	24
	.section	.text.main,"ax",@progbits
                                        // -- End function
	.section	.text.vector_addition_host,"ax",@progbits
	.type	vector_addition_host,@function  // -- Begin function vector_addition_host
vector_addition_host:                   // @vector_addition_host
.Lfunc_begin2:
	.loc	2 55 0                          // app_baseline_streamlined.c:55:0
	.cfi_startproc
// %bb.0:
	.cfi_def_cfa_offset -16
	.cfi_offset 23, -8
	.cfi_offset 22, -4
	sd r22, 8, d22
	add r22, r22, 16
	sw r22, -16, r0
.Ltmp10:
	.loc	2 58 5 prologue_end             // app_baseline_streamlined.c:58:5
	call r23, start_region
.Ltmp11:
	.loc	2 59 14                         // app_baseline_streamlined.c:59:14
	sw r22, -12, 0
	.loc	2 59 10 is_stmt 0               // app_baseline_streamlined.c:59:10
	jump .LBB2_1
.LBB2_1:                                // =>This Inner Loop Header: Depth=1
.Ltmp12:
	.loc	2 59 21                         // app_baseline_streamlined.c:59:21
	lw r0, r22, -12
	.loc	2 59 25                         // app_baseline_streamlined.c:59:25
	lw r1, r22, -16
.Ltmp13:
	.loc	2 59 5                          // app_baseline_streamlined.c:59:5
	jgeu r0, r1, .LBB2_4
	jump .LBB2_2
.LBB2_2:                                //   in Loop: Header=BB2_1 Depth=1
.Ltmp14:
	.loc	2 60 16 is_stmt 1               // app_baseline_streamlined.c:60:16
	lw r0, zero, A
	.loc	2 60 18 is_stmt 0               // app_baseline_streamlined.c:60:18
	lw r2, r22, -12
	.loc	2 60 16                         // app_baseline_streamlined.c:60:16
	lsl_add r0, r0, r2, 2
	lw r0, r0, 0
	.loc	2 60 23                         // app_baseline_streamlined.c:60:23
	lw r1, zero, B
	lsl_add r1, r1, r2, 2
	lw r1, r1, 0
	.loc	2 60 21                         // app_baseline_streamlined.c:60:21
	add r1, r0, r1
	.loc	2 60 9                          // app_baseline_streamlined.c:60:9
	lw r0, zero, C
	lsl_add r0, r0, r2, 2
	.loc	2 60 14                         // app_baseline_streamlined.c:60:14
	sw r0, 0, r1
	.loc	2 61 5 is_stmt 1                // app_baseline_streamlined.c:61:5
	jump .LBB2_3
.Ltmp15:
.LBB2_3:                                //   in Loop: Header=BB2_1 Depth=1
	.loc	2 59 39                         // app_baseline_streamlined.c:59:39
	lw r0, r22, -12
	add r0, r0, 1
	sw r22, -12, r0
	.loc	2 59 5 is_stmt 0                // app_baseline_streamlined.c:59:5
	jump .LBB2_1
.Ltmp16:
.LBB2_4:
	.loc	2 62 5 is_stmt 1                // app_baseline_streamlined.c:62:5
	call r23, end_region
	.loc	2 63 1                          // app_baseline_streamlined.c:63:1
	ld d22, r22, -8
	jump r23
.Ltmp17:
.Lfunc_end2:
	.size	vector_addition_host, .Lfunc_end2-vector_addition_host
	.cfi_endproc
	.section	.stack_sizes,"o",@progbits,.text.vector_addition_host
	.long	.Lfunc_begin2
	.byte	16
	.section	.text.vector_addition_host,"ax",@progbits
                                        // -- End function
	.type	.L.str,@object                  // @.str
	.section	.rodata.str1.1,"aMS",@progbits,1
.L.str:
	.asciz	"nr_elements\t%u\t"
	.size	.L.str, 16

	.type	A,@object                       // @A
	.section	.bss.A,"aw",@nobits
	.p2align	2
A:
	.long	0
	.size	A, 4

	.type	B,@object                       // @B
	.section	.bss.B,"aw",@nobits
	.p2align	2
B:
	.long	0
	.size	B, 4

	.type	C,@object                       // @C
	.section	.bss.C,"aw",@nobits
	.p2align	2
C:
	.long	0
	.size	C, 4

	.section	.debug_abbrev,"",@progbits
	.byte	1                               // Abbreviation Code
	.byte	17                              // DW_TAG_compile_unit
	.byte	1                               // DW_CHILDREN_yes
	.byte	37                              // DW_AT_producer
	.byte	14                              // DW_FORM_strp
	.byte	19                              // DW_AT_language
	.byte	5                               // DW_FORM_data2
	.byte	3                               // DW_AT_name
	.byte	14                              // DW_FORM_strp
	.byte	16                              // DW_AT_stmt_list
	.byte	23                              // DW_FORM_sec_offset
	.byte	27                              // DW_AT_comp_dir
	.byte	14                              // DW_FORM_strp
	.byte	17                              // DW_AT_low_pc
	.byte	1                               // DW_FORM_addr
	.byte	85                              // DW_AT_ranges
	.byte	23                              // DW_FORM_sec_offset
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	2                               // Abbreviation Code
	.byte	52                              // DW_TAG_variable
	.byte	0                               // DW_CHILDREN_no
	.byte	3                               // DW_AT_name
	.byte	14                              // DW_FORM_strp
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	2                               // DW_AT_location
	.byte	24                              // DW_FORM_exprloc
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	3                               // Abbreviation Code
	.byte	15                              // DW_TAG_pointer_type
	.byte	0                               // DW_CHILDREN_no
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	4                               // Abbreviation Code
	.byte	22                              // DW_TAG_typedef
	.byte	0                               // DW_CHILDREN_no
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	3                               // DW_AT_name
	.byte	14                              // DW_FORM_strp
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	5                               // Abbreviation Code
	.byte	36                              // DW_TAG_base_type
	.byte	0                               // DW_CHILDREN_no
	.byte	3                               // DW_AT_name
	.byte	14                              // DW_FORM_strp
	.byte	62                              // DW_AT_encoding
	.byte	11                              // DW_FORM_data1
	.byte	11                              // DW_AT_byte_size
	.byte	11                              // DW_FORM_data1
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	6                               // Abbreviation Code
	.byte	46                              // DW_TAG_subprogram
	.byte	1                               // DW_CHILDREN_yes
	.byte	17                              // DW_AT_low_pc
	.byte	1                               // DW_FORM_addr
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.byte	64                              // DW_AT_frame_base
	.byte	24                              // DW_FORM_exprloc
	.byte	3                               // DW_AT_name
	.byte	14                              // DW_FORM_strp
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	39                              // DW_AT_prototyped
	.byte	25                              // DW_FORM_flag_present
	.byte	63                              // DW_AT_external
	.byte	25                              // DW_FORM_flag_present
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	7                               // Abbreviation Code
	.byte	5                               // DW_TAG_formal_parameter
	.byte	0                               // DW_CHILDREN_no
	.byte	2                               // DW_AT_location
	.byte	24                              // DW_FORM_exprloc
	.byte	3                               // DW_AT_name
	.byte	14                              // DW_FORM_strp
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	8                               // Abbreviation Code
	.byte	11                              // DW_TAG_lexical_block
	.byte	1                               // DW_CHILDREN_yes
	.byte	17                              // DW_AT_low_pc
	.byte	1                               // DW_FORM_addr
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	9                               // Abbreviation Code
	.byte	52                              // DW_TAG_variable
	.byte	0                               // DW_CHILDREN_no
	.byte	2                               // DW_AT_location
	.byte	24                              // DW_FORM_exprloc
	.byte	3                               // DW_AT_name
	.byte	14                              // DW_FORM_strp
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	10                              // Abbreviation Code
	.byte	46                              // DW_TAG_subprogram
	.byte	1                               // DW_CHILDREN_yes
	.byte	17                              // DW_AT_low_pc
	.byte	1                               // DW_FORM_addr
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.byte	64                              // DW_AT_frame_base
	.byte	24                              // DW_FORM_exprloc
	.byte	3                               // DW_AT_name
	.byte	14                              // DW_FORM_strp
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	63                              // DW_AT_external
	.byte	25                              // DW_FORM_flag_present
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	11                              // Abbreviation Code
	.byte	46                              // DW_TAG_subprogram
	.byte	1                               // DW_CHILDREN_yes
	.byte	17                              // DW_AT_low_pc
	.byte	1                               // DW_FORM_addr
	.byte	18                              // DW_AT_high_pc
	.byte	6                               // DW_FORM_data4
	.byte	64                              // DW_AT_frame_base
	.byte	24                              // DW_FORM_exprloc
	.byte	3                               // DW_AT_name
	.byte	14                              // DW_FORM_strp
	.byte	58                              // DW_AT_decl_file
	.byte	11                              // DW_FORM_data1
	.byte	59                              // DW_AT_decl_line
	.byte	11                              // DW_FORM_data1
	.byte	39                              // DW_AT_prototyped
	.byte	25                              // DW_FORM_flag_present
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	12                              // Abbreviation Code
	.byte	38                              // DW_TAG_const_type
	.byte	0                               // DW_CHILDREN_no
	.byte	73                              // DW_AT_type
	.byte	19                              // DW_FORM_ref4
	.byte	0                               // EOM(1)
	.byte	0                               // EOM(2)
	.byte	0                               // EOM(3)
	.section	.debug_info,"",@progbits
.Lcu_begin0:
	.long	.Ldebug_info_end0-.Ldebug_info_start0 // Length of Unit
.Ldebug_info_start0:
	.short	4                               // DWARF version number
	.long	.debug_abbrev                   // Offset Into Abbrev. Section
	.byte	4                               // Address Size (in bytes)
	.byte	1                               // Abbrev [1] 0xb:0x106 DW_TAG_compile_unit
	.long	.Linfo_string0                  // DW_AT_producer
	.short	12                              // DW_AT_language
	.long	.Linfo_string1                  // DW_AT_name
	.long	.Lline_table_start0             // DW_AT_stmt_list
	.long	.Linfo_string2                  // DW_AT_comp_dir
	.long	0                               // DW_AT_low_pc
	.long	.Ldebug_ranges0                 // DW_AT_ranges
	.byte	2                               // Abbrev [2] 0x26:0x11 DW_TAG_variable
	.long	.Linfo_string3                  // DW_AT_name
	.long	55                              // DW_AT_type
	.byte	2                               // DW_AT_decl_file
	.byte	19                              // DW_AT_decl_line
	.byte	5                               // DW_AT_location
	.byte	3
	.long	A
	.byte	3                               // Abbrev [3] 0x37:0x5 DW_TAG_pointer_type
	.long	60                              // DW_AT_type
	.byte	4                               // Abbrev [4] 0x3c:0xb DW_TAG_typedef
	.long	71                              // DW_AT_type
	.long	.Linfo_string5                  // DW_AT_name
	.byte	1                               // DW_AT_decl_file
	.byte	29                              // DW_AT_decl_line
	.byte	5                               // Abbrev [5] 0x47:0x7 DW_TAG_base_type
	.long	.Linfo_string4                  // DW_AT_name
	.byte	5                               // DW_AT_encoding
	.byte	4                               // DW_AT_byte_size
	.byte	2                               // Abbrev [2] 0x4e:0x11 DW_TAG_variable
	.long	.Linfo_string6                  // DW_AT_name
	.long	55                              // DW_AT_type
	.byte	2                               // DW_AT_decl_file
	.byte	20                              // DW_AT_decl_line
	.byte	5                               // DW_AT_location
	.byte	3
	.long	B
	.byte	2                               // Abbrev [2] 0x5f:0x11 DW_TAG_variable
	.long	.Linfo_string7                  // DW_AT_name
	.long	55                              // DW_AT_type
	.byte	2                               // DW_AT_decl_file
	.byte	21                              // DW_AT_decl_line
	.byte	5                               // DW_AT_location
	.byte	3
	.long	C
	.byte	6                               // Abbrev [6] 0x70:0x38 DW_TAG_subprogram
	.long	.Lfunc_begin0                   // DW_AT_low_pc
	.long	.Lfunc_end0-.Lfunc_begin0       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	102
	.long	.Linfo_string8                  // DW_AT_name
	.byte	2                               // DW_AT_decl_file
	.byte	30                              // DW_AT_decl_line
                                        // DW_AT_prototyped
                                        // DW_AT_external
	.byte	7                               // Abbrev [7] 0x81:0xe DW_TAG_formal_parameter
	.byte	2                               // DW_AT_location
	.byte	145
	.byte	104
	.long	.Linfo_string11                 // DW_AT_name
	.byte	2                               // DW_AT_decl_file
	.byte	30                              // DW_AT_decl_line
	.long	260                             // DW_AT_type
	.byte	8                               // Abbrev [8] 0x8f:0x18 DW_TAG_lexical_block
	.long	.Ltmp1                          // DW_AT_low_pc
	.long	.Ltmp6-.Ltmp1                   // DW_AT_high_pc
	.byte	9                               // Abbrev [9] 0x98:0xe DW_TAG_variable
	.byte	2                               // DW_AT_location
	.byte	145
	.byte	108
	.long	.Linfo_string13                 // DW_AT_name
	.byte	2                               // DW_AT_decl_file
	.byte	37                              // DW_AT_decl_line
	.long	71                              // DW_AT_type
	.byte	0                               // End Of Children Mark
	.byte	0                               // End Of Children Mark
	.byte	10                              // Abbrev [10] 0xa8:0x24 DW_TAG_subprogram
	.long	.Lfunc_begin1                   // DW_AT_low_pc
	.long	.Lfunc_end1-.Lfunc_begin1       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	102
	.long	.Linfo_string9                  // DW_AT_name
	.byte	2                               // DW_AT_decl_file
	.byte	128                             // DW_AT_decl_line
	.long	71                              // DW_AT_type
                                        // DW_AT_external
	.byte	9                               // Abbrev [9] 0xbd:0xe DW_TAG_variable
	.byte	2                               // DW_AT_location
	.byte	145
	.byte	108
	.long	.Linfo_string14                 // DW_AT_name
	.byte	2                               // DW_AT_decl_file
	.byte	133                             // DW_AT_decl_line
	.long	267                             // DW_AT_type
	.byte	0                               // End Of Children Mark
	.byte	11                              // Abbrev [11] 0xcc:0x38 DW_TAG_subprogram
	.long	.Lfunc_begin2                   // DW_AT_low_pc
	.long	.Lfunc_end2-.Lfunc_begin2       // DW_AT_high_pc
	.byte	1                               // DW_AT_frame_base
	.byte	102
	.long	.Linfo_string10                 // DW_AT_name
	.byte	2                               // DW_AT_decl_file
	.byte	55                              // DW_AT_decl_line
                                        // DW_AT_prototyped
	.byte	7                               // Abbrev [7] 0xdd:0xe DW_TAG_formal_parameter
	.byte	2                               // DW_AT_location
	.byte	145
	.byte	112
	.long	.Linfo_string11                 // DW_AT_name
	.byte	2                               // DW_AT_decl_file
	.byte	55                              // DW_AT_decl_line
	.long	260                             // DW_AT_type
	.byte	8                               // Abbrev [8] 0xeb:0x18 DW_TAG_lexical_block
	.long	.Ltmp11                         // DW_AT_low_pc
	.long	.Ltmp16-.Ltmp11                 // DW_AT_high_pc
	.byte	9                               // Abbrev [9] 0xf4:0xe DW_TAG_variable
	.byte	2                               // DW_AT_location
	.byte	145
	.byte	116
	.long	.Linfo_string13                 // DW_AT_name
	.byte	2                               // DW_AT_decl_file
	.byte	59                              // DW_AT_decl_line
	.long	71                              // DW_AT_type
	.byte	0                               // End Of Children Mark
	.byte	0                               // End Of Children Mark
	.byte	5                               // Abbrev [5] 0x104:0x7 DW_TAG_base_type
	.long	.Linfo_string12                 // DW_AT_name
	.byte	7                               // DW_AT_encoding
	.byte	4                               // DW_AT_byte_size
	.byte	12                              // Abbrev [12] 0x10b:0x5 DW_TAG_const_type
	.long	260                             // DW_AT_type
	.byte	0                               // End Of Children Mark
.Ldebug_info_end0:
	.section	.debug_ranges,"",@progbits
.Ldebug_ranges0:
	.long	.Lfunc_begin0
	.long	.Lfunc_end0
	.long	.Lfunc_begin1
	.long	.Lfunc_end1
	.long	.Lfunc_begin2
	.long	.Lfunc_end2
	.long	0
	.long	0
	.section	.debug_str,"MS",@progbits,1
.Linfo_string0:
	.asciz	"clang version 12.0.0 (https://github.com/upmem/llvm-project.git d36425841d9a4d1420b7aa155675f6ae8bcf9f08)" // string offset=0
.Linfo_string1:
	.asciz	"app_baseline_streamlined.c"    // string offset=106
.Linfo_string2:
	.asciz	"/home/joao/Git/PiM-Performance-Model/workloads/prim-benchmarks/Benchmarks/VA/baselines/cpu" // string offset=133
.Linfo_string3:
	.asciz	"A"                             // string offset=224
.Linfo_string4:
	.asciz	"int"                           // string offset=226
.Linfo_string5:
	.asciz	"int32_t"                       // string offset=230
.Linfo_string6:
	.asciz	"B"                             // string offset=238
.Linfo_string7:
	.asciz	"C"                             // string offset=240
.Linfo_string8:
	.asciz	"create_test_file"              // string offset=242
.Linfo_string9:
	.asciz	"main"                          // string offset=259
.Linfo_string10:
	.asciz	"vector_addition_host"          // string offset=264
.Linfo_string11:
	.asciz	"nr_elements"                   // string offset=285
.Linfo_string12:
	.asciz	"unsigned int"                  // string offset=297
.Linfo_string13:
	.asciz	"i"                             // string offset=310
.Linfo_string14:
	.asciz	"file_size"                     // string offset=312
	.addrsig
	.addrsig_sym create_test_file
	.addrsig_sym srand
	.addrsig_sym printf
	.addrsig_sym malloc
	.addrsig_sym rand
	.addrsig_sym vector_addition_host
	.addrsig_sym free
	.addrsig_sym start_region
	.addrsig_sym end_region
	.addrsig_sym A
	.addrsig_sym B
	.addrsig_sym C
	.section	.debug_line,"",@progbits
.Lline_table_start0:
