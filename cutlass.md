4090算力测试报告

# no sparsity

## fp16 with fp32 accum

```plaintext
root@iZ0jlecu8rgol6m4rd2b23Z-devel:~/repo/cutlass/build/tools/profiler# ./cutlass_profiler  --op_class=tensorop --m=4096 --n=4096 --k=8192 --kernels=cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_nt_align8



=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_nt_align8

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed
          cuBLAS: Not run
           cuDNN: Not run

       Arguments: --gemm_kind=universal --m=4096 --n=4096 --k=8192 --A=f16:column --B=f16:row --C=f16:column --D=f16:column  \
                  --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1 --batch_count=1 --raster_order=heuristic  \
                  --runtime_input_datatype_a=invalid --runtime_input_datatype_b=invalid --use_pdl=false --enable_sm90_mixed_dtype_shuffle_test=false  \
                  --swizzle_size=1 --op_class=tensorop --accum=f32 --cta_m=256 --cta_n=128 --cta_k=32 --cluster_m=1 --cluster_n=1  \
                  --cluster_k=1 --cluster_m_fallback=0 --cluster_n_fallback=0 --cluster_k_fallback=0 --stages=3 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=16 --min_cc=80 --max_cc=1024

           Bytes: 167772160  bytes
           FLOPs: 274911461376  flops
           FLOPs/Byte: 1638

         Runtime: 1.63447  ms
          Memory: 95.5969 GiB/s

            Math: 168196 GFLOP/s


=============================

CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,A,B,C,D,alpha,beta,split_k_mode,split_k_slices,batch_count,raster_order,runtime_input_datatype_a,runtime_input_datatype_b,use_pdl,enable_sm90_mixed_dtype_shuffle_test,swizzle_size,op_class,accum,cta_m,cta_n,cta_k,cluster_m,cluster_n,cluster_k,cluster_m_fallback,cluster_n_fallback,cluster_k_fallback,stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,Flops,Flops/Byte,Runtime,GB/s,GFLOPs
1,CUTLASS,gemm,cutlass_tensorop_f16_s16816gemm_f16_256x128_32x3_nt_align8,passed,success,universal,4096,4096,8192,f16:column,f16:row,f16:column,f16:column,1,0,serial,1,1,heuristic,invalid,invalid,false,false,1,tensorop,f32,256,128,32,1,1,1,0,0,0,3,4,2,1,16,8,16,80,1024,167772160,274911461376,1638,1.63447,95.5969,168196
```

## fp16 with fp16 accum

```plaintext
root@iZ0jlecu8rgol6m4rd2b23Z-devel:~/repo/cutlass/build/tools/profiler# ./cutlass_profiler  --op_class=tensorop --m=4096 --n=4096 --k=8192 --kernels=cutlass_tensorop_h16816gemm_256x128_32x3_nt_align8



=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_tensorop_h16816gemm_256x128_32x3_nt_align8

          Status: Success
    Verification: ON
     Disposition: Passed

reference_device: Passed
          cuBLAS: Not run
           cuDNN: Not run

       Arguments: --gemm_kind=universal --m=4096 --n=4096 --k=8192 --A=f16:column --B=f16:row --C=f16:column --D=f16:column  \
                  --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1 --batch_count=1 --raster_order=heuristic  \
                  --runtime_input_datatype_a=invalid --runtime_input_datatype_b=invalid --use_pdl=false --enable_sm90_mixed_dtype_shuffle_test=false  \
                  --swizzle_size=1 --op_class=tensorop --accum=f16 --cta_m=256 --cta_n=128 --cta_k=32 --cluster_m=1 --cluster_n=1  \
                  --cluster_k=1 --cluster_m_fallback=0 --cluster_n_fallback=0 --cluster_k_fallback=0 --stages=3 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=16 --min_cc=80 --max_cc=1024

           Bytes: 167772160  bytes
           FLOPs: 274911461376  flops
           FLOPs/Byte: 1638

         Runtime: 0.89428  ms
          Memory: 174.722 GiB/s

            Math: 307411 GFLOP/s


=============================

CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,A,B,C,D,alpha,beta,split_k_mode,split_k_slices,batch_count,raster_order,runtime_input_datatype_a,runtime_input_datatype_b,use_pdl,enable_sm90_mixed_dtype_shuffle_test,swizzle_size,op_class,accum,cta_m,cta_n,cta_k,cluster_m,cluster_n,cluster_k,cluster_m_fallback,cluster_n_fallback,cluster_k_fallback,stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,Flops,Flops/Byte,Runtime,GB/s,GFLOPs
1,CUTLASS,gemm,cutlass_tensorop_h16816gemm_256x128_32x3_nt_align8,passed,success,universal,4096,4096,8192,f16:column,f16:row,f16:column,f16:column,1,0,serial,1,1,heuristic,invalid,invalid,false,false,1,tensorop,f16,256,128,32,1,1,1,0,0,0,3,4,2,1,16,8,16,80,1024,167772160,274911461376,1638,0.89428,174.722,307411
```

## fp8 with fp16 accum

```plaintext
root@iZ0jlecu8rgol6m4rd2b23Z-devel:~/repo/cutlass_newest/cutlass/build/tools/profiler# ./cutlass_profiler  --op_class=tensorop --m=4096 --n=4096 --k=8192 --A=fe4m3 --B=fe4m3 --accum=f16 --kernels=cutlass_tensorop_bf16_h16832fastaccumgemm_e4m3_256x128_64x3_tn_align16



=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_tensorop_bf16_h16832fastaccumgemm_e4m3_256x128_64x3_tn_align16

          Status: Success
    Verification: ON
     Disposition: Not verified

reference_device: Not run
          cuBLAS: Not run
           cuDNN: Not run

       Arguments: --gemm_kind=universal --m=4096 --n=4096 --k=8192 --A=fe4m3:row --B=fe4m3:column --C=bf16:column --D=bf16:column  \
                  --alpha=1 --beta=0 --split_k_mode=serial --split_k_slices=1 --batch_count=1 --raster_order=heuristic  \
                  --runtime_input_datatype_a=invalid --runtime_input_datatype_b=invalid --use_pdl=false --enable_sm90_mixed_dtype_shuffle_test=false  \
                  --swizzle_size=1 --op_class=tensorop --accum=f16 --cta_m=256 --cta_n=128 --cta_k=64 --cluster_m=1 --cluster_n=1  \
                  --cluster_k=1 --cluster_m_fallback=0 --cluster_n_fallback=0 --cluster_k_fallback=0 --stages=3 --warps_m=4  \
                  --warps_n=2 --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=32 --min_cc=89 --max_cc=100

           Bytes: 100663296  bytes
           FLOPs: 274911461376  flops
           FLOPs/Byte: 2731

         Runtime: 0.451748  ms
          Memory: 207.527 GiB/s

            Math: 608551 GFLOP/s


=============================

CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,A,B,C,D,alpha,beta,split_k_mode,split_k_slices,batch_count,raster_order,runtime_input_datatype_a,runtime_input_datatype_b,use_pdl,enable_sm90_mixed_dtype_shuffle_test,swizzle_size,op_class,accum,cta_m,cta_n,cta_k,cluster_m,cluster_n,cluster_k,cluster_m_fallback,cluster_n_fallback,cluster_k_fallback,stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,Flops,Flops/Byte,Runtime,GB/s,GFLOPs
1,CUTLASS,gemm,cutlass_tensorop_bf16_h16832fastaccumgemm_e4m3_256x128_64x3_tn_align16,not_verified,success,universal,4096,4096,8192,fe4m3:row,fe4m3:column,bf16:column,bf16:column,1,0,serial,1,1,heuristic,invalid,invalid,false,false,1,tensorop,f16,256,128,64,1,1,1,0,0,0,3,4,2,1,16,8,32,89,100,100663296,274911461376,2731,0.451748,207.527,608551
```

## int8

ncu --kernel-name Kernel2 --launch-skip 73 --launch-count 1 -o int8 --set full --cache-control none --clock-control none "./cutlass_profiler" --op_class=tensorop --m=4096 --n=4096 --k=8192 --A=s8:* --B=s8:* --operation=gemm --kernels=cutlass_tensorop_s8_i16832gemm_s8_256x128_64x3_n32t32_align16

```plaintext
root@iZ0jlecu8rgol6m4rd2b23Z-devel:~/repo/cutlass/build/tools/profiler# ./cutlass_profiler  --op_class=tensorop --m=4096 --n=4096 --k=8192 --A=s8:* --B=s8:* --operation=gemm --kernels=cutlass_tensorop_s8_i16832gemm_s8_256x128_64x3_n32t32_align16



=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: gemm
       Operation: cutlass_tensorop_s8_i16832gemm_s8_256x128_64x3_n32t32_align16

          Status: Success
    Verification: ON
     Disposition: Not verified

reference_device: Not run
          cuBLAS: Not run
           cuDNN: Not run

       Arguments: --gemm_kind=universal --m=4096 --n=4096 --k=8192 --A=s8:nk32 --B=s8:tk32 --C=s8:nk32 --D=s8:nk32 --alpha=1  \
                  --beta=0 --split_k_mode=serial --split_k_slices=1 --batch_count=1 --raster_order=heuristic --runtime_input_datatype_a=invalid  \
                  --runtime_input_datatype_b=invalid --use_pdl=false --enable_sm90_mixed_dtype_shuffle_test=false --swizzle_size=1  \
                  --op_class=tensorop --accum=s32 --cta_m=256 --cta_n=128 --cta_k=64 --cluster_m=1 --cluster_n=1 --cluster_k=1  \
                  --cluster_m_fallback=0 --cluster_n_fallback=0 --cluster_k_fallback=0 --stages=3 --warps_m=4 --warps_n=2  \
                  --warps_k=1 --inst_m=16 --inst_n=8 --inst_k=32 --min_cc=80 --max_cc=1024

           Bytes: 83886080  bytes
           FLOPs: 274911461376  flops
           FLOPs/Byte: 3277

         Runtime: 0.437043  ms
          Memory: 178.758 GiB/s

            Math: 629026 GFLOP/s


=============================

CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,A,B,C,D,alpha,beta,split_k_mode,split_k_slices,batch_count,raster_order,runtime_input_datatype_a,runtime_input_datatype_b,use_pdl,enable_sm90_mixed_dtype_shuffle_test,swizzle_size,op_class,accum,cta_m,cta_n,cta_k,cluster_m,cluster_n,cluster_k,cluster_m_fallback,cluster_n_fallback,cluster_k_fallback,stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,Flops,Flops/Byte,Runtime,GB/s,GFLOPs
1,CUTLASS,gemm,cutlass_tensorop_s8_i16832gemm_s8_256x128_64x3_n32t32_align16,not_verified,success,universal,4096,4096,8192,s8:nk32,s8:tk32,s8:nk32,s8:nk32,1,0,serial,1,1,heuristic,invalid,invalid,false,false,1,tensorop,s32,256,128,64,1,1,1,0,0,0,3,4,2,1,16,8,32,80,1024,83886080,274911461376,3277,0.437043,178.758,629026
```

# sparsity

`./cutlass_profiler  --op_class=tensorop --m=4096 --n=4096 --k=8192 --operation=spgemm`

## fp16 with fp32 accum

```plaintext
root@iZ0jlecu8rgol6m4rd2b23Z-devel:~/repo/cutlass/build/tools/profiler# ./cutlass_profiler  --op_class=tensorop --m=4096 --n=4096 --k=8192 --operation=spgemm --kernels=cutlass_tensorop_f16_s16832spgemm_f16_128x128_64x3_nt_align8



=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: spgemm
       Operation: cutlass_tensorop_f16_s16832spgemm_f16_128x128_64x3_nt_align8

          Status: Success
    Verification: ON
     Disposition: Not verified

reference_device: Not run
          cuBLAS: Not run
           cuDNN: Not run

       Arguments: --gemm_kind=spgemm --m=4096 --n=4096 --k=8192 --A=f16:column --B=f16:row --C=f16:row --E=u16:nk2 --alpha=1  \
                  --beta=0 --split_k_slices=1 --batch_count=1 --op_class=tensorop --accum=f32 --cta_m=128 --cta_n=128  \
                  --cta_k=64 --cluster_m=1 --cluster_n=1 --cluster_k=1 --stages=3 --warps_m=2 --warps_n=2 --warps_k=1  \
                  --inst_m=16 --inst_n=8 --inst_k=32 --min_cc=80 --max_cc=1024

           Bytes: 138412032  bytes
           FLOPs: 274911461376  flops
           FLOPs/Byte: 1986

         Runtime: 0.82135  ms
          Memory: 156.944 GiB/s

            Math: 334707 GFLOP/s


=============================

CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,A,B,C,E,alpha,beta,split_k_slices,batch_count,op_class,accum,cta_m,cta_n,cta_k,cluster_m,cluster_n,cluster_k,cluster_m_fallback,cluster_n_fallback,cluster_k_fallback,stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,Flops,Flops/Byte,Runtime,GB/s,GFLOPs
1,CUTLASS,spgemm,cutlass_tensorop_f16_s16832spgemm_f16_128x128_64x3_nt_align8,not_verified,success,spgemm,4096,4096,8192,f16:column,f16:row,f16:row,u16:nk2,1,0,1,1,tensorop,f32,128,128,64,1,1,1,,,,3,2,2,1,16,8,32,80,1024,138412032,274911461376,1986,0.82135,156.944,334707
```

## fp8 with fp32 accum

```plaintext
root@iZ0jlecu8rgol6m4rd2b23Z-devel:~/repo/cutlass/build/tools/profiler# ./cutlass_profiler  --op_class=tensorop --m=4096 --n=4096 --k=8192 --operation=spgemm --kernels=cutlass_tensorop_s16864spgemm_e4m3_128x128_128x3_tn_align16



=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: spgemm
       Operation: cutlass_tensorop_s16864spgemm_e4m3_128x128_128x3_tn_align16

          Status: Success
    Verification: ON
     Disposition: Not verified

reference_device: Not run
          cuBLAS: Not run
           cuDNN: Not run

       Arguments: --gemm_kind=spgemm --m=4096 --n=4096 --k=8192 --A=fe4m3:row --B=fe4m3:column --C=f32:row --E=u32:nk2  \
                  --alpha=1 --beta=0 --split_k_slices=1 --batch_count=1 --op_class=tensorop --accum=f32 --cta_m=128 --cta_n=128  \
                  --cta_k=128 --cluster_m=1 --cluster_n=1 --cluster_k=1 --stages=3 --warps_m=2 --warps_n=2 --warps_k=1  \
                  --inst_m=16 --inst_n=8 --inst_k=64 --min_cc=89 --max_cc=89

           Bytes: 121634816  bytes
           FLOPs: 274911461376  flops
           FLOPs/Byte: 2260

         Runtime: 0.468879  ms
          Memory: 241.6 GiB/s

            Math: 586316 GFLOP/s


=============================

CSV Results:

Problem,Provider,OperationKind,Operation,Disposition,Status,gemm_kind,m,n,k,A,B,C,E,alpha,beta,split_k_slices,batch_count,op_class,accum,cta_m,cta_n,cta_k,cluster_m,cluster_n,cluster_k,cluster_m_fallback,cluster_n_fallback,cluster_k_fallback,stages,warps_m,warps_n,warps_k,inst_m,inst_n,inst_k,min_cc,max_cc,Bytes,Flops,Flops/Byte,Runtime,GB/s,GFLOPs
1,CUTLASS,spgemm,cutlass_tensorop_s16864spgemm_e4m3_128x128_128x3_tn_align16,not_verified,success,spgemm,4096,4096,8192,fe4m3:row,fe4m3:column,f32:row,u32:nk2,1,0,1,1,tensorop,f32,128,128,128,1,1,1,,,,3,2,2,1,16,8,64,89,89,121634816,274911461376,2260,0.468879,241.6,586316
```

## int8 sparsity

W8A8 int8 sparse gemm算力只能到732，可更新cutlass版本试下
https://github.com/NVIDIA/cutlass/issues/1335

```plaintext
root@iZ0jlecu8rgol6m4rd2b23Z-devel:~/repo/cutlass_newest/cutlass/build/tools/profiler# ./cutlass_profiler  --op_class=tensorop --m=4096 --n=4096 --k=8192 --A=s8:* --B=s8:* --operation=spgemm --kernels=cutlass_tensorop_s8_i16864spgemm_s8_128x128_128x3_tn_align16



=============================
  Problem ID: 1

        Provider: CUTLASS
   OperationKind: spgemm
       Operation: cutlass_tensorop_s8_i16864spgemm_s8_128x128_128x3_tn_align16

          Status: Success
    Verification: ON
     Disposition: Not verified

reference_device: Not run
          cuBLAS: Not run
           cuDNN: Not run

       Arguments: --gemm_kind=spgemm --m=4096 --n=4096 --k=8192 --A=s8:row --B=s8:column --C=s8:row --E=u32:nk2 --alpha=1  \
                  --beta=0 --split_k_slices=1 --batch_count=1 --op_class=tensorop --accum=s32 --cta_m=128 --cta_n=128  \
                  --cta_k=128 --cluster_m=1 --cluster_n=1 --cluster_k=1 --stages=3 --warps_m=2 --warps_n=2 --warps_k=1  \
                  --inst_m=16 --inst_n=8 --inst_k=64 --min_cc=80 --max_cc=1024

           Bytes: 71303168  bytes
           FLOPs: 274911461376  flops
           FLOPs/Byte: 3855

         Runtime: 0.392714  ms
          Memory: 169.096 GiB/s

            Math: 700029 GFLOP/s


=============================
```
