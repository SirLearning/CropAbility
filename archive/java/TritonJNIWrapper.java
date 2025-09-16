// TritonJNIWrapper.java - 高级封装类
package com.example.triton;

public class TritonJNIWrapper implements AutoCloseable {
    
    private long modelPtr;
    private TritonJNI jni;
    
    public TritonJNIWrapper(String modelPath) {
        this.jni = new TritonJNI();
        this.modelPtr = jni.loadModel(modelPath);
        
        if (modelPtr == 0) {
            throw new RuntimeException("Failed to load Triton model: " + modelPath);
        }
    }
    
    /**
     * 执行 Triton 加法运算
     */
    public float[] add(float[] input1, float[] input2) {
        if (input1.length != input2.length) {
            throw new IllegalArgumentException("Input arrays must have same length");
        }
        
        return jni.runTritonKernel(modelPtr, input1, input2, input1.length);
    }
    
    /**
     * 批量处理
     */
    public float[][] batchAdd(float[][] batch1, float[][] batch2) {
        if (batch1.length != batch2.length) {
            throw new IllegalArgumentException("Batch sizes must match");
        }
        
        float[][] results = new float[batch1.length][];
        for (int i = 0; i < batch1.length; i++) {
            results[i] = add(batch1[i], batch2[i]);
        }
        return results;
    }
    
    @Override
    public void close() {
        if (modelPtr != 0) {
            jni.releaseModel(modelPtr);
            modelPtr = 0;
        }
    }
    
    // 测试方法
    public static void main(String[] args) {
        String modelPath = "triton_add_model.pt";
        
        try (TritonJNIWrapper triton = new TritonJNIWrapper(modelPath)) {
            
            // 创建测试数据
            int size = 1000;
            float[] input1 = new float[size];
            float[] input2 = new float[size];
            
            for (int i = 0; i < size; i++) {
                input1[i] = (float) Math.random();
                input2[i] = (float) Math.random();
            }
            
            // 运行 Triton kernel
            long startTime = System.nanoTime();
            float[] result = triton.add(input1, input2);
            long endTime = System.nanoTime();
            
            System.out.printf("Triton kernel 执行时间: %.2f ms%n", 
                            (endTime - startTime) / 1_000_000.0);
            
            // 验证结果
            boolean correct = true;
            for (int i = 0; i < Math.min(10, size); i++) {
                float expected = input1[i] + input2[i];
                if (Math.abs(result[i] - expected) > 1e-6) {
                    correct = false;
                    break;
                }
            }
            
            System.out.println("结果验证: " + (correct ? "通过" : "失败"));
            
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
