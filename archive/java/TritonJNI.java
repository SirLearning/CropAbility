// TritonJNI.java - Java 端 JNI 接口
package com.example.triton;

/**
 * JNI接口类，用于与C++原生库交互
 */
public class TritonJNI {
    
    // 加载本地库
    static {
        try {
            System.loadLibrary("triton_jni");
            System.out.println("Successfully loaded triton_jni library");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Failed to load triton_jni library: " + e.getMessage());
            throw e;
        }
    }
    
    /**
     * 加载 TorchScript 模型
     * @param modelPath 模型文件路径
     * @return 模型指针，如果加载失败返回0
     */
    public native long loadModel(String modelPath);
    
    /**
     * 运行 Triton kernel
     * @param modelPtr 模型指针
     * @param input1 第一个输入数组
     * @param input2 第二个输入数组
     * @param size 数组大小
     * @return 计算结果数组
     */
    public native float[] runTritonKernel(long modelPtr, float[] input1, float[] input2, int size);
    
    /**
     * 释放模型资源
     * @param modelPtr 模型指针
     */
    public native void releaseModel(long modelPtr);
}
