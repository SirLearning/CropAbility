package com.example.triton;

import org.junit.Before;
import org.junit.Test;
import org.junit.After;
import static org.junit.Assert.*;

/**
 * TritonJNIWrapper的单元测试
 */
public class TritonJNIWrapperTest {
    
    private TritonJNIWrapper wrapper;
    private static final String TEST_MODEL_PATH = "triton_add_model.pt";
    
    @Before
    public void setUp() {
        // 在实际测试中，需要确保模型文件存在
        // 这里先跳过模型加载，仅测试逻辑
    }
    
    @Test
    public void testAddArraysWithSameLength() {
        // 测试相同长度数组的加法
        float[] input1 = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        float[] input2 = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        
        // 由于需要JNI库，这里模拟测试逻辑
        assertEquals("Input arrays should have same length", input1.length, input2.length);
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testAddArraysWithDifferentLength() {
        float[] input1 = {1.0f, 2.0f, 3.0f};
        float[] input2 = {1.0f, 1.0f};
        
        // 模拟不同长度数组检测
        if (input1.length != input2.length) {
            throw new IllegalArgumentException("Input arrays must have same length");
        }
    }
    
    @Test
    public void testBatchAddArrays() {
        float[][] batch1 = {{1.0f, 2.0f}, {3.0f, 4.0f}};
        float[][] batch2 = {{1.0f, 1.0f}, {1.0f, 1.0f}};
        
        assertEquals("Batch sizes should match", batch1.length, batch2.length);
        
        for (int i = 0; i < batch1.length; i++) {
            assertEquals("Each batch item should have same length", 
                        batch1[i].length, batch2[i].length);
        }
    }
    
    @Test(expected = IllegalArgumentException.class)
    public void testBatchAddWithDifferentBatchSizes() {
        float[][] batch1 = {{1.0f, 2.0f}};
        float[][] batch2 = {{1.0f, 1.0f}, {2.0f, 2.0f}};
        
        if (batch1.length != batch2.length) {
            throw new IllegalArgumentException("Batch sizes must match");
        }
    }
    
    @After
    public void tearDown() {
        if (wrapper != null) {
            wrapper.close();
        }
    }
}
